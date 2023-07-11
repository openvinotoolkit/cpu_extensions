// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <climits>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <sstream>

namespace llmdnn {

#define PLAINTENSOR_RANK_MAX 8
struct plain_tensor_base {
    size_t m_strides[PLAINTENSOR_RANK_MAX];
    size_t m_dims[PLAINTENSOR_RANK_MAX];
    size_t m_rank;

    std::shared_ptr<void> m_ptr;
    size_t m_capacity = 0;
    size_t m_element_size = 1;

    uint8_t* batched_ptr_buff[8];
    std::vector<uint8_t*> batched_ptr_backup;

    operator bool() {
        return static_cast<bool>(m_ptr);
    }

    size_t size(int i) {
        assert(i < m_rank);
        return m_dims[i];
    }
    size_t stride(int i) {
        assert(i < m_rank);
        return m_strides[i];
    }
};

struct plain_tensor : public plain_tensor_base {
    plain_tensor();
    ~plain_tensor();

    struct tensor_index {
        int start;
        int end;
        int step;
        int count;
        // select all
        tensor_index() {
            start = 0;
            end = INT_MAX;
            step = 1;
        }
        bool slice_with_squeeze() {
            return end == INT_MIN;
        }
        // tensor_index(start)            : select 1 element (with squeeze)
        // tensor_index(start, end, step) : select a range w/o squeeze
        tensor_index(int start, int end = INT_MIN, int step = 1) : start(start), end(end), step(step) {}

        void regularize(int size) {
            if (start < 0)
                start += size;
            assert(start >= 0 && start < size);
            if (end != INT_MIN) {
                if (end < 0)
                    end += size;
                if (end > size)
                    end = size;
                assert(end >= 0 && end <= size);
                count = (end - start + step - 1) / step;
            } else {
                count = 1;
            }
        }
    };

    plain_tensor index(const std::initializer_list<tensor_index>& indices) const;

    // slice: return a sub-view (w/o ownership/refcount to original data)
    plain_tensor slice(int axis, int start, int end) const;

    bool is_dense() const;

    /*
       suppose current shape is [a0,a1,...,am]
       and target shape is [b0,b1,...,bn]
       reshape is only valid when (a0*a1*...*am) == (b0*b1*...*bn) <======= (A)

       uniform a tensor's shape into groups from last to first, the dimension is merged
       into current group if the subtensor in the group is still dense after merge.
       otherwise a new group is formed.

       then reshape is performed on group basis, the check (A) is performed on group bases.
       which means any reshape inside the group is OK, but not across the group boundary.

       this can be done in one-loop, while group is forming, and checks are performed.

       simplified form is when whole tensor is dense
    */
    plain_tensor reshape(const std::initializer_list<size_t>& target_shape) const;

    plain_tensor permute(const std::initializer_list<size_t>& order) const;

    template<typename DT>
    void resize(const std::vector<size_t>& new_dims, DT* data = nullptr) {
        resize(new_dims, data, sizeof(DT));
    }

    void resize(const std::vector<size_t>& new_dims, void* data, size_t element_size);

    template<typename DT>
    DT* data() const {
        return reinterpret_cast<DT*>(m_ptr.get());
    }

    template<typename DT>
    DT& at(const std::initializer_list<size_t>& index) const {
        size_t off = 0;
        auto it = index.begin();
        for (auto& stride : m_strides) {
            auto coordinate = (it != index.end()) ? (*it++) : 0;
            off += stride * coordinate;
        }
        return *reinterpret_cast<DT*>(reinterpret_cast<uint8_t*>(m_ptr.get()) + off);
    }

    template<typename DT>
    DT& operator()(const std::initializer_list<size_t>& index) const {
        return at<DT>(index);
    }

    void assert_dims(const std::initializer_list<size_t>& expect_dims) const;
    uint8_t** get_batched_ptrs() {
        uint8_t** ret_ptrs = batched_ptr_buff;
        auto batch_size = m_dims[0];
        if (batch_size > sizeof(batched_ptr_buff) / sizeof(batched_ptr_buff[0])) {
            batched_ptr_backup.resize(batch_size);
            ret_ptrs = &batched_ptr_backup[0];
        }
        for (size_t b = 0; b < batch_size; b++) {
            ret_ptrs[b] = &at<uint8_t>({b});
        }
        return ret_ptrs;
    }

    template<typename DT>
    std::string repr(int max_total_lines = 16, int lines_per_row = 1) const {
        std::stringstream ss;
        ss << typeid(DT).name() << " shape=[";
        const char* sep = "";
        size_t sz = 1;
        for (size_t i = 0; i < m_rank; i++) {
            ss << sep << m_dims[i];
            sz *= m_dims[i];
            sep = ",";
        }
        ss << "] strides=[";
        sep = "";
        for (size_t i = 0; i < m_rank; i++) {
            ss << sep << m_strides[i];
            sep = ",";
        }
        ss << "] {";
        if (m_rank > 1)
            ss << "\n";
        auto last_dim_size = m_dims[m_rank - 1];
        int row_id = 0;
        int cur_row_lines_left = lines_per_row;
        int cur_line_elecnt = 0;
        int cur_row_elecnt = 0;
        size_t i;
        auto* p = reinterpret_cast<DT*>(m_ptr.get());
        for (i = 0; i < sz && max_total_lines > 0; i++) {
            if ((i % last_dim_size) == 0) {
                ss << row_id << ":\t\t";
                row_id++;
                cur_row_lines_left = lines_per_row;
            }

            // display current element if we still have buget
            if (cur_row_lines_left > 0) {
                ss << p[i] << ",";
                cur_line_elecnt++;
                cur_row_elecnt++;
                if ((cur_line_elecnt % 16) == 15 || (cur_row_elecnt == last_dim_size)) {
                    max_total_lines--;
                    cur_row_lines_left--;
                    if (cur_row_lines_left == 0) {
                        if (cur_row_elecnt == last_dim_size)
                            ss << ",\n";
                        else
                            ss << "...\n";
                        cur_row_elecnt = 0;
                    } else {
                        ss << "\n\t\t";
                    }
                    cur_line_elecnt = 0;
                }
            }
        }
        if (i < sz) {
            ss << "... ... ... ... \n";
        }
        ss << "}";
        return ss.str();
    }

    template <typename U>
    friend std::ostream& operator<<(std::ostream& os, const plain_tensor& dt);    
};

template <typename U>
std::ostream& operator<<(std::ostream& os, const plain_tensor& dt) {
    os << dt.repr<U>();
    return os;
}

}  // namespace llmdnn
