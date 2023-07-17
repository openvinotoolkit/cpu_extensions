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
#include <memory.h>

#include "llm_types.hpp"

// forward declaration
namespace ov {
class bfloat16;
};

namespace llmdnn {

template <typename T>
struct precision_of {
    static constexpr data_type_t value = dnnl_data_type_undef;
};

template <>
struct precision_of<float> {
    static constexpr data_type_t value = dnnl_f32;
};

template <>
struct precision_of<int32_t> {
    static constexpr data_type_t value = dnnl_s32;
};

template <>
struct precision_of<ov::bfloat16> {
    static constexpr data_type_t value = dnnl_bf16;
};

template <>
struct precision_of<uint8_t> {
    static constexpr data_type_t value = dnnl_u8;
};

template <>
struct precision_of<int8_t> {
    static constexpr data_type_t value = dnnl_s8;
};


#define TENSOR_RANK_MAX 8
struct tensor {
    size_t m_strides[TENSOR_RANK_MAX];
    size_t m_dims[TENSOR_RANK_MAX];
    size_t m_rank = 0;

    void* m_ptr = nullptr;
    size_t m_capacity = 0;          // 0 means not own m_ptr
    size_t m_element_size = 0;
    data_type_t m_dtype = dnnl_data_type_undef;

    tensor();
    ~tensor();
    tensor(const tensor&) = delete;
    tensor& operator = (const tensor&) = delete;
    tensor(tensor&& t) {
        memcpy(reinterpret_cast<void*>(this), &t, sizeof(*this));
        t.m_capacity = 0;
        t.m_ptr = nullptr;
    }
    tensor& operator = (tensor&& t) {
        if (m_capacity && m_ptr)
            free(m_ptr);
        memcpy(reinterpret_cast<void*>(this), &t, sizeof(*this));
        t.m_capacity = 0;
        t.m_ptr = nullptr;
        return *this;
    }
    operator bool() const {
        return m_ptr != nullptr;
    }

    size_t size(int i) const {
        assert(static_cast<size_t>(i) < m_rank);
        return m_dims[i];
    }
    size_t stride(int i) const {
        assert(static_cast<size_t>(i) < m_rank);
        return m_strides[i];
    }

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

    tensor index(const std::initializer_list<tensor_index>& indices) const;

    // slice: return a sub-view (w/o ownership/refcount to original data)
    tensor slice(int axis, int start, int end) const;

    bool is_dense() const;

    tensor reshape(const std::initializer_list<size_t>& target_shape) const;

    tensor permute(const std::initializer_list<size_t>& order) const;

    template<typename DT>
    void resize(const size_t* new_dims, size_t dim_num, DT* data = nullptr) {
        resize(new_dims, dim_num, data, sizeof(DT), precision_of<DT>::value);
    }

    template<typename DT>
    void resize(const std::vector<size_t>& new_dims, DT* data = nullptr) {
        resize(new_dims.data(), new_dims.size(), data);
    }

    void resize(const size_t* new_dims, size_t dim_num, void* data, size_t element_size, data_type_t dtype);
    void resize(const std::vector<size_t>& new_dims, void* data, size_t element_size, data_type_t dtype) {
        resize(new_dims.data(), new_dims.size(), data, element_size, dtype);
    }

    template<typename DT>
    DT* data() const {
        return reinterpret_cast<DT*>(m_ptr);
    }

    template<typename DT>
    DT& at(const std::initializer_list<size_t>& index) const {
        size_t off = 0;
        auto it = index.begin();
        for (auto& stride : m_strides) {
            auto coordinate = (it != index.end()) ? (*it++) : 0;
            off += stride * coordinate;
        }
        return *reinterpret_cast<DT*>(reinterpret_cast<uint8_t*>(m_ptr) + off);
    }

    template<typename DT>
    DT& operator()(const std::initializer_list<size_t>& index) const {
        return at<DT>(index);
    }

    void assert_dims(const std::initializer_list<size_t>& expect_dims) const;
};

}  // namespace llmdnn
