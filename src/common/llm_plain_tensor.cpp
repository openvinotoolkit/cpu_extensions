// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <string>
#include <sys/types.h>
#include <vector>
#include <iostream>
#include <sstream>

#include "bf16.hpp"
#include "llm_plain_tensor.hpp"

namespace llmdnn {

plain_tensor::plain_tensor() {
}

plain_tensor::~plain_tensor() {
    if (m_capacity && m_ptr)
        free(m_ptr);
}

plain_tensor plain_tensor::index(const std::initializer_list<tensor_index>& indices) const {
    plain_tensor sub_tensor;
    assert(indices.size() <= m_rank);
    int i_src = 0;
    int i_dst = 0;
    sub_tensor.m_capacity = 0;
    size_t off = 0;
    for (auto idx : indices) {
        auto src_dim = m_dims[i_src];
        auto src_stride = m_strides[i_src];
        idx.regularize(src_dim);
        off += idx.start * src_stride;
        if (idx.slice_with_squeeze()) {
            // no output dimension
            i_src++;
            continue;
        }
        sub_tensor.m_dims[i_dst] = idx.count;
        sub_tensor.m_strides[i_dst] = src_stride;
        i_dst++;
        i_src++;
    }
    sub_tensor.m_rank = i_dst;  // index may imply squeeze
    sub_tensor.m_ptr = reinterpret_cast<uint8_t*>(m_ptr) + off;
    return sub_tensor;
}

// slice: return a sub-view (w/o ownership/refcount to original data)
plain_tensor plain_tensor::slice(int axis, int start, int end) const {
    plain_tensor sub_tensor;
    assert(axis < m_rank);

    sub_tensor.m_capacity = 0;
    sub_tensor.m_rank = m_rank;  // slice dosen't change rank & strides
    for (size_t i = 0; i < m_rank; i++) {
        sub_tensor.m_strides[i] = m_strides[i];
        sub_tensor.m_dims[i] = m_dims[i];
    }
    sub_tensor.m_dims[axis] = end - start;

    auto off = start * m_strides[axis];
    auto* data = reinterpret_cast<uint8_t*>(m_ptr) + off;
    sub_tensor.m_ptr = reinterpret_cast<void*>(data);

    return sub_tensor;
}

bool plain_tensor::is_dense() const {
    // check if it's dense tensor
    size_t stride = m_element_size;
    for (int i = m_rank - 1; i >= 0; i--) {
        if (m_strides[i] != stride)
            return false;
        stride *= m_dims[i];
    }
    return true;
}

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
plain_tensor plain_tensor::reshape(const std::initializer_list<size_t>& target_shape) const {
    // only valid for dense memory
    plain_tensor new_tensor_view;
    assert(is_dense());
    //assert(shape_size(target_shape) == shape_size(m_dims));
    new_tensor_view.resize(std::vector<size_t>(target_shape), m_ptr, m_element_size);
    return new_tensor_view;
}

plain_tensor plain_tensor::permute(const std::initializer_list<size_t>& order) const {
    plain_tensor new_tensor_view;
    assert(order.size() == m_rank);
    new_tensor_view.m_capacity = 0;
    new_tensor_view.m_ptr = m_ptr;
    new_tensor_view.m_rank = m_rank;
    auto it_order = order.begin();
    // also should check order has no repeat element
    for (size_t i = 0; i < m_rank; i++) {
        auto j = *it_order++;
        assert(j >= 0 && j < m_rank);
        new_tensor_view.m_dims[i] = m_dims[j];
        new_tensor_view.m_strides[i] = m_strides[j];
    }
    return new_tensor_view;
}

void plain_tensor::resize(const std::vector<size_t>& new_dims, void* data, size_t element_size) {
    // initialize strides for compact/dense tensor
    m_element_size = element_size;
    m_rank = new_dims.size();
    assert(m_rank <= PLAINTENSOR_RANK_MAX);
    size_t stride = element_size;
    for (int i = m_rank - 1; i >= 0; i--) {
        m_dims[i] = new_dims[i];
        m_strides[i] = stride;
        stride *= new_dims[i];
    }

    if (!data) {
        auto capacity_new = m_strides[0] * m_dims[0];
        if (capacity_new > m_capacity) {
            m_ptr = aligned_alloc(64, capacity_new);
            m_capacity = capacity_new;
        }
    } else {
        // m_capacity is zero to indicate that we don't own the memory
        m_capacity = 0;
        m_ptr = reinterpret_cast<void*>(data);
    }
}

void plain_tensor::assert_dims(const std::initializer_list<size_t>& expect_dims) const {
    if (m_rank != expect_dims.size()) {
        asm("int3");
        std::cout << "dims not same\n";
    }
    if (!std::equal(expect_dims.begin(), expect_dims.end(), m_dims)) {
        std::stringstream ss;
        ss << " m_dims=[";
        for (size_t i = 0; i < m_rank; i++)
            ss << m_dims[i] << ",";
        ss << "] expect_dims=[";
        for (auto& i : expect_dims)
            ss << i << ",";
        ss << "]";
        asm("int3");
        std::cout << ss.str();
    }
}

}  // namespace llmdnn
