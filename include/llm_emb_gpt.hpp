// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "llm_types.hpp"
#include "llm_tensor.hpp"

namespace llmdnn {

status_t emb_gpt(const tensor& q_src,              // q shape: [batch, query_seq_len, head_num, head_size]
                 const tensor& k_src,              // k shape: [batch, query_seq_len, head_num, head_size]
                 const tensor& v_src,              // v shape: [batch, query_seq_len, head_num, head_size]
                 const tensor& k_past,             // k_past shape: [batch, num_heads, past_seq_len, head_size]
                 const tensor& v_past,             // v_past shape: [batch, num_heads, past_seq_len, head_size]
                 const tensor& q_dst,              // q_dst, shape: [batch, num_heads, query_seq_len, head_size]
                 const tensor& k_dst,              // k_past shape: [batch, num_heads, query_seq_len+past_seq_len, head_size]
                                                   // if k_past!=k_past_dst, will copy k_past to k_past_dst
                 const tensor& v_dst,              // v_past shape: [batch, num_heads, query_seq_len+past_seq_len, head_size]
                 const tensor& cos,                // cos lookup table, shape: [1, 1, max_seq_len, rotary_dims]
                 const tensor& sin,                // sin lookup table, shape: [1, 1, max_seq_len, rotary_dims]
                 const tensor& position2d_ids);    // shape: [batch, 2, query_seq_len]

}  // namespace llmdnn
