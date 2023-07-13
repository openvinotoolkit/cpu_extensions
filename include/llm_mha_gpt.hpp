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

class mha_gpt {
public:
    mha_gpt();
    ~mha_gpt();

    void exec(const tensor& q,              // q shape: [batch, num_heads, query_seq_len, head_size]
              const tensor& k,              // k shape: [batch, num_heads, key_seq_len, head_size]
              const tensor& v,              // v shape: [batch, num_heads, value_seq_len, head_size]
              const tensor& output,         // output, compact, shape: [batch, query_seq_len, num_heads * head_size]
              const tensor& attn_mask,      // attention mask[opt], shape:
                                            //      [batch, 1, 1, key_seq_len],
                                            //      [batch, 1, query_seq_len, key_seq_len]
              const tensor& alibi,          // alibi[opt] shape: [batch, num_heads, 1, key_seq_len]
              float normal_factor,
              bool use_causal_mask = false);// add causal mask

    struct impl {
        virtual ~impl() {}
        virtual void exec(const tensor& q,
                          const tensor& k,
                          const tensor& v,
                          const tensor& output,
                          const tensor& attn_mask,
                          const tensor& alibi,
                          float normal_factor,
                          bool use_causal_mask = false) = 0;
    };
protected:
    impl* _impl;
};

}
