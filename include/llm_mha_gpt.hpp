// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "llm_types.hpp"
#include "llm_plain_tensor.hpp"

namespace llmdnn {

class mha_gpt {
public:
    struct create_param {
        size_t num_heads;
        size_t head_size;
        size_t max_seq_len;             // max seq length for computing the size of matmul tmp result
        float normal_factor;
        // supported (qkv, dst): (bf16, bf16), (s8, s8)
        data_type_t qkv_precision;
        data_type_t dst_precision;
        bool is_bloom;                  // for bloom mha
    };
    struct exec_param {
        size_t batch;
        size_t query_seq_len;
        size_t key_seq_len;
        bool is_causal_in_attention;        // causal mask is fused in attention mask: chatglm uses it.
        plain_tensor q;                     // q buffer, shape: [batch, num_heads, query_seq_len, head_size]
        plain_tensor k;                     // k buffer, shape: [batch, num_heads, key_seq_len, head_size]
        plain_tensor v;                     // v buffer, shape: [batch, num_heads, value_seq_len, head_size]
        plain_tensor attention_mask;        // attention mask, shape:
                                            //      [batch, 1, 1, key_seq_len], when is_causal_in_attention is false
                                            //      [batch, 1, query_seq_len, key_seq_len], when is_causal_in_attention is true
        plain_tensor attn_output;           // output, compact, shape: [batch, query_seq_len, num_heads * head_size]
        plain_tensor alibi;                 // only is_bloom is true will use, shape: [batch, num_heads, 1, key_seq_len]
        // expected quant schema:
        //   q,k,v use per tensor quant, attn_output may use per tensor/channel quant
        float q_dequant;
        float k_dequant;
        float v_dequant;
        float qk_quant;
        std::vector<float> qkv_quant;       // size==1 per tensor, size==head_size per channel
    };

    mha_gpt();
    ~mha_gpt();
    bool create(const create_param& param);
    void exec(const exec_param& param);

    struct impl {
        virtual ~impl() {}
        virtual bool create(const create_param& param) = 0;
        virtual void exec(const exec_param& param) = 0;
    };
protected:
    impl* _impl;
};

}
