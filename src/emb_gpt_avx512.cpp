// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <string>
#include <vector>
#include <memory.h>

#include "common/bf16.hpp"
#include "common/simple_parallel.hpp"
#include "common/utility.hpp"
#include "utility_kernel_avx512.hpp"
#include "transpose_kernel_avx512.hpp"
#include "llm_emb_gpt.hpp"
#include "emb_gpt_avx512.hpp"
#include "rotary_kernel_avx512.hpp"

using namespace utility;

namespace llmdnn {

static void memcpy_past_kv(const tensor& k_past, const tensor& v_past, const tensor& k_dst, const tensor& v_dst) {
    auto batch = k_past.m_dims[0];
    auto head_num = k_past.m_dims[1];
    auto past_seq_len = k_past.m_dims[2];
    auto size = k_past.m_dims[3];
    parallel_for3d(batch, head_num, past_seq_len, [&](size_t b, size_t h, size_t s) {
        memcpy(&k_dst.at<uint8_t>({b, h, s}), &k_past.at<uint8_t>({b, h, s}), k_past.m_strides[2]);
        memcpy(&v_dst.at<uint8_t>({b, h, s}), &v_past.at<uint8_t>({b, h, s}), v_past.m_strides[2]);
    });
}

// q_src shape: [batch, q_seq_len, head_hum, head_size]
// q_dst shape: [batch, head_hum, q_seq_len, head_size]
// kv_src shape: [batch, q_seq_len, head_hum, head_size]
// kv_past shape: [batch, head_hum, past_seq_len, head_size]
// kv_dst shape: [batch, head_hum, q_seq_len+past_seq_len, head_size]
// position2d_ids: [batch, 2, q_seq_len]
// cos/sin: [max_seq_len, rotary_dims]
static void rotary_emb_position2d(const tensor& q_src,
                                  const tensor& k_src,
                                  const tensor& v_src,
                                  const tensor& k_past,
                                  const tensor& v_past,
                                  const tensor& q_dst,
                                  const tensor& k_dst,
                                  const tensor& v_dst,
                                  const tensor& cos,
                                  const tensor& sin,
                                  const tensor& position2d_ids) {
    auto batch = k_past.m_dims[0];
    auto head_num = k_past.m_dims[1];
    auto past_seq_len = k_past.m_dims[2];
    auto head_size = k_past.m_dims[3];
    auto query_seq_len = q_src.m_dims[1];
    auto rotary_ndim = cos.m_dims[3];

    parallel_for3d(batch, head_num, query_seq_len, [&](size_t b, size_t h, size_t s) {
        auto kv_dst_s = s + past_seq_len;
        // q, k rotary encoding
        if (position2d_ids) {
            auto pos = position2d_ids.at<uint32_t>({b, 0, s});
            rotary_avx512(rotary_ndim, &cos.at<float>({0, 0, pos}), &sin.at<float>({0, 0, pos}),
                &q_src.at<ov::bfloat16>({b, s, h}),
                &k_src.at<ov::bfloat16>({b, s, h}),
                &q_dst.at<ov::bfloat16>({b, h, s}),
                &k_dst.at<ov::bfloat16>({b, h, kv_dst_s}));
            pos = position2d_ids.at<uint32_t>({b, 1, s});
            rotary_avx512(rotary_ndim, &cos.at<float>({0, 0, pos}), &sin.at<float>({0, 0, pos}),
                &q_src.at<ov::bfloat16>({b, s, h, rotary_ndim}),
                &k_src.at<ov::bfloat16>({b, s, h, rotary_ndim}),
                &q_dst.at<ov::bfloat16>({b, h, s, rotary_ndim}),
                &k_dst.at<ov::bfloat16>({b, h, kv_dst_s, rotary_ndim}));
        } else {
            rotary_avx512(rotary_ndim, &cos.at<float>({0, 0, s + past_seq_len}), &sin.at<float>({0, 0, s + past_seq_len}),
                &q_src.at<ov::bfloat16>({b, s, h}),
                &k_src.at<ov::bfloat16>({b, s, h}),
                &q_dst.at<ov::bfloat16>({b, h, s}),
                &k_dst.at<ov::bfloat16>({b, h, kv_dst_s}));
            memcpy(&q_dst.at<ov::bfloat16>({b, h, s, rotary_ndim}), &q_src.at<ov::bfloat16>({b, s, h, rotary_ndim}), (head_size - rotary_ndim) * sizeof(ov::bfloat16));
            memcpy(&k_dst.at<ov::bfloat16>({b, h, kv_dst_s, rotary_ndim}), &k_src.at<ov::bfloat16>({b, s, h, rotary_ndim}), (head_size - rotary_ndim) * sizeof(ov::bfloat16));
        }

        // v concat
        memcpy(&v_dst.at<ov::bfloat16>({b, h, kv_dst_s}), &v_src.at<ov::bfloat16>({b, s, h}), head_size * sizeof(ov::bfloat16));
    });
}

void emb_gpt_avx512(const tensor& q_src,
                    const tensor& k_src,
                    const tensor& v_src,
                    const tensor& k_past,
                    const tensor& v_past,
                    const tensor& q_dst,
                    const tensor& k_dst,
                    const tensor& v_dst,
                    const tensor& cos,
                    const tensor& sin,
                    const tensor& position2d_ids) {
    if (q_src.m_rank != 4 || k_src.m_rank != 4 || v_src.m_rank != 4 || k_past.m_rank != 4 || v_past.m_rank != 4 || q_dst.m_rank != 4||
        k_dst.m_rank != 4 || v_dst.m_rank != 4 || cos.m_rank != 4 || sin.m_rank != 4) {
        std::cout << "emb_gpt_avx512: rank is not correct: should be 4\n";
        return;
    }
    if (position2d_ids) {
        if (position2d_ids.m_rank != 3) {
            std::cout << "emb_gpt_avx512: position2d_ids rank should be 3\n";
            return;
        }
        if (position2d_ids.m_dims[0] != q_src.m_dims[0] || position2d_ids.m_dims[1] != 2 || position2d_ids.m_dims[2] != q_src.m_dims[1]) {
            std::cout << "emb_gpt_avx512: position2d_ids dims should be [batch, 2, seq_len]\n";
            return;
        }
    }

    // [batch, seq_len, (num_heads * 3 * head_size)]
    //   --> [batch, seq_len, num_heads, 3 * head_size]
    auto past_seq_len = k_past.m_dims[2];

    // past kv src != dst, copy src to dst first
    if (k_past.m_ptr != k_dst.m_ptr && past_seq_len)
        memcpy_past_kv(k_past, v_past, k_dst, v_dst);

    // transpose + rotary embbeding:
    // transpose: [batch, seq_len, head_hum, 3 * head_size] -->
    //          3 [batch, head_hum, seq_len, head_size]
    // rotary embbeding: part of key will write to past_key, part of query will write to tempory buffer
    if (q_src.m_dtype == dnnl_s8) {
        assert(false);
    } else {
        // query pass part(temp buffer): query = torch.cat((query, query_pass), dim=-1)
        // key pass part(past_key): key = torch.cat((key, key_pass), dim=-1)
        // value(pastKeys): value = torch.cat((past_value, value), dim=-2)
        rotary_emb_position2d(q_src, k_src, v_src, k_past, v_past, q_dst, k_dst, v_dst, cos, sin, position2d_ids);
    }
}

}