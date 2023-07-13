// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <torch/extension.h>
#include <memory>
#include <tuple>
#include "alloca.h"
#include "llm_tensor.hpp"
#include "module.hpp"
#include "common/utility.hpp"
#include "utility_kernel_amx.hpp"
#include "llm_emb_gpt.hpp"
#include "test_common.hpp"

using namespace torch::indexing;

void regclass_emb_gpt(pybind11::module m) {
    m.def("emb_gpt", [] (
        const torch::Tensor& qkv,
        const torch::Tensor& k_past,
        const torch::Tensor& v_past,
        const torch::Tensor& cos,
        const torch::Tensor& sin,
        const torch::Tensor& position2d_ids) {
            // qkv: [batch, seq_len, (num_heads * 3 * head_size)]
            // k_past: [batch, head_num, past_seq_len, head_size]
            // q_dst: [batch, head_num, query_seq_len, head_size]
            // k_dst: [batch, head_num, query_seq_len+past_seq_len, head_size]
            // cos: [max_seq_len, rotary_dims]
            // position2d_ids: [batch, 2, query_seq_len]
            AT_ASSERT(qkv.dim() == 3 && k_past.dim() == 4 && v_past.dim() == 4);
            auto batch = qkv.size(0);
            auto query_seq_len = qkv.size(1);
            auto head_num = k_past.size(1);
            auto head_size = k_past.size(3);
            auto past_seq_len = k_past.size(2);
            auto kv_seq_len = query_seq_len + past_seq_len;

            torch::Tensor q_dst = qkv.new_empty({batch, head_num, query_seq_len, head_size});
            torch::Tensor k_dst = qkv.new_empty({batch, head_num, kv_seq_len, head_size});
            torch::Tensor v_dst = qkv.new_empty({batch, head_num, kv_seq_len, head_size});
            llmdnn::tensor q_, k_, v_, k_past_, v_past_, q_dst_, k_dst_, v_dst_, cos_, sin_, position2d_ids_;
            q_.resize({batch, query_seq_len, head_num, head_size * 3}, reinterpret_cast<ov::bfloat16*>(qkv.data_ptr()) + head_size * 0);
            q_.m_dims[3] = head_size;
            k_.resize({batch, query_seq_len, head_num, head_size * 3}, reinterpret_cast<ov::bfloat16*>(qkv.data_ptr()) + head_size * 1);
            k_.m_dims[3] = head_size;
            v_.resize({batch, query_seq_len, head_num, head_size * 3}, reinterpret_cast<ov::bfloat16*>(qkv.data_ptr()) + head_size * 2);
            v_.m_dims[3] = head_size;
            k_past_.resize({batch, head_num, past_seq_len, head_size}, reinterpret_cast<ov::bfloat16*>(k_past.data_ptr()));
            v_past_.resize({batch, head_num, past_seq_len, head_size}, reinterpret_cast<ov::bfloat16*>(v_past.data_ptr()));
            q_dst_.resize({batch, head_num, query_seq_len, head_size}, reinterpret_cast<ov::bfloat16*>(q_dst.data_ptr()));
            k_dst_.resize({batch, head_num, kv_seq_len, head_size}, reinterpret_cast<ov::bfloat16*>(k_dst.data_ptr()));
            v_dst_.resize({batch, head_num, kv_seq_len, head_size}, reinterpret_cast<ov::bfloat16*>(v_dst.data_ptr()));
            cos_.resize({cos.size(0), cos.size(1), cos.size(2), cos.size(3)}, cos.data_ptr<float>());
            sin_.resize({sin.size(0), sin.size(1), sin.size(2), sin.size(3)}, sin.data_ptr<float>());
            if (position2d_ids.numel())
                position2d_ids_.resize({batch, 2, query_seq_len}, position2d_ids.data_ptr<int32_t>());
            
            llmdnn::emb_gpt(q_, k_, v_, k_past_, v_past_, q_dst_, k_dst_, v_dst_, cos_, sin_, position2d_ids_);

            return std::make_tuple(q_dst, k_dst, v_dst);
            // auto options = torch::TensorOptions().dtype(torch::kBFloat16);
            // auto query = torch::from_blob(param.query, {batch, num_heads, query_seq_len, head_size}, options);
        },
        py::arg("qkv"),
        py::arg("k_past"),
        py::arg("v_past"),
        py::arg("cos"),
        py::arg("sin"),
        py::arg("position2d_ids"),
        R"(
            exec emb

            :param num_heads: heads number.
            :type num_heads: int
        )");
}