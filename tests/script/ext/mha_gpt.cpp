// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>
#include <torch/extension.h>
#include <memory>
#include "alloca.h"
#include "common/bf16.hpp"
#include "llm_tensor.hpp"
#include "module.hpp"
#include "common/utility.hpp"
#include "utility_kernel_amx.hpp"
#include "llm_mha_gpt.hpp"
#include "test_common.hpp"

void regclass_mha_gpt(pybind11::module m) {
    py::class_<llmdnn::mha_gpt, std::shared_ptr<llmdnn::mha_gpt>> cls(m, "mha_gpt");
    cls.def(py::init<>());
    cls.def("exec", [] (llmdnn::mha_gpt& self, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& alibi,
                        const torch::Tensor& attn_mask, const torch::Tensor& causal_mask, bool select_nfltmax_at_0, float normal_factor, bool use_causal) {
            // q: [batch, num_heads, query_seq_len, head_size]
            // k: [batch, num_heads, key_seq_len, head_size]
            // v: [batch, num_heads, key_seq_len, head_size]
            // attn_mask: [batch, 1, 1/query_seq_len, key_seq_len]
            // out: [batch, query_seq_len, num_heads * head_size]
            // alibi: [batch, num_heads, 1, key_seq_len]
            // causal_mask: [batch, 1, query_seq_len, key_seq_len]
            AT_ASSERT(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && attn_mask.dim() == 4);
            auto batch = q.size(0);
            auto num_heads = q.size(1);
            auto query_seq_len = q.size(2);
            auto head_size = q.size(3);
            AT_ASSERT(batch == k.size(0) && batch == v.size(0) && batch == attn_mask.size(0) &&
                    num_heads == k.size(1) && num_heads == v.size(1) &&
                    head_size == v.size(3));

            llmdnn::tensor q_, k_, v_, out_, attn_mask_, alibi_, causal_mask_;
            q_.resize({q.size(0), q.size(1), q.size(2), q.size(3)}, reinterpret_cast<ov::bfloat16*>(q.data_ptr()));
            k_.resize({k.size(0), k.size(1), k.size(2), k.size(3)}, reinterpret_cast<ov::bfloat16*>(k.data_ptr()));
            if (k.size(2) != v.size(2)) {
                // bloom k shape: [batch, num_heads, head_size, key_seq_len]
                std::swap(k_.m_dims[2], k_.m_dims[3]);
                std::swap(k_.m_strides[2], k_.m_strides[3]);
            }
            v_.resize({v.size(0), v.size(1), v.size(2), v.size(3)}, reinterpret_cast<ov::bfloat16*>(v.data_ptr()));
            auto out = q.new_empty({batch, query_seq_len, num_heads * head_size});
            out_.resize({batch, query_seq_len, num_heads * head_size}, reinterpret_cast<ov::bfloat16*>(out.data_ptr()));
            if (attn_mask.numel())
                attn_mask_.resize({attn_mask.size(0), attn_mask.size(1), attn_mask.size(2), attn_mask.size(3)}, attn_mask.data_ptr<float>());
            if (alibi.numel())
                alibi_.resize({alibi.size(0), alibi.size(1), alibi.size(2), alibi.size(3)}, alibi.data_ptr<float>());
            if (causal_mask.numel())
                causal_mask_.resize({causal_mask.size(0), causal_mask.size(1), causal_mask.size(2), causal_mask.size(3)}, causal_mask.data_ptr<uint8_t>());
            self.exec(q_, k_, v_, out_, attn_mask_, alibi_, causal_mask_, select_nfltmax_at_0, normal_factor, use_causal);

            return out;
        },
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("alibi"),
        py::arg("attn_mask"),
        py::arg("causal_mask"),
        py::arg("select_nfltmax_at_0"),
        py::arg("normal_factor"),
        py::arg("use_causal"),
        R"(
            exec mha

            :param num_heads: heads number.
            :type num_heads: int
        )");
}