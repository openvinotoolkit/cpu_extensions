// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>
#include <torch/extension.h>
#include <memory>
#include "alloca.h"
#include "common/bf16.hpp"
#include "module.hpp"
#include "common/utility.hpp"
#include "utility_kernel_amx.hpp"
#include "llm_mha_gpt.hpp"
#include "test_common.hpp"

void regclass_mha_gpt(pybind11::module m) {
    py::class_<llmdnn::mha_gpt, std::shared_ptr<llmdnn::mha_gpt>> cls(m, "mha_gpt");
    cls.def(py::init<>());
    cls.def("create", [] (llmdnn::mha_gpt& self,
        const size_t num_heads,
        const size_t head_size,
        const float normal_factor,
        const std::string qkv_precision_name,
        const std::string dst_precision_name,
        const size_t max_seq_len,
        bool is_bloom) {
            llmdnn::mha_gpt::create_param param = {0};
            param.num_heads = num_heads;
            param.head_size = head_size;
            param.normal_factor = normal_factor;
            param.qkv_precision = llmdnn::get_dt_from_str(qkv_precision_name);
            param.dst_precision = llmdnn::get_dt_from_str(dst_precision_name);
            param.max_seq_len = max_seq_len;
            param.is_bloom = is_bloom;
            if (param.qkv_precision == llmdnn::dnnl_data_type_undef)
                throw pybind11::type_error("Incorrect qkv type " + qkv_precision_name);
            if (param.dst_precision == llmdnn::dnnl_data_type_undef)
                throw pybind11::type_error("Incorrect dst type " + dst_precision_name);
            if (!self.create(param))
                throw pybind11::type_error("Incorrect param");
        },
        py::arg("num_heads"),
        py::arg("head_size"),
        py::arg("normal_factor"),
        py::arg("qkv_precision_name"),
        py::arg("dst_precision_name"),
        py::arg("max_seq_len"),
        py::arg("is_bloom") = false,
        R"(
            Create mha

            :param num_heads: heads number.
            :type num_heads: int
        )");
    cls.def("exec", [] (llmdnn::mha_gpt& self, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& alibi, const torch::Tensor& attn_mask, int64_t head_size, int64_t key_seq_len) {
            // q: [batch, num_heads, query_seq_len, head_size_aligned]
            // k: [batch, num_heads, head_size_aligned, max_seq_len] valid in max_seq_len: key_seq_len
            // v: [batch, num_heads, max_seq_len, head_size_aligned] valid in max_seq_len: key_seq_len
            // attn_mask: [batch, 1, 1/query_seq_len, key_seq_len]
            // out: [batch, query_seq_len, num_heads * head_size]
            AT_ASSERT(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && attn_mask.dim() == 4);
            auto batch = q.size(0);
            auto num_heads = q.size(1);
            auto query_seq_len = q.size(2);
            auto head_size_aligned = q.size(3);
            auto max_seq_len = v.size(2);
            auto attn_len = attn_mask.size(3);
            AT_ASSERT(max_seq_len == v.size(2) &&
                    batch == k.size(0) && batch == v.size(0) && batch == attn_mask.size(0) &&
                    num_heads == k.size(1) && num_heads == v.size(1) &&
                    head_size_aligned == v.size(3));

            llmdnn::mha_gpt::exec_param param = {0};
            param.batch = batch;
            param.query_seq_len = query_seq_len;
            param.key_seq_len = key_seq_len == 0 ? max_seq_len : key_seq_len;
            head_size = head_size == 0 ? head_size_aligned : head_size;
            auto out = q.new_empty({batch, query_seq_len, num_heads * head_size});
            AT_ASSERT((int64_t)param.key_seq_len == attn_len);
            param.q.resize({q.size(0), q.size(1), q.size(2), q.size(3)}, reinterpret_cast<ov::bfloat16*>(q.data_ptr()));
            param.attn_output.resize({batch, query_seq_len, num_heads * head_size}, reinterpret_cast<ov::bfloat16*>(out.data_ptr()));
            param.is_causal_in_attention = attn_mask.size(2) != 1;
            param.attention_mask.resize({attn_mask.size(0), attn_mask.size(1), attn_mask.size(2), attn_mask.size(3)}, attn_mask.data_ptr<float>());
            param.k.resize({k.size(0), k.size(1), k.size(2), k.size(3)}, reinterpret_cast<ov::bfloat16*>(k.data_ptr()));
            if (alibi.dim() == 3) {
                std::swap(param.k.m_dims[2], param.k.m_dims[3]);
                std::swap(param.k.m_strides[2], param.k.m_strides[3]);
                param.alibi.resize({alibi.size(0), alibi.size(1), alibi.size(2)}, alibi.data_ptr<float>());
            }
            param.v.resize({v.size(0), v.size(1), v.size(2), v.size(3)}, reinterpret_cast<ov::bfloat16*>(v.data_ptr()));

            self.exec(param);
            return out;
        },
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("alibi"),
        py::arg("attn_mask"),
        py::arg("head_size") = 0,
        py::arg("key_seq_len") = 0,
        R"(
            exec mha

            :param num_heads: heads number.
            :type num_heads: int
        )");
    cls.def("exec_quant", [] (llmdnn::mha_gpt& self, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& attn_mask,
        float q_dequant, float k_dequant, float v_dequant, float qk_quant, const std::vector<float>& qkv_quant, int64_t head_size, int64_t key_seq_len) {
            // q: [batch, num_heads, query_seq_len, head_size_aligned]
            // k: [batch, num_heads, max_seq_len, head_size_aligned] valid in max_seq_len: key_seq_len
            // v: [batch, num_heads, max_seq_len, head_size_aligned] valid in max_seq_len: key_seq_len
            // attn_mask: [batch, 1, 1/query_seq_len, key_seq_len]
            // out: [batch, query_seq_len, num_heads * head_size]
            AT_ASSERT(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && attn_mask.dim() == 4);
            auto batch = q.size(0);
            auto num_heads = q.size(1);
            auto query_seq_len = q.size(2);
            auto head_size_aligned = q.size(3);
            auto max_seq_len = k.size(2);
            auto attn_len = attn_mask.size(3);
            AT_ASSERT(max_seq_len == v.size(2) &&
                    batch == k.size(0) && batch == v.size(0) && batch == attn_mask.size(0) &&
                    num_heads == k.size(1) && num_heads == v.size(1) &&
                    head_size_aligned == k.size(3) && head_size_aligned == v.size(3));

            llmdnn::mha_gpt::exec_param param = {0};
            param.batch = batch;
            param.query_seq_len = query_seq_len;
            param.key_seq_len = key_seq_len == 0 ? max_seq_len : key_seq_len;
            head_size = head_size == 0 ? head_size_aligned : head_size;
            auto out = q.new_empty({batch, query_seq_len, num_heads * head_size}, torch::TensorOptions(torch::kInt8));
            AT_ASSERT((int64_t)param.key_seq_len == attn_len);

            param.q.resize({q.size(0), q.size(1), q.size(2), q.size(3)}, reinterpret_cast<uint8_t*>(q.data_ptr()));
            param.attn_output.resize({batch, query_seq_len, num_heads * head_size}, reinterpret_cast<uint8_t*>(out.data_ptr()));
            param.is_causal_in_attention = attn_mask.size(2) != 1;
            param.attention_mask.resize({attn_mask.size(0), attn_mask.size(1), attn_mask.size(2), attn_mask.size(3)}, attn_mask.data_ptr<float>());
            param.k.resize({k.size(0), k.size(1), k.size(2), k.size(3)}, reinterpret_cast<uint8_t*>(k.data_ptr()));
            param.v.resize({v.size(0), v.size(1), v.size(2), v.size(3)}, reinterpret_cast<uint8_t*>(v.data_ptr()));

            param.q_dequant = q_dequant;
            param.k_dequant = k_dequant;
            param.v_dequant = v_dequant;
            param.qk_quant = qk_quant;
            param.qkv_quant = qkv_quant;

            self.exec(param);
            return out;
        },
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("attn_mask"),
        py::arg("q_dequant"),
        py::arg("k_dequant"),
        py::arg("v_dequant"),
        py::arg("qk_quant"),
        py::arg("qkv_quant"),
        py::arg("head_size") = 0,
        py::arg("key_seq_len") = 0,
        R"(
            exec mha quant

            :param num_heads: heads number.
            :type num_heads: int
        )");
}