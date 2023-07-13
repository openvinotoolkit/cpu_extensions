// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <string>
#include <vector>

#include "common/simple_parallel.hpp"
#include "common/tensor2d.hpp"
#include "common/utility.hpp"
#include "llm_types.hpp"
#include "utility_kernel_avx512.hpp"
#include "mm_kernel_common_amx.hpp"
#include "softmax_kernel_avx512.hpp"
#include "transpose_kernel_avx512.hpp"
#include "llm_mha_gpt.hpp"
#include "mha_gpt_amx.hpp"

using namespace ov::cpu;

namespace llmdnn {

struct mha_gpt_impl_amx : public mha_gpt::impl {
    void create(data_type_t in_type, size_t seq_len, size_t head_size, bool is_bloom);
    void exec(const tensor& q, const tensor& k, const tensor& v, const tensor& output, const tensor& attn_mask, const tensor& alibi, float normal_factor, bool use_causal_mask) override;

    void mha_bf16(const tensor& q, const tensor& k, const tensor& v, const tensor& output, const tensor& attn_mask, const tensor& alibi, float normal_factor, bool use_causal_mask);

    size_t _head_size_aligned = 0;
    size_t _buffer_mat0_out_size = 0;
    size_t _buffer_mat1_out_size = 0;
    size_t _num_threads = 0;

    std::shared_ptr<uint8_t> _buffer_mat0_out;
    std::shared_ptr<uint8_t> _buffer_mat1_out;

    std::vector<std::shared_ptr<amx_kernel::MatmulVector<ov::bfloat16, ov::bfloat16>>> gemAvB_BF16xBF16;
    std::vector<std::shared_ptr<amx_kernel::Matmul<ov::bfloat16, ov::bfloat16>>> qKtrGemm_BF16xBF16;
    std::vector<std::shared_ptr<amx_kernel::Matmul<ov::bfloat16, ov::bfloat16>>> qKVGemm_BF16xBF16;

    std::vector<std::shared_ptr<amx_kernel::Matmul<int8_t, int8_t>>> qKtrGemm_i8xi8;
    std::vector<std::shared_ptr<amx_kernel::Matmul<uint8_t, int8_t>>> qKVGemm_u8xi8;
    std::vector<std::shared_ptr<amx_kernel::MatmulVector<int8_t, int8_t>>> gemAvB_i8xi8;
};

void mha_gpt_impl_amx::create(data_type_t in_type, size_t seq_len, size_t head_size, bool is_bloom) {
    // q: [batch, head_num, query_seq_len, head_size]
    // k: [batch, head_num, maxSeqLen(valid: key_seq_len), head_size]
    // v: [batch, head_num, maxSeqLen(valid: value_seq_len), head_size]
    // attention_mask: [batch, 1, 1, maxSeqLen(valid: key_seq_len)]
    // matmul1: [batch, head_num, query_seq_len, head_size]
    // attn_output: [batch, query_seq_len, head_num * head_size]
    if (_num_threads == 0) {
        _num_threads = getTotalThreads();
        if (in_type == dnnl_s8) {
            _head_size_aligned = rndup(head_size, 64);
            qKtrGemm_i8xi8.resize(_num_threads);
            for (size_t i = 0; i < _num_threads; i++) {
                qKtrGemm_i8xi8[i] = std::make_shared<amx_kernel::Matmul<int8_t, int8_t>>(false, !is_bloom);
            }
            qKVGemm_u8xi8.resize(_num_threads);
            for (size_t i = 0; i < _num_threads; i++) {
                qKVGemm_u8xi8[i] = std::make_shared<amx_kernel::Matmul<uint8_t, int8_t>>(false, false);
            }
            gemAvB_i8xi8.resize(_num_threads);
            for (size_t i = 0; i < _num_threads; i++) {
                gemAvB_i8xi8[i] = std::make_shared<amx_kernel::MatmulVector<int8_t, int8_t>>();
            }
        } else {
            _head_size_aligned = rndup(head_size, 32);
            gemAvB_BF16xBF16.resize(_num_threads);
            for (size_t i = 0; i < _num_threads; i++) {
                gemAvB_BF16xBF16[i] = std::make_shared<amx_kernel::MatmulVector<ov::bfloat16, ov::bfloat16>>();
            }
            qKtrGemm_BF16xBF16.resize(_num_threads);
            for (size_t i = 0; i < _num_threads; i++) {
                qKtrGemm_BF16xBF16[i] = std::make_shared<amx_kernel::Matmul<ov::bfloat16, ov::bfloat16>>(false, !is_bloom);
            }
            qKVGemm_BF16xBF16.resize(_num_threads);
            for (size_t i = 0; i < _num_threads; i++) {
                qKVGemm_BF16xBF16[i] = std::make_shared<amx_kernel::Matmul<ov::bfloat16, ov::bfloat16>>(false, false);
            }
        }
    }

    auto buffer_mat0_out_size = seq_len * rndup(seq_len * sizeof(float), 64);
    if (buffer_mat0_out_size > _buffer_mat0_out_size) {
        _buffer_mat0_out_size = seq_len * rndup(seq_len * sizeof(float), 64) * 3 / 2;
        _buffer_mat1_out_size = seq_len * _head_size_aligned * sizeof(float) * 3 / 2;

        _buffer_mat0_out = std::shared_ptr<uint8_t>(
                                reinterpret_cast<uint8_t*>(aligned_alloc(64, _num_threads * _buffer_mat0_out_size)),
                                [](void * p) { ::free(p); });
        memset(_buffer_mat0_out.get(), 0, _num_threads * _buffer_mat0_out_size);
        _buffer_mat1_out = std::shared_ptr<uint8_t>(
                                reinterpret_cast<uint8_t*>(aligned_alloc(64, _num_threads * _buffer_mat1_out_size)),
                                [](void * p) { ::free(p); });
        memset(_buffer_mat1_out.get(), 0, _num_threads * _buffer_mat1_out_size);
    }
}

void mha_gpt_impl_amx::mha_bf16(const tensor& q, const tensor& k, const tensor& v, const tensor& output, const tensor& attn_mask,
    const tensor& alibi, float normal_factor, bool use_causal_mask) {
    auto batch = q.m_dims[0];
    auto head_num = q.m_dims[1];
    auto query_seq_len = q.m_dims[2];
    auto head_size = q.m_dims[3];
    auto key_seq_len = k.m_dims[2];
    bool is_bloom = k.m_strides[3] > k.m_strides[2];

    uint8_t* out = output.data<uint8_t>();

    auto& gemAvB_ops = gemAvB_BF16xBF16;
    auto& qKtrGemm_ops = qKtrGemm_BF16xBF16;
    auto& qKVGemm_ops = qKVGemm_BF16xBF16;
    bool use_vector = query_seq_len == 1 && head_size >= 32 && head_size <= 32 * 6 && !is_bloom && !alibi && attn_mask;
    size_t head_stride_in_attn = head_size;
    size_t batch_stride_in_attn = head_size * head_num * query_seq_len;
    size_t causal_mask_offset_start = use_causal_mask ? key_seq_len - query_seq_len : key_seq_len;

    if (use_vector) {
        parallel_for2d(batch, head_num, [&](size_t thread_id, size_t i0, size_t i1) {
            auto q_sub = &q.at<uint8_t>({i0, i1});
            auto k_sub = &k.at<uint8_t>({i0, i1});
            auto v_sub = &v.at<uint8_t>({i0, i1});

            auto mat0_out = reinterpret_cast<uint8_t*>(_buffer_mat0_out.get() + thread_id * _buffer_mat0_out_size);
            auto mat1_out = reinterpret_cast<uint8_t*>(_buffer_mat1_out.get() + thread_id * _buffer_mat1_out_size);

            tensor2D<ov::bfloat16> matK(key_seq_len, head_size, reinterpret_cast<ov::bfloat16*>(k_sub), k.m_strides[2]);
            // N: key_seq_len, K: head_size
            // q[1, K] * transpose(k[N, K])        ==>
            //     k[N, K] * transpose(q[1, K])    ==>
            //     k[N, K] * q[K, 1]
            (*gemAvB_ops[thread_id])(matK, reinterpret_cast<ov::bfloat16*>(q_sub), reinterpret_cast<float*>(mat0_out));

            float* pMatMul0Out = reinterpret_cast<float*>(mat0_out);
            mul_add_f32_avx512(pMatMul0Out, pMatMul0Out, normal_factor, &attn_mask.at<float>({i0}), key_seq_len);
            softmax_avx512<ov::bfloat16>(reinterpret_cast<ov::bfloat16*>(pMatMul0Out), pMatMul0Out, key_seq_len, nullptr);
            auto out_sub = out + (i0 * batch_stride_in_attn + i1 * head_stride_in_attn) * sizeof(ov::bfloat16);
            tensor2D<ov::bfloat16> matQK(query_seq_len, key_seq_len, reinterpret_cast<ov::bfloat16*>(mat0_out), rndup(key_seq_len * sizeof(ov::bfloat16), 64));
            tensor2D<ov::bfloat16> matV(key_seq_len, head_size, reinterpret_cast<ov::bfloat16*>(v_sub), v.m_strides[2]);
            tensor2D<float> matQKV(query_seq_len, head_size, reinterpret_cast<float*>(mat1_out), _head_size_aligned * sizeof(float));
            amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(matQKV);
            (*qKVGemm_ops[thread_id])(matQK, matV, 0, head_size, pp);
            memcpy2d_stride_avx512<ov::bfloat16>(reinterpret_cast<ov::bfloat16*>(out_sub), reinterpret_cast<float*>(mat1_out), query_seq_len,
                head_size, _head_size_aligned * sizeof(float), head_num * head_size * sizeof(ov::bfloat16), nullptr);
        });
    } else {
        size_t seq_cout_all = rndup(query_seq_len, 32) / 32;
        auto work_amount = batch * head_num * seq_cout_all;
        parallel_for(_num_threads, [&](size_t thread_id) {
            size_t i0;
            size_t i1;
            size_t seq;
            size_t start {0}, end {0};
            splitter(work_amount, _num_threads, thread_id, start, end);
            if (start >= work_amount) return;

            parallel_it_init(start, i0, batch, i1, head_num, seq, seq_cout_all);
            ov::bfloat16* prev_k = nullptr;
            ov::bfloat16* prev_v = nullptr;
            for (int iwork = start; iwork < end; ++iwork) {
                auto seq_start = seq * 32;
                auto seq_end = std::min(seq_start + 32, query_seq_len);
                auto seq_cout = seq_end - seq_start;
                // q: [batch, head_num, query_seq_len, head_size]
                // k: [batch, head_num, key_seq_len, head_size]
                // v: [batch, head_num, value_seq_len, head_size]
                auto q_sub = &q.at<ov::bfloat16>({i0, i1, seq_start});
                auto k_sub = &k.at<ov::bfloat16>({i0, i1});
                auto v_sub = &v.at<ov::bfloat16>({i0, i1});

                auto mat0_out = reinterpret_cast<float*>(_buffer_mat0_out.get() + thread_id * _buffer_mat0_out_size);
                auto mat1_out = reinterpret_cast<float*>(_buffer_mat1_out.get() + thread_id * _buffer_mat1_out_size);
                
                tensor2D<ov::bfloat16> matQ(seq_cout, head_size, q_sub, q.m_strides[2]);
                tensor2D<float> matQK(seq_cout, key_seq_len, mat0_out, rndup(key_seq_len * sizeof(float), 64));
                amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(matQK);
                if (!is_bloom) {
                    tensor2D<ov::bfloat16> matK(key_seq_len, head_size, k_sub, k.m_strides[2]);
                    (*qKtrGemm_ops[thread_id])(matQ, matK, 0, key_seq_len, pp, k_sub == prev_k);
                } else {
                    tensor2D<ov::bfloat16> matK(head_size, key_seq_len, k_sub, k.m_strides[3]);
                    (*qKtrGemm_ops[thread_id])(matQ, matK, 0, key_seq_len, pp, k_sub == prev_k);
                }
                prev_k = k_sub;
                tensor2D<ov::bfloat16> softmax_dst(seq_cout, key_seq_len, reinterpret_cast<ov::bfloat16*>(mat0_out), rndup(key_seq_len * sizeof(ov::bfloat16), 64));
                // no attention mask
                size_t valid_softmax_items = std::min(causal_mask_offset_start + seq_start + 1, key_seq_len);
                if (!attn_mask) {
                    for (int m = 0; m < seq_cout; m++) {
                        float* src = &matQK(m, 0);
                        ov::bfloat16* dst = &softmax_dst(m, 0);
                        if (!alibi)
                            mul_f32_avx512(src, src, normal_factor, valid_softmax_items);
                        else
                            // alibi shape: [batch, head_num, 1, key_seq_len]
                            mul_add_f32_avx512(src, src, normal_factor,
                                &alibi.at<float>({i0, i1}),
                                valid_softmax_items);
                        softmax_avx512<ov::bfloat16>(dst, src, valid_softmax_items, nullptr);
                        if (key_seq_len > valid_softmax_items) {
                            auto *invalidPtr = dst + valid_softmax_items;
                            memset(static_cast<void*>(invalidPtr), 0, (key_seq_len - valid_softmax_items) * sizeof(ov::bfloat16));
                            valid_softmax_items = std::min(valid_softmax_items + 1, key_seq_len);
                        }
                    }
                } else {
                    // [batch, 1, 1, key_seq_len] or [batch, 1, query_seq_len, key_seq_len]
                    for (int m = 0; m < seq_cout; m++) {
                        auto attn_sub = &attn_mask.at<float>({i0, 0, attn_mask.m_dims[2] == 1 ? 0 : m + seq_start}); // s  + i0 * key_seq_len * query_seq_len;
                        float* src = &matQK(m, 0);
                        ov::bfloat16* dst = &softmax_dst(m, 0);
                        if (!alibi)
                            mul_add_f32_avx512(src, src, normal_factor, attn_sub, valid_softmax_items);
                        else
                            // alibi shape: [batch, head_num, 1, key_seq_len]
                            mul_add2_f32_avx512(src, src, normal_factor,
                                &alibi.at<float>({i0, i1}),
                                attn_sub,
                                valid_softmax_items);
                        softmax_avx512<ov::bfloat16>(dst, src, valid_softmax_items, nullptr);
                        if (key_seq_len > valid_softmax_items) {
                            auto *invalidPtr = dst + valid_softmax_items;
                            memset(static_cast<void*>(invalidPtr), 0, (key_seq_len - valid_softmax_items) * sizeof(ov::bfloat16));
                            valid_softmax_items = std::min(valid_softmax_items + 1, key_seq_len);
                        }
                    }
                }

                auto out_sub = out + (i0 * batch_stride_in_attn + i1 * head_stride_in_attn
                    + seq_start * head_stride_in_attn * head_num) * sizeof(ov::bfloat16);
                tensor2D<ov::bfloat16> matQKBF16(seq_cout, key_seq_len, softmax_dst.data, softmax_dst.stride);
                tensor2D<ov::bfloat16> matV(key_seq_len, head_size, v_sub, v.m_strides[2]);
                tensor2D<float> matQKV(seq_cout, head_size, mat1_out, _head_size_aligned * sizeof(float));
                amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp2(matQKV);
                (*qKVGemm_ops[thread_id])(matQKBF16, matV, 0, head_size, pp2, prev_v == v_sub);
                prev_v = v_sub;
                memcpy2d_stride_avx512<ov::bfloat16>(reinterpret_cast<ov::bfloat16*>(out_sub), mat1_out, seq_cout,
                    head_size, _head_size_aligned * sizeof(float), head_num * head_size * sizeof(ov::bfloat16), nullptr);
                parallel_it_step(i0, batch, i1, head_num, seq, seq_cout_all);
            }
        });
    }
}

void mha_gpt_impl_amx::exec(const tensor& q, const tensor& k, const tensor& v, const tensor& output, const tensor& attn_mask, const tensor& alibi, float normal_factor, bool use_causal_mask) {
    if (q.m_rank != 4 || k.m_rank != 4 || v.m_rank != 4) {
        std::cout << "q,k,v rank does not equal 4.\n";
        return;
    }
    if (output.m_rank != 3) {
        std::cout << "output rank should be 3.\n";
    }
    if (attn_mask) {
        if (attn_mask.m_rank != 4) {
            std::cout << "attn_mask rank should be 4.\n";
            return;
        }
        if (attn_mask.m_dims[1] != 1) {
            std::cout << "attn_mask dim 1 should be 1.\n";
            return;
        }
    }
    if (alibi) {
        if (alibi.m_rank != 4) {
            std::cout << "alibi rank should be 4.\n";
            return;
        }
        if (alibi.m_dims[1] != k.m_dims[1]) {
            std::cout << "alibi dim 1 should be equal to k dim 1.\n";
            return;
        }
        if (alibi.m_dims[2] != 1) {
            std::cout << "alibi dim 2 should be 1.\n";
            return;
        }
    }
    auto batch = q.m_dims[0];
    auto head_num = q.m_dims[1];
    auto query_seq_len = q.m_dims[2];
    auto head_size = q.m_dims[3];
    auto key_seq_len = k.m_dims[2];

    if (!(batch == k.m_dims[0] && batch == v.m_dims[0] &&
          head_num == k.m_dims[1] && head_num == v.m_dims[1] &&
          key_seq_len == v.m_dims[2] &&
          head_size == k.m_dims[3] && head_size == v.m_dims[3])) {
        std::cout << "dim of q,k,v is error.\n";
        return;
    }

    bool is_bloom = k.m_strides[3] > k.m_strides[2];

    auto in_dtype = q.m_dtype;
    auto out_dtype = output.m_dtype;

    if (in_dtype == dnnl_bf16 && out_dtype == dnnl_bf16) {
        create(in_dtype, key_seq_len, head_size, is_bloom);
        mha_bf16(q, k, v, output, attn_mask, alibi, normal_factor, use_causal_mask);
    } else {
        std::cout << "doesn't support provided input precisions.\n";
    }
}

mha_gpt::impl* new_impl_amx() {
    return new mha_gpt_impl_amx();
}

}