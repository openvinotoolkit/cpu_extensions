// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <climits>
#include <cstdint>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <map>
#include <cassert>

#include "llm_fc.hpp"
#include "llm_types.hpp"
#include "mm_kernel_common_amx.hpp"
#include "utility_kernel_avx512.hpp"
#include "fc_kernel_amx.hpp"
#include "common/compatible.hpp"

namespace llmdnn {

using ov::bfloat16;
struct fc_kernel {
    std::unique_ptr<amx_kernel::Matmul<ov::bfloat16, ov::bfloat16>> bf16xbf16;
    std::unique_ptr<amx_kernel::Matmul<ov::bfloat16, uint8_t>> bf16xi8;
    std::unique_ptr<amx_kernel::Matmul<int8_t, int8_t>> i8xi8;
    std::unique_ptr<amx_kernel::Matmul<uint8_t, int8_t>> u8xi8;

    data_type_t dt_a;
    data_type_t dt_b;
    data_type_t dt_c;
    size_t stride_b;
    postops_types postops_type;
    bool b_is_transpose;
};

using supported_key = std::tuple<data_type_t, data_type_t, data_type_t>;
using supported_value = std::pair<size_t, size_t>;
static bool check_valid_postops(size_t value, data_type_t dt_a, data_type_t dt_b, data_type_t dt_c) {
    llm_map<supported_key, supported_value> supported_postops = {
        { { llmdnn_s8, llmdnn_s8, llmdnn_s8 }, { DEQUANT | QUANT, BIAS | GELU | GELU_TANH } },
        { { llmdnn_s8, llmdnn_s8, llmdnn_bf16 }, { DEQUANT, BIAS | GELU | GELU_TANH } },
        { { llmdnn_s8, llmdnn_s8, llmdnn_f32 }, { DEQUANT, BIAS | GELU | GELU_TANH } },
        { { llmdnn_bf16, llmdnn_bf16, llmdnn_bf16 }, { 0, BIAS | GELU | GELU_TANH } },
        { { llmdnn_bf16, llmdnn_bf16, llmdnn_f32 }, { 0, BIAS | GELU | GELU_TANH } },
        { { llmdnn_bf16, llmdnn_u8, llmdnn_f32 }, { DEQUANT, BIAS | GELU | GELU_TANH } },
        { { llmdnn_bf16, llmdnn_u8, llmdnn_bf16 }, { DEQUANT, BIAS | GELU | GELU_TANH } },
    };

    auto it = supported_postops.find(std::make_tuple(dt_a, dt_b, dt_c));
    if (it == supported_postops.end()) {
        return false;
    }
    
    size_t must_have;
    size_t opt_have;
    must_have = (*it).second.first;
    opt_have = (*it).second.second;

    if ((value & must_have) != must_have)
        return false;
    // value must in must_have and opt_have
    if ((value & ~(must_have | opt_have)) != 0)
        return false;
    
    return true;
}

// interface
status_t fc_kernel_create_amx(fc_kernel** mm, const fc_create_param* param) {
    fc_kernel* m = nullptr;
    if (param == nullptr || mm == nullptr) {
        DEBUG_LOG << "fc_kernel_create: invalid input parameter.\n";
        goto ERR;
    }

    if (!check_valid_postops(static_cast<size_t>(param->postops_type), param->dt_a, param->dt_b, param->dt_c)) {
        DEBUG_LOG << "fc_kernel_create: unsupported data type, a: " << param->dt_a <<", b: " << param->dt_b << ", c: " << param->dt_c <<
            ", postops type: " << param->postops_type << ".\n";
        goto ERR;
    }

    m = new fc_kernel;
    if (param->dt_a == llmdnn_s8 && param->dt_b == llmdnn_s8) {
        m->i8xi8 = std::make_unique<amx_kernel::Matmul<int8_t, int8_t>>(true, param->b_is_trans);
    } else if (param->dt_a == llmdnn_u8 && param->dt_b == llmdnn_s8) {
        m->u8xi8 = std::make_unique<amx_kernel::Matmul<uint8_t, int8_t>>(true, param->b_is_trans);
    } else if (param->dt_a == llmdnn_bf16 && (param->dt_b == llmdnn_bf16 || param->dt_b == llmdnn_f32)) {
        m->bf16xbf16 = std::make_unique<amx_kernel::Matmul<bfloat16, bfloat16>>(true, param->b_is_trans);
    } else if (param->dt_a == llmdnn_bf16 && param->dt_b == llmdnn_u8) {
        m->bf16xi8 = std::make_unique<amx_kernel::Matmul<bfloat16, uint8_t>>(true, param->b_is_trans);
        m->bf16xi8->dequant_scale_B = param->scale;
        m->bf16xi8->zp = param->zp;
    } else {
        DEBUG_LOG << "fc_kernel_create: unsupport input type, a: " << param->dt_a << ", b: " << param->dt_b << ".\n";
        goto ERR;
    }

    m->dt_a = param->dt_a;
    m->dt_b = param->dt_b;
    m->dt_c = param->dt_c;
    m->b_is_transpose = param->b_is_trans;
    m->postops_type = param->postops_type;

    *mm = m;
    return status_t::status_ok;
ERR:
    delete m;
    return status_t::status_invalid_arguments;
}

void fc_kernel_destroy_amx(fc_kernel* mm) {
    if (mm) {
        delete mm;
    }
}

void fc_kernel_pack_weight_amx(fc_kernel* mm, void* ptr_b, data_type_t dt_b, size_t N, size_t K, size_t stride_b, size_t n_start, size_t n_end) {
    mm->stride_b = stride_b;
    size_t b_d0 = K, b_d1 = N;
    if (mm->b_is_transpose) {
        b_d0 = N;
        b_d1 = K;
    }
    if (mm->i8xi8) {
        tensor2D<int8_t> b(b_d0, b_d1, static_cast<int8_t*>(ptr_b), mm->stride_b);
        auto matB = amx_kernel::getSubMatB(b, n_start, n_end, mm->b_is_transpose);
        amx_kernel::repackB_1x2(matB, mm->b_is_transpose, mm->i8xi8->internalB, true);
    } else if (mm->u8xi8) {
        tensor2D<int8_t> b(b_d0, b_d1, static_cast<int8_t*>(ptr_b), mm->stride_b);
        auto matB = amx_kernel::getSubMatB(b, n_start, n_end, mm->b_is_transpose);
        amx_kernel::repackB_1x2(matB, mm->b_is_transpose, mm->u8xi8->internalB, true);
    } else if (mm->bf16xbf16) {
        if (dt_b == llmdnn_bf16) {
            tensor2D<bfloat16> b(b_d0, b_d1, static_cast<bfloat16*>(ptr_b), mm->stride_b);
            auto matB = amx_kernel::getSubMatB(b, n_start, n_end, mm->b_is_transpose);
            amx_kernel::repackB_1x2(matB, mm->b_is_transpose, mm->bf16xbf16->internalB, true);
        } else {
            tensor2D<float> b(b_d0, b_d1, static_cast<float*>(ptr_b), mm->stride_b);
            auto matB = amx_kernel::getSubMatB(b, n_start, n_end, mm->b_is_transpose);
            amx_kernel::repackB_1x2(matB, mm->b_is_transpose, mm->bf16xbf16->internalB, true);
        }
    } else {
        assert(dt_b == llmdnn_u8);
        tensor2D<uint8_t> b(b_d0, b_d1, static_cast<uint8_t*>(ptr_b), mm->stride_b);
        auto matB = amx_kernel::getSubMatB(b, n_start, n_end, mm->b_is_transpose);
        amx_kernel::repackB_1x2_compressed(matB, mm->b_is_transpose, mm->bf16xi8->internalBI8, true);
    }
}

void fc_kernel_pack_weight_to_dst_amx(fc_kernel* mm, void* src_b, void* dst_b, data_type_t dt_b, size_t N, size_t K, size_t stride_b, size_t n_start, size_t n_end) {
    mm->stride_b = stride_b;
    size_t b_d0 = K, b_d1 = N;
    if (mm->b_is_transpose) {
        b_d0 = N;
        b_d1 = K;
    }
    if (mm->i8xi8) {
        tensor2D<int8_t> b(b_d0, b_d1, static_cast<int8_t*>(src_b), mm->stride_b);
        // do not care about the real dimension, only ensure .capacity big enough
        mm->i8xi8->internalB = tensor2D<int8_t>(1, 1, static_cast<int8_t*>(dst_b), 1);
        mm->i8xi8->internalB.capacity = INT_MAX;
        auto matB = amx_kernel::getSubMatB(b, n_start, n_end, mm->b_is_transpose);
        amx_kernel::repackB_1x2(matB, mm->b_is_transpose, mm->i8xi8->internalB, true);
    } else if (mm->u8xi8) {
        tensor2D<int8_t> b(b_d0, b_d1, static_cast<int8_t*>(src_b), mm->stride_b);
        mm->u8xi8->internalB = tensor2D<int8_t>(1, 1, static_cast<int8_t*>(dst_b), 1);
        mm->u8xi8->internalB.capacity = INT_MAX;
        auto matB = amx_kernel::getSubMatB(b, n_start, n_end, mm->b_is_transpose);
        amx_kernel::repackB_1x2(matB, mm->b_is_transpose, mm->u8xi8->internalB, true);
    } else if (mm->bf16xbf16) {
        mm->bf16xbf16->internalB = tensor2D<bfloat16>(1, 1, static_cast<bfloat16*>(dst_b), 1);
        mm->bf16xbf16->internalB.capacity = INT_MAX;
        if (dt_b == llmdnn_bf16) {
            tensor2D<bfloat16> b(b_d0, b_d1, static_cast<bfloat16*>(src_b), mm->stride_b);
            auto matB = amx_kernel::getSubMatB(b, n_start, n_end, mm->b_is_transpose);
            amx_kernel::repackB_1x2(matB, mm->b_is_transpose, mm->bf16xbf16->internalB, true);
        } else {
            tensor2D<float> b(b_d0, b_d1, static_cast<float*>(src_b), mm->stride_b);
            auto matB = amx_kernel::getSubMatB(b, n_start, n_end, mm->b_is_transpose);
            amx_kernel::repackB_1x2(matB, mm->b_is_transpose, mm->bf16xbf16->internalB, true);
        }
    } else {
        assert(dt_b == llmdnn_u8);
        mm->bf16xi8->internalBI8 = tensor2D<uint8_t>(1, 1, static_cast<uint8_t*>(dst_b), 1);
        mm->bf16xi8->internalBI8.capacity = INT_MAX;
        tensor2D<uint8_t> b(b_d0, b_d1, static_cast<uint8_t*>(src_b), mm->stride_b);
        auto matB = amx_kernel::getSubMatB(b, n_start, n_end, mm->b_is_transpose);
        amx_kernel::repackB_1x2_compressed(matB, mm->b_is_transpose, mm->bf16xi8->internalBI8, true);
    }
}

void fc_kernel_execute_amx(fc_kernel* mm, void* ptr_a, void* ptr_b, void* ptr_c, size_t stride_a, size_t stride_c,
        size_t M, size_t N, size_t K, size_t n_start, size_t n_end, float* dq, float* q, float* bias) {
    size_t b_d0 = K, b_d1 = N;
    if (mm->b_is_transpose) {
        b_d0 = N;
        b_d1 = K;
    }
    if (mm->i8xi8) {
        tensor2D<int8_t> a(M, K, reinterpret_cast<int8_t*>(ptr_a), stride_a);
        tensor2D<int8_t> b(b_d0, b_d1, nullptr, mm->stride_b);
        if (ptr_b) {
            auto K_padded = rndup(K, 64);
            mm->i8xi8->internalB = tensor2D<int8_t>(N / 32, 32 * K_padded, static_cast<int8_t*>(ptr_b), 32 * K_padded);
        }

        if (mm->dt_c == llmdnn_s8) {
            tensor2D<int8_t> c(M, N, reinterpret_cast<int8_t*>(ptr_c), stride_c);
            if (!(mm->postops_type & BIAS)) {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<int8_t, amx_kernel::PP::Steps::DEQUANT_GELU_QUANT> ppkernel(c);
                    ppkernel.set_deq_scale(dq);
                    ppkernel.set_q_scale(q);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                } else if (mm->postops_type & GELU_TANH) {
                    amx_kernel::PP::BiasGeluStore<int8_t, amx_kernel::PP::Steps::DEQUANT_GELU_TANH_QUANT> ppkernel(c);
                    ppkernel.set_deq_scale(dq);
                    ppkernel.set_q_scale(q);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<int8_t, amx_kernel::PP::Steps::DEQUANT_QUANT> ppkernel(c);
                    ppkernel.set_deq_scale(dq);
                    ppkernel.set_q_scale(q);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                }
            } else {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<int8_t, amx_kernel::PP::Steps::DEQUANT_BIAS_GELU_QUANT> ppkernel(c, bias);
                    ppkernel.set_deq_scale(dq);
                    ppkernel.set_q_scale(q);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                } else if (mm->postops_type & GELU_TANH) {
                    amx_kernel::PP::BiasGeluStore<int8_t, amx_kernel::PP::Steps::DEQUANT_BIAS_GELU_TANH_QUANT> ppkernel(c, bias);
                    ppkernel.set_deq_scale(dq);
                    ppkernel.set_q_scale(q);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<int8_t, amx_kernel::PP::Steps::DEQUANT_BIAS_QUANT> ppkernel(c, bias);
                    ppkernel.set_deq_scale(dq);
                    ppkernel.set_q_scale(q);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                }
            }
        } else if (mm->dt_c == llmdnn_bf16) {
            tensor2D<bfloat16> c(M, N, reinterpret_cast<bfloat16*>(ptr_c), stride_c);
            if (!bias) {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<bfloat16, amx_kernel::PP::Steps::DEQUANT_GELU> ppkernel(c);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                } else if (mm->postops_type & GELU_TANH) {
                    amx_kernel::PP::BiasGeluStore<bfloat16, amx_kernel::PP::Steps::DEQUANT_GELU_TANH> ppkernel(c);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<bfloat16, amx_kernel::PP::Steps::DEQUANT> ppkernel(c);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                }
            } else {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<bfloat16, amx_kernel::PP::Steps::DEQUANT_BIAS_GELU> ppkernel(c, bias);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                } else if (mm->postops_type & GELU_TANH) {
                    amx_kernel::PP::BiasGeluStore<bfloat16, amx_kernel::PP::Steps::DEQUANT_BIAS_GELU_TANH> ppkernel(c, bias);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<bfloat16, amx_kernel::PP::Steps::DEQUANT_BIAS> ppkernel(c, bias);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                }
            }
        } else if (mm->dt_c == llmdnn_f32) {
            tensor2D<float> c(M, N, reinterpret_cast<float*>(ptr_c), stride_c);
            if (!bias) {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT_GELU> ppkernel(c);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                } else if (mm->postops_type & GELU_TANH) {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT_GELU_TANH> ppkernel(c);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT> ppkernel(c);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                }
            } else {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT_BIAS_GELU> ppkernel(c, bias);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                } else if (mm->postops_type & GELU_TANH) {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT_BIAS_GELU_TANH> ppkernel(c, bias);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT_BIAS> ppkernel(c, bias);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                }
            }
        }
    } else if (mm->u8xi8) {
        tensor2D<uint8_t> a(M, K, reinterpret_cast<uint8_t*>(ptr_a), stride_a);
        tensor2D<int8_t> b(b_d0, b_d1, nullptr, mm->stride_b);
        tensor2D<float> c(M, N, reinterpret_cast<float*>(ptr_c), stride_c);
        amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(c);
        (*mm->u8xi8)(a, b, n_start, n_end, pp);
    } else if (mm->bf16xbf16) {
        tensor2D<bfloat16> a(M, K, reinterpret_cast<bfloat16*>(ptr_a), stride_a);
        tensor2D<bfloat16> b(b_d0, b_d1, nullptr, mm->stride_b);
        if (ptr_b) {
            auto K_padded = rndup(K, 32);
            mm->bf16xbf16->internalB = tensor2D<bfloat16>(N / 32, 32 * K_padded, static_cast<bfloat16*>(ptr_b), 32 * K_padded * sizeof(bfloat16));
        }
        if (mm->dt_c == llmdnn_bf16) {
            tensor2D<bfloat16> c(M, N, reinterpret_cast<bfloat16*>(ptr_c), stride_c);
            if (!(mm->postops_type & BIAS)) {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::GELU> ppkernel(c);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                } else if (mm->postops_type & GELU_TANH) {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::GELU_TANH> ppkernel(c);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::NONE> ppkernel(c);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                }
            } else {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::BIAS_GELU> ppkernel(c, bias);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                } else if (mm->postops_type & GELU_TANH) {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::BIAS_GELU_TANH> ppkernel(c, bias);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::BIAS> ppkernel(c, bias);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                }
            }
        } else if (mm->dt_c == llmdnn_f32) {
            tensor2D<float> c(M, N, reinterpret_cast<float*>(ptr_c), stride_c);
            if (!(mm->postops_type & BIAS)) {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::GELU> ppkernel(c);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                } else if (mm->postops_type & GELU_TANH) {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::GELU_TANH> ppkernel(c);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> ppkernel(c);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                }
            } else {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::BIAS_GELU> ppkernel(c, bias);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                } else if (mm->postops_type & GELU_TANH) {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::BIAS_GELU_TANH> ppkernel(c, bias);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::BIAS> ppkernel(c, bias);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                }
            }
        }
    } else {
        tensor2D<bfloat16> a(M, K, reinterpret_cast<bfloat16*>(ptr_a), stride_a);
        tensor2D<bfloat16> b(b_d0, b_d1, nullptr, mm->stride_b);

        if (ptr_b) {
            auto K_padded = rndup(K, 32);
            mm->bf16xi8->internalBI8 = tensor2D<uint8_t>(N / 32, 32 * K_padded, static_cast<uint8_t*>(ptr_b), 32 * K_padded);
        }

        if (mm->dt_c == llmdnn_bf16) {
            tensor2D<bfloat16> c(M, N, reinterpret_cast<bfloat16*>(ptr_c), stride_c);
            if (!(mm->postops_type & BIAS)) {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::DEQUANT_GELU> ppkernel(c);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                } else if (mm->postops_type & GELU_TANH) {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::DEQUANT_GELU_TANH> ppkernel(c);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::DEQUANT> ppkernel(c);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                }
            } else {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::DEQUANT_BIAS_GELU> ppkernel(c, bias);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                } else if (mm->postops_type & GELU_TANH) {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::DEQUANT_BIAS_GELU_TANH> ppkernel(c, bias);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::DEQUANT_BIAS> ppkernel(c, bias);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                }
            }
        } else if (mm->dt_c == llmdnn_f32) {
            tensor2D<float> c(M, N, reinterpret_cast<float*>(ptr_c), stride_c);
            if (!(mm->postops_type & BIAS)) {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT_GELU> ppkernel(c);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                } else if (mm->postops_type & GELU_TANH) {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT_GELU_TANH> ppkernel(c);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT> ppkernel(c);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                }
            } else {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT_BIAS_GELU> ppkernel(c, bias);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                } else if (mm->postops_type & GELU_TANH) {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT_BIAS_GELU_TANH> ppkernel(c, bias);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT_BIAS> ppkernel(c, bias);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                }
            }
        }
    }
}

void fc_kernel_bf16w8_get_q_dq_amx(size_t K, size_t N, size_t stride, void* ptr, float* q, float* dq) {
    float min, max;
    tensor2D<bfloat16> B(K, N, reinterpret_cast<bfloat16*>(ptr), stride);
    amx_kernel::functional::get_min_max(B, min, max);
    max = std::max(std::abs(max), std::abs(min));
    *q = 127 / max;
    *dq = max / 127;
}

}  // namespace llmdnn
