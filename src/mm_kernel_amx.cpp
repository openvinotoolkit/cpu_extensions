// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <memory>
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <cassert>

#include "llm_mm.hpp"
#include "llm_types.hpp"
#include "mm_kernel_common_amx.hpp"
#include "utility_kernel_avx512.hpp"
#include "mm_kernel_amx.hpp"

namespace llmdnn {

using ov::bfloat16;
struct mm_kernel {
    std::unique_ptr<amx_kernel::Matmul<ov::bfloat16, ov::bfloat16>> bf16xbf16;
    std::unique_ptr<amx_kernel::Matmul<int8_t, int8_t>> i8xi8;
    std::unique_ptr<amx_kernel::Matmul<uint8_t, int8_t>> u8xi8;

    std::unique_ptr<amx_kernel::MatmulVector<int8_t, int8_t>> i8xi8_gemv;
    std::unique_ptr<amx_kernel::MatmulVector<ov::bfloat16, ov::bfloat16>> bf16xbf16_gemv;

    data_type_t dt_a;
    data_type_t dt_b;
    bool b_is_transpose;
};

// interface
status_t mm_kernel_create_amx(mm_kernel** mm, const mm_create_param* param) {
    mm_kernel* m = nullptr;
    if (param == nullptr || mm == nullptr) {
        DEBUG_LOG << "mm_kernel_create: invalid input parameter.\n";
        goto ERR;
    }

    m = new mm_kernel;
    if (param->b_is_gemv) {
        if (param->dt_a == llmdnn_s8 && param->dt_b == llmdnn_s8) {
            m->i8xi8_gemv = std::make_unique<amx_kernel::MatmulVector<int8_t, int8_t>>();
        } else if (param->dt_a == llmdnn_bf16 && param->dt_b == llmdnn_bf16) {
            m->bf16xbf16_gemv = std::make_unique<amx_kernel::MatmulVector<bfloat16, bfloat16>>();
        } else {
            DEBUG_LOG << "mm_kernel_create: unsupport gemv input type, a: " << param->dt_a << ", b: " << param->dt_b << ".\n";
            goto ERR;
        }
    } else {
        if (param->dt_a == llmdnn_s8 && param->dt_b == llmdnn_s8) {
            m->i8xi8 = std::make_unique<amx_kernel::Matmul<int8_t, int8_t>>(false, param->b_is_trans);
        } else if (param->dt_a == llmdnn_u8 && param->dt_b == llmdnn_s8) {
            m->u8xi8 = std::make_unique<amx_kernel::Matmul<uint8_t, int8_t>>(false, param->b_is_trans);
        } else if (param->dt_a == llmdnn_bf16 && param->dt_b == llmdnn_bf16) {
            m->bf16xbf16 = std::make_unique<amx_kernel::Matmul<bfloat16, bfloat16>>(false, param->b_is_trans);
        } else {
            DEBUG_LOG << "mm_kernel_create: unsupport input type, a: " << param->dt_a << ", b: " << param->dt_b << ".\n";
            goto ERR;
        }
    }
    m->dt_a = param->dt_a;
    m->dt_b = param->dt_b;
    m->b_is_transpose = param->b_is_trans;

    *mm = m;
    return status_t::status_ok;
ERR:
    delete m;
    return status_t::status_invalid_arguments;
}

void mm_kernel_destroy_amx(const mm_kernel* mm) {
    if (mm) {
        delete mm;
    }
}

status_t mm_kernel_execute_amx(const mm_kernel* mm, void* ptr_a, void* ptr_b, void* ptr_c, size_t lda, size_t ldb, size_t ldc,
        size_t M, size_t N, size_t K) {
    size_t b_d0 = K, b_d1 = N;
    if (mm->b_is_transpose) {
        b_d0 = N;
        b_d1 = K;
    }
    if (mm->i8xi8_gemv) {
        tensor2D<int8_t> a(M, K, reinterpret_cast<int8_t*>(ptr_a), lda);
        (*mm->i8xi8_gemv)(a, reinterpret_cast<int8_t*>(ptr_b), reinterpret_cast<int32_t*>(ptr_c));
        cvt_i32_f32_avx512(reinterpret_cast<float*>(ptr_c), reinterpret_cast<int32_t*>(ptr_c), M);
    } else if (mm->i8xi8) {
        tensor2D<int8_t> a(M, K, reinterpret_cast<int8_t*>(ptr_a), lda);
        tensor2D<int8_t> b(b_d0, b_d1, reinterpret_cast<int8_t*>(ptr_b), ldb);
        tensor2D<float> c(M, N, reinterpret_cast<float*>(ptr_c), ldc);
        amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(c);
        (*mm->i8xi8)(a, b, 0, N, pp);
    } else if (mm->u8xi8) {
        tensor2D<uint8_t> a(M, K, reinterpret_cast<uint8_t*>(ptr_a), lda);
        tensor2D<int8_t> b(b_d0, b_d1, reinterpret_cast<int8_t*>(ptr_b), ldb);
        tensor2D<float> c(M, N, reinterpret_cast<float*>(ptr_c), ldc);
        amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(c);
        (*mm->u8xi8)(a, b, 0, N, pp);
    } else if (mm->bf16xbf16_gemv) {
        tensor2D<bfloat16> a(M, K, reinterpret_cast<bfloat16*>(ptr_a), lda);
        (*mm->bf16xbf16_gemv)(a, reinterpret_cast<bfloat16*>(ptr_b), reinterpret_cast<float*>(ptr_c));
    } else if (mm->bf16xbf16) {
        tensor2D<bfloat16> a(M, K, reinterpret_cast<bfloat16*>(ptr_a), lda);
        tensor2D<bfloat16> b(b_d0, b_d1, reinterpret_cast<bfloat16*>(ptr_b), ldb);
        tensor2D<float> c(M, N, reinterpret_cast<float*>(ptr_c), ldc);
        amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(c);
        (*mm->bf16xbf16)(a, b, 0, N, pp);
    } else {
        DEBUG_LOG << "mm_kernel_execute: no valid kernel created, call create first.\n";
        return status_t::status_invalid_arguments;
    }

    return status_t::status_ok;
}

}  // namespace llmdnn
