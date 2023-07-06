// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdint.h>
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <immintrin.h>
#endif
#include "common/bf16.hpp"
#include "llm_types.hpp"
#include "utility_kernel_avx2.hpp"

namespace llmdnn {
    inline void rotary_avx2(size_t N, float* cos, float* sin, float* q_src, float* k_src, float* q_dst, float* k_dst) {
        auto half = N / 2;
        // for (size_t i = 0; i < half; i++) {
        //     q_dst[i] = q_src[i] * cos[i] - q_src[i + half] * sin[i];
        //     k_dst[i] = k_src[i] * cos[i] - k_src[i + half] * sin[i];
        // }
        // for (size_t i = half; i < N; i++) {
        //     q_dst[i] = q_src[i] * cos[i] + q_src[i - half] * sin[i];
        //     k_dst[i] = k_src[i] * cos[i] + k_src[i - half] * sin[i];
        // }
        size_t tail = half % 8;
        auto x_mask = get_mask(tail);
        size_t i;
        for (i = 0; i < half - tail; i += 8) {
            auto q_f = _mm256_loadu_ps(q_src + i + half);
            auto k_f = _mm256_loadu_ps(k_src + i + half);
            auto cos_f = _mm256_loadu_ps(cos + i);
            auto sin_f = _mm256_loadu_ps(sin + i);
            auto q_dst_f = _mm256_mul_ps(q_f, sin_f);
            auto k_dst_f = _mm256_mul_ps(k_f, sin_f);

            q_f = _mm256_loadu_ps(q_src + i);
            k_f = _mm256_loadu_ps(k_src + i);

            q_dst_f = _mm256_fmsub_ps(q_f, cos_f, q_dst_f);
            k_dst_f = _mm256_fmsub_ps(k_f, cos_f, k_dst_f);

            _mm256_storeu_ps(q_dst + i, q_dst_f);
            _mm256_storeu_ps(k_dst + i, k_dst_f);
        }
        if (tail) {
            auto q_f = _mm256_maskload_ps(q_src + i + half, x_mask);
            auto k_f = _mm256_maskload_ps(k_src + i + half, x_mask);
            auto cos_f = _mm256_maskload_ps(cos + i, x_mask);
            auto sin_f = _mm256_maskload_ps(sin + i, x_mask);
            auto q_dst_f = _mm256_mul_ps(q_f, sin_f);
            auto k_dst_f = _mm256_mul_ps(k_f, sin_f);

            q_f = _mm256_maskload_ps(q_src + i, x_mask);
            k_f = _mm256_maskload_ps(k_src + i, x_mask);

            q_dst_f = _mm256_fmsub_ps(q_f, cos_f, q_dst_f);
            k_dst_f = _mm256_fmsub_ps(k_f, cos_f, k_dst_f);

            _mm256_maskstore_ps(q_dst + i, x_mask, q_dst_f);
            _mm256_maskstore_ps(k_dst + i, x_mask, k_dst_f);
        }
        // second half
        q_src += half;
        k_src += half;
        cos += half;
        sin += half;
        q_dst += half;
        k_dst += half;
        for (i = 0; i < half - tail; i += 8) {
            auto q_f = _mm256_loadu_ps(q_src + i - half);
            auto k_f = _mm256_loadu_ps(k_src + i - half);
            auto cos_f = _mm256_loadu_ps(cos + i);
            auto sin_f = _mm256_loadu_ps(sin + i);
            auto q_dst_f = _mm256_mul_ps(q_f, sin_f);
            auto k_dst_f = _mm256_mul_ps(k_f, sin_f);

            q_f = _mm256_loadu_ps(q_src + i);
            k_f = _mm256_loadu_ps(k_src + i);

            q_dst_f = _mm256_fmadd_ps(q_f, cos_f, q_dst_f);
            k_dst_f = _mm256_fmadd_ps(k_f, cos_f, k_dst_f);

            _mm256_storeu_ps(q_dst + i, q_dst_f);
            _mm256_storeu_ps(k_dst + i, k_dst_f);
        }
        if (tail) {
            auto q_f = _mm256_maskload_ps(q_src + i - half, x_mask);
            auto k_f = _mm256_maskload_ps(k_src + i - half, x_mask);
            auto cos_f = _mm256_maskload_ps(cos + i, x_mask);
            auto sin_f = _mm256_maskload_ps(sin + i, x_mask);
            auto q_dst_f = _mm256_mul_ps(q_f, sin_f);
            auto k_dst_f = _mm256_mul_ps(k_f, sin_f);

            q_f = _mm256_maskload_ps(q_src + i, x_mask);
            k_f = _mm256_maskload_ps(k_src + i, x_mask);

            q_dst_f = _mm256_fmadd_ps(q_f, cos_f, q_dst_f);
            k_dst_f = _mm256_fmadd_ps(k_f, cos_f, k_dst_f);

            _mm256_maskstore_ps(q_dst + i, x_mask, q_dst_f);
            _mm256_maskstore_ps(k_dst + i, x_mask, k_dst_f);
        }
    }
}