// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdint.h>
#include "common/bf16.hpp"
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <immintrin.h>
#endif

namespace llmdnn {

/// Convert Packed BF16 Data to Packed float Data.
///
/// \headerfile <x86intrin.h>
///
/// \param __A
///    A 256-bit vector of [16 x bfloat].
/// \returns A 512-bit vector of [16 x float] come from convertion of __A
static __inline__ __m512 _mm512_cvtpbh_ps(__m256bh __A) {
  return _mm512_castsi512_ps((__m512i)_mm512_slli_epi32(
      (__m512i)_mm512_cvtepi16_epi32((__m256i)__A), 16));
}

// Store masks. The highest bit in each byte indicates the byte to store.
alignas(16) const unsigned char masks[16][16] =
{
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00 }
};

inline void store_n(__m128i mm, unsigned int n, void* storage)
{
    _mm_maskmoveu_si128(mm, reinterpret_cast< const __m128i& >(masks[n]), static_cast< char* >(storage));
}

inline void quant_i8_avx512(void* dst, void* src, size_t ele_num, float scale) {
    size_t i = 0;
    auto* a = reinterpret_cast<ov::bfloat16*>(src);
    int8_t* d = reinterpret_cast<int8_t*>(dst);
    auto s = _mm512_set1_ps(scale);
    for (; i < ele_num / 16 * 16; i += 16) {
        auto a0 = _mm256_loadu_epi16(a);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        auto d_f = _mm512_mul_ps(a0_f, s);
        auto d_i = _mm512_cvtps_epi32(d_f);
        auto d_i8 = _mm512_cvtsepi32_epi8(d_i);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(d), d_i8);
        a += 16;
        d += 16;
    }
    if (i != ele_num) {
        // https://stackoverflow.com/questions/40391708/convert-16-bit-mask-mmask16-to-m128i-control-byte-mask-on-knl-xeon-phi-72
        __mmask16 msk = _cvtu32_mask16(0xFFFFu >> (16 - (ele_num % 16)));
        auto a0 = _mm256_maskz_loadu_epi16(msk, a);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        auto d_f = _mm512_mul_ps(a0_f, s);
        auto d_i = _mm512_cvtps_epi32(d_f);
        auto d_i8 = _mm512_cvtsepi32_epi8(d_i);
        store_n(d_i8, ele_num % 16, d);
    }
}

// NOTE: did not handle tail because there should be enough room
inline void cvt_i32_f32_avx512(float* dst, int32_t* src, size_t ele_num) {
    for (size_t i = 0; i < (ele_num + 15) / 16 * 16; i += 16) {
        auto a0 = _mm512_load_epi32(src);
        auto a_f = _mm512_cvtepi32_ps(a0);
        _mm512_storeu_ps(dst, a_f);
        src += 16;
        dst += 16;
    }
}

inline void mul_add_f32_avx512(float* dst, float* src, float mul, float* add, int ele_num) {
    auto mul_f = _mm512_set1_ps(mul);
    int i;
    auto tail = ele_num % 16;
    __mmask16 msk = _cvtu32_mask16(0xFFFFu >> (16 - tail));
    for (i = 0; i < ele_num - tail; i += 16) {
        auto a_f = _mm512_loadu_ps(src);
        auto add_f = _mm512_loadu_ps(add);
        _mm512_storeu_ps(dst, _mm512_fmadd_ps(a_f, mul_f, add_f));
        src += 16;
        dst += 16;
        add += 16;
    }
    if (tail) {
        auto a_f = _mm512_maskz_loadu_ps(msk, src);
        auto add_f = _mm512_maskz_loadu_ps(msk, add);
        _mm512_mask_storeu_ps(dst, msk, _mm512_fmadd_ps(a_f, mul_f, add_f));
    }
}

inline void mul_add2_f32_avx512(float* dst, float* src, float mul, float* add1, float* add2, int ele_num) {
    auto mul_f = _mm512_set1_ps(mul);
    int i;
    auto tail = ele_num % 16;
    __mmask16 msk = _cvtu32_mask16(0xFFFFu >> (16 - tail));
    for (i = 0; i < ele_num - tail; i += 16) {
        auto a_f = _mm512_loadu_ps(src);
        auto add1_f = _mm512_loadu_ps(add1);
        auto add2_f = _mm512_loadu_ps(add2);
        _mm512_storeu_ps(dst, _mm512_add_ps(_mm512_fmadd_ps(a_f, mul_f, add1_f), add2_f));
        src += 16;
        dst += 16;
        add1 += 16;
        add2 += 16;
    }
    if (tail) {
        auto a_f = _mm512_maskz_loadu_ps(msk, src);
        auto add1_f = _mm512_maskz_loadu_ps(msk, add1);
        auto add2_f = _mm512_maskz_loadu_ps(msk, add2);
        _mm512_mask_storeu_ps(dst, msk, _mm512_add_ps(_mm512_fmadd_ps(a_f, mul_f, add1_f), add2_f));
    }
}
}