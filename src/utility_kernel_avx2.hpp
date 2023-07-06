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

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cmath>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <immintrin.h>
#endif

inline __m256i get_mask(int N7) {
	static __m256i mask[] = {
		_mm256_set_epi32( 0, 0, 0, 0, 0, 0, 0, 0),
		_mm256_set_epi32( 0, 0, 0, 0, 0, 0, 0,-1),
		_mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1),
		_mm256_set_epi32( 0, 0, 0, 0, 0,-1,-1,-1),
		_mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1),
		_mm256_set_epi32( 0, 0, 0,-1,-1,-1,-1,-1),
		_mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1),
		_mm256_set_epi32( 0,-1,-1,-1,-1,-1,-1,-1),
		_mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1),
	};
	return _mm256_loadu_si256(&mask[N7]);
}

// https://stackoverflow.com/questions/23189488/horizontal-sum-of-32-bit-floats-in-256-bit-avx-vector
static inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}

}