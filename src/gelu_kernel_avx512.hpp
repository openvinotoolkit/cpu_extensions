// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <stdint.h>
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <immintrin.h>
#endif
#include "common/bf16.hpp"
#include "llm_types.hpp"
#include "common/utility.hpp"
#include "utility_kernel_avx512.hpp"

namespace llmdnn {

    // gelu_erf_minimax_approx_compute_vector_fwd in oneDNN
    //   x*0.5*(1+erf(x/sqrt(2))) = x*0.5*(1 + x*Polynomial(x^2))
    inline __m512 gelu_erf_minmax_approx_avx512(__m512 & x) {
        auto x2 = _mm512_mul_ps(x, x); // x^2
        
        auto x_positive = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(x), _mm512_set1_epi32(0x7FFFFFFF)));    // clear sign mask
        auto x_half = _mm512_mul_ps(x, _mm512_set1_ps(0.5f));

        auto poly = _mm512_castsi512_ps(_mm512_set1_epi32(0x1f1c83fd));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0xa3198977))); // poly * x^2 + xxx
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0x268a7927)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0xa998c963)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0x2c67ddb2)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0xaf013b2c)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0x315d4a4f)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0xb3969b11)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0x35a776e9)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0xb79b0914)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0x3970b255)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0xbb1b7399)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0x3ca3621f)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0xbe082bc7)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0x3f4c4228)));

        // 1.0f + erf(x * inv_sqrt2) = 1.0f + x * P(x^2)
        poly = _mm512_fmadd_ps(poly, x, _mm512_set1_ps(1.0f));
        // x*0.5*(1 + x*Polynomial(x^2))
        poly = _mm512_mul_ps(poly, x_half);

        // combine:
        // zone_id
        //  1 -inf; -saturation_lbound           : 0.0f
        //  2 -saturation_lbound; -linear_ubound : x*0.5*(1 + x*Polynomial(x^2))
        //  3 -linear_ubound, linear_ubound         : x*0.5
        //  4 linear_ubound : saturation_lbound     : x*0.5*(1 + x*Polynomial(x^2))
        //  5 saturation_lbound: +inf               : x
        constexpr int neg_saturation_lbound = 0xc0a00000;
        constexpr int linear_ubound = 0x33800000;
        constexpr int saturation_lbound = 0x40a00000;

        auto mask_x_not_zone1 = _mm512_cmpnlt_ps_mask(x, _mm512_castsi512_ps(_mm512_set1_epi32(neg_saturation_lbound)));
        x = _mm512_maskz_mov_ps(mask_x_not_zone1, x);

        auto mask_x_in_zone5 = _mm512_cmpnle_ps_mask(x_positive, _mm512_castsi512_ps(_mm512_set1_epi32(saturation_lbound)));
        poly = _mm512_mask_mov_ps(poly, mask_x_in_zone5, x);

        auto mask_x_in_zone3 = _mm512_cmple_ps_mask(x_positive, _mm512_castsi512_ps(_mm512_set1_epi32(linear_ubound)));
        poly = _mm512_mask_mov_ps(poly, mask_x_in_zone3, x_half);
        return poly;
    }

    // gelu_tanh_compute_vector_fwd in oneDNN
    inline __m512 gelu_tanh_avx512(__m512& x) {
        // compute G(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x * x)
        auto x2 = _mm512_mul_ps(x, x);
        auto y = _mm512_fmadd_ps(x2, (__m512)_mm512_set1_epi32(0x3d372713), _mm512_set1_ps(1.0f));
        y = _mm512_mul_ps(y, x);
        y = _mm512_mul_ps(y, (__m512)_mm512_set1_epi32(0x3f4c422a));

        // compute tanh(G(x))
        // We split the positive domain in 33 intervals:
        // a) [0; linear_ubound]: in this interval tanh(x) = x
        // b) [linear_ubound; 0x1.8p-12]: This interval spans part of a
        //    half binade
        // c) [0x1.8p-12; 0x1.0p-11], ..., [0x1.8p2; 0x1.0p3]:
        //    one interval for each half binade, there are 29 of those
        // d) [0x1.0p3; saturation_ubound]:
        //    This interval spans part of a half binade
        // e) [0x1.205966p3; saturation_ubound]: in this interval, tanh(x) = 1
        // For b-d, we need 31 polynomials and will do a table lookup for those.
        // To simplify the logic, we will also put a) in the table.

        // The polynomials are of degree 6, so we need to gather 7 coefficients.
        // - sse4.1: we do it the naive way using vextract/vinsert.
        //           Here we will extract the indices in gpr only once and
        //           reuse them as there are only 4 of them.
        // - avx: we do the same as for sse4.1 but use half of the 64-bits
        //           registers to store the idx of second half of YMM and half for
        //           responding XMM. Halfway through the copy we exchange Xmm and
        //           higher half of Ymm and we get the expected result.
        // - avx2: we use vpermps and blend for each coefficient.
        //         This needs an extra vmm to store the mask
        // - avx512: because the table fits in 2 registers, we can use vpermi2d.
        
        // because tanh(x) = -tanh(-x), we extract sign to make x postive
        // and reapply sign at the end
        auto y_positive = _mm512_and_ps(y, (__m512)(_mm512_set1_epi32(0x7fffffff)));

        // We compute the indices for the table lookup
        auto indices = _mm512_sub_epi32((__m512i)y_positive, _mm512_set1_epi32(0x39800000));

        indices = _mm512_and_epi32(indices, _mm512_set1_epi32(0xffc00000));
        indices = _mm512_srli_epi32(indices, 22);
        // we do the argument reduction
        auto y_shift = _mm512_and_ps(y_positive, (__m512)_mm512_set1_epi32(0xffc00000));

        y_shift = _mm512_sub_ps(y_positive, y_shift);

        static uint32_t OV_DECL_ALIGNED(64) tanh_pol_table[] = {
                // coefficients of degree 0
                0x00000000,
                0x39bfffff,
                0x39ffffff,
                0x3a3ffffe,
                0x3a7ffffb,
                0x3abffff7,
                0x3affffeb,
                0x3b3fffdc,
                0x3b7fffab,
                0x3bbfff70,
                0x3bfffeab,
                0x3c3ffdc0,
                0x3c7ffaab,
                0x3cbff701,
                0x3cffeaad,
                0x3d3fdc08,
                0x3d7faacd,
                0x3dbf7081,
                0x3dfeacc9,
                0x3e3dc7fd,
                0x3e7acbf5,
                0x3eb77a9f,
                0x3eec9a9f,
                0x3f22991f,
                0x3f42f7d6,
                0x3f67b7cc,
                0x3f76ca83,
                0x3f7ebbe9,
                0x3f7fd40c,
                0x3f7fff32,
                0x3f7ffffc,
                0x3f800000,
                // coefficients of degree 1
                0x3f800000,
                0x3f800018,
                0x3f7fffe8,
                0x3f7fffda,
                0x3f7fffdc,
                0x3f7fffdc,
                0x3f7fffac,
                0x3f7fff70,
                0x3f7ffeec,
                0x3f7ffdc0,
                0x3f7ffbed,
                0x3f7ff704,
                0x3f7feff5,
                0x3f7fdbca,
                0x3f7fbfff,
                0x3f7f7041,
                0x3f7f009b,
                0x3f7dc36c,
                0x3f7c0aa8,
                0x3f7734b8,
                0x3f70a4de,
                0x3f5f1fd8,
                0x3f495493,
                0x3f18b9ec,
                0x3ed706cb,
                0x3e390b06,
                0x3d90b11f,
                0x3c21a053,
                0x3aaf7fdb,
                0x37ccc1a3,
                0x355c6733,
                0x00000000,
                // coefficients of degree 2
                0x00000000,
                0xbe4e0ff1,
                0x3d25b1b1,
                0x3d6b6dab,
                0x3c9fb1d5,
                0xbabff06f,
                0x3c07b3f6,
                0xbb3fc1bc,
                0x3a9f5921,
                0xbbbf06f2,
                0xbbb0f402,
                0xbc47db9e,
                0xbc73d5e7,
                0xbca25bda,
                0xbcfca780,
                0xbd40e07c,
                0xbd7dab03,
                0xbdbe4a0f,
                0xbdfb14a5,
                0xbe36cc8d,
                0xbe6bd102,
                0xbe9fe7c5,
                0xbeba0f10,
                0xbec206a8,
                0xbea3c388,
                0xbe277d62,
                0xbd8b7960,
                0xbc209f49,
                0xbaad44ca,
                0xb7c6eeac,
                0xb663aa41,
                0x00000000,
                // coefficients of degree 3
                0x00000000,
                0x45b3ae96,
                0xc414eb20,
                0xc450e02e,
                0xc3152b4e,
                0xbead2f56,
                0xc2162e02,
                0xbeb4bd5a,
                0xc11a59a4,
                0xbed2f507,
                0xc020d32c,
                0x3dd0f506,
                0xbf2a75e2,
                0xbff950e3,
                0xbed47334,
                0xbe809b8c,
                0xbeb64532,
                0xbe961a5b,
                0xbe9b63ac,
                0xbea0d4b2,
                0xbe828a77,
                0xbe378612,
                0xbdc20908,
                0x3d2d3957,
                0x3dd46e89,
                0x3db3f629,
                0x3d2c5e7b,
                0x3bd20403,
                0x3a59dfae,
                0x3770af45,
                0x372cc014,
                0x00000000,
                // coefficients of degree 4
                0x00000000,
                0xcc981a1b,
                0x4a7edd3d,
                0x4ab1007c,
                0x48fedd9c,
                0x41a557b5,
                0x477ee32a,
                0x422557f5,
                0x45ff3ce4,
                0x42a55641,
                0x446e0867,
                0xc33dc19a,
                0x42915214,
                0x43af4fad,
                0x4110fe88,
                0xc1099b75,
                0x3fc8a8dc,
                0xbfbeaef5,
                0xbe365aad,
                0x3f4d9652,
                0x3ddfa08f,
                0x3e34e9b8,
                0x3e2d07a6,
                0x3dc63567,
                0x3cdaeb78,
                0xbcd17537,
                0xbc92829c,
                0xbb43ab99,
                0xb9b471dd,
                0xb6baad5a,
                0xb78bafc7,
                0x00000000,
                // coefficients of degree 5
                0x00000000,
                0x52f688d5,
                0xd0505c72,
                0xd08f98e3,
                0xce505cc9,
                0xc7162b8a,
                0xcc5061d6,
                0xc7162bdf,
                0xca50b37f,
                0xc7162a3a,
                0xc8422086,
                0x471a714e,
                0xc5ece1f1,
                0xc70e3d90,
                0xc3eba94a,
                0x43e0c424,
                0xc21f4552,
                0x42217cc8,
                0x405e7dc4,
                0xc10dd401,
                0x3e96b602,
                0xbd1a6d2f,
                0xbd393883,
                0xbd674682,
                0xbd310016,
                0xb961e269,
                0x3ba32495,
                0x3a7680d5,
                0x38b3173c,
                0x35a9deea,
                0x375c3f2a,
                0x00000000,
                // coefficients of degree 6
                0x00000000,
                0xd8995ed1,
                0x558285ea,
                0x55b2cd69,
                0x53028625,
                0x4bc9991f,
                0x5082898a,
                0x4b4999b3,
                0x4e02c07c,
                0x4ac99764,
                0x4b72c822,
                0xca40c0e1,
                0x489413e4,
                0x49b12224,
                0x46134c4e,
                0xc60c2d57,
                0x43c83910,
                0xc3c872d1,
                0xc186bc9e,
                0x42325bc3,
                0xbf2ffa4a,
                0x3d9a203c,
                0xbc545a43,
                0xbae08fee,
                0x3c80225d,
                0x3b1fd1df,
                0xba36b9d1,
                0xb91de544,
                0xb71f100f,
                0xb408e2ed,
                0xb685fec8,
                0x00000000,
        };
        auto pol = _mm512_permutex2var_ps(_mm512_load_ps(tanh_pol_table + 32 * 6), indices, _mm512_load_ps(tanh_pol_table + 32 * 6 + 16));

        pol = _mm512_fmadd_ps(pol, y_shift, _mm512_permutex2var_ps(_mm512_load_ps(tanh_pol_table + 32 * 5), indices, _mm512_load_ps(tanh_pol_table + 32 * 5 + 16)));
        pol = _mm512_fmadd_ps(pol, y_shift, _mm512_permutex2var_ps(_mm512_load_ps(tanh_pol_table + 32 * 4), indices, _mm512_load_ps(tanh_pol_table + 32 * 4 + 16)));
        pol = _mm512_fmadd_ps(pol, y_shift, _mm512_permutex2var_ps(_mm512_load_ps(tanh_pol_table + 32 * 3), indices, _mm512_load_ps(tanh_pol_table + 32 * 3 + 16)));
        pol = _mm512_fmadd_ps(pol, y_shift, _mm512_permutex2var_ps(_mm512_load_ps(tanh_pol_table + 32 * 2), indices, _mm512_load_ps(tanh_pol_table + 32 * 2 + 16)));
        pol = _mm512_fmadd_ps(pol, y_shift, _mm512_permutex2var_ps(_mm512_load_ps(tanh_pol_table + 32 * 1), indices, _mm512_load_ps(tanh_pol_table + 32 * 1 + 16)));
        pol = _mm512_fmadd_ps(pol, y_shift, _mm512_permutex2var_ps(_mm512_load_ps(tanh_pol_table + 32 * 0), indices, _mm512_load_ps(tanh_pol_table + 32 * 0 + 16)));

        // we restore src with cleared sign, and keep sign
        auto sign = _mm512_and_ps(y, (__m512)_mm512_set1_epi32(0x80000000));

        // Now we blend the results
        // [saturation_ubound; +inf[ : we return +/- 1
        auto dst = (__m512)_mm512_set1_epi32(0x3f800000);
        // [linear_ubound; saturation_lbound] : we return +/- P(x)
        auto mask = (__m512)_mm512_set1_epi32(0x41102cb3);
        auto mask16 = _mm512_cmp_ps_mask(mask, y_positive, _CMP_GT_OS);
        dst = _mm512_mask_blend_ps(mask16, dst, pol);
        // [0; linear_ubound]  : we return x
        mask = (__m512)_mm512_set1_epi32(0x39ddb3d7);
        mask16 = _mm512_cmp_ps_mask(mask, y_positive, _CMP_GT_OS);
        dst = _mm512_mask_blend_ps(mask16, dst, y_positive);

        // We reapply the sign and return
        dst = _mm512_xor_ps(dst, sign);

        // compute 0.5 * x * (1 + tanh(G(x)))
        dst = _mm512_add_ps(dst, (__m512)_mm512_set1_epi32(0x3f800000));
        dst = _mm512_mul_ps(dst, (__m512)_mm512_set1_epi32(0x3f000000));
        dst = _mm512_mul_ps(dst, x);
        return dst;
    }
}