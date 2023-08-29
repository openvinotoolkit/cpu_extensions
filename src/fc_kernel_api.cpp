// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cstring>
#include <map>

#include "llm_fc.hpp"
#include "fc_kernel_amx.hpp"
#include "mm_kernel_common_amx.hpp"
#include "utility_kernel_avx512.hpp"

namespace llmdnn {

static decltype(&fc_kernel_create) fc_kernel_create_ptr = fc_kernel_create_amx;
static decltype(&fc_kernel_destroy) fc_kernel_destroy_ptr = fc_kernel_destroy_amx;
static decltype(&fc_kernel_pack_weight) fc_kernel_pack_weight_ptr = fc_kernel_pack_weight_amx;
static decltype(&fc_kernel_pack_weight_to_dst) fc_kernel_pack_weight_to_dst_ptr = fc_kernel_pack_weight_to_dst_amx;
static decltype(&fc_kernel_execute) fc_kernel_execute_ptr = fc_kernel_execute_amx;

// interface
status_t fc_kernel_create(fc_kernel** mm, const fc_create_param* param) {
    return fc_kernel_create_ptr(mm, param);
}

void fc_kernel_destroy(fc_kernel* mm) {
    fc_kernel_destroy_ptr(mm);
}

void fc_kernel_pack_weight(fc_kernel* mm, void* ptr_b, data_type_t dt_b, size_t N, size_t K, size_t stride_b, size_t n_start, size_t n_end) {
    fc_kernel_pack_weight_ptr(mm, ptr_b, dt_b, N, K, stride_b, n_start, n_end);
}

void fc_kernel_pack_weight_to_dst(fc_kernel* mm, void* src_b, void* dst_b, data_type_t dt_b, size_t N, size_t K, size_t stride_b, size_t n_start, size_t n_end) {
    fc_kernel_pack_weight_to_dst_ptr(mm, src_b, dst_b, dt_b, N, K, stride_b, n_start, n_end);
}

void fc_kernel_execute(fc_kernel* mm, void* ptr_a, void* ptr_b, void* ptr_c, size_t stride_a, size_t stride_c,
        size_t M, size_t N, size_t K, size_t n_start, size_t n_end, float* dq, float* q, float* bias) {
    fc_kernel_execute_ptr(mm, ptr_a, ptr_b, ptr_c, stride_a, stride_c, M, N, K, n_start, n_end, dq, q, bias);
}

}  // namespace llmdnn
