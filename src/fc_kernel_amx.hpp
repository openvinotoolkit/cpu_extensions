// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_fc.hpp"

namespace llmdnn {

status_t fc_kernel_create_amx(fc_kernel** mm, const fc_create_param* param);

void fc_kernel_destroy_amx(fc_kernel* mm);

void fc_kernel_pack_weight_amx(fc_kernel* mm, void* ptr_b, size_t N, size_t K, size_t stride_b, size_t n_start, size_t n_end);

void fc_kernel_execute_amx(fc_kernel* mm, void* ptr_a, void* ptr_c, size_t stride_a, size_t stride_c,
        size_t M, size_t N, size_t K, size_t n_start, size_t n_end, float* dq, float* q, float* bias);

void fc_kernel_bf16w8_get_q_dq_amx(size_t K, size_t N, size_t stride, void* ptr, float* q, float* dq);

}  // namespace llmdnn
