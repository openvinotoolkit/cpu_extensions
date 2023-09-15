// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "llm_types.hpp"
#include "llm_tensor.hpp"

namespace llmdnn {

typedef enum {
    NONE = 0,
    DEQUANT = 1 << 0,
    BIAS = 1 << 1,
    GELU_ERF = 1 << 2,
    GELU_TANH = 1 << 3,
    QUANT = 1 << 4,
    GELU = GELU_ERF,        // default is ERF

    BIAS_GELU = BIAS | GELU,
    DEQUANT_BIAS_GELU = DEQUANT | BIAS_GELU,
    DEQUANT_BIAS_GELU_QUANT = DEQUANT_BIAS_GELU | QUANT,
    DEQUANT_BIAS_QUANT = DEQUANT | BIAS | QUANT,
    DEQUANT_GELU_QUANT = DEQUANT | GELU | QUANT,
    DEQUANT_QUANT = DEQUANT | QUANT,

    DEQUANT_GELU = DEQUANT | GELU,
    DEQUANT_BIAS = DEQUANT | BIAS,

    BIAS_GELU_TANH = BIAS | GELU_TANH,
    DEQUANT_BIAS_GELU_TANH = DEQUANT | BIAS_GELU_TANH,
    DEQUANT_BIAS_GELU_TANH_QUANT = DEQUANT_BIAS_GELU_TANH | QUANT,
    DEQUANT_GELU_TANH_QUANT = DEQUANT | GELU_TANH | QUANT,
    
    DEQUANT_GELU_TANH = DEQUANT | GELU_TANH,
} postops_types;

struct fc_create_param {
    data_type_t dt_a;
    data_type_t dt_b;
    data_type_t dt_c;
    bool b_is_trans;
    postops_types postops_type;
    // for weight compression
    float* scale;
    float* zp;
    int scale_zp_size;
};

struct fc_kernel;

/// Generates a mm kernel based on param
///
/// @param mm Output kernel
/// @param param kernel parameters, supported:
///        fc: (s8,s8,s8),dq,[bias],[gelu],q
///        fc: (s8,s8,bf16),dq,[bias],[gelu]
///        fc: (s8,s8,f32),dq,[bias],[gelu]
///        fc: (bf16,bf16,bf16),[bias],[gelu]
///        fc: (bf16,bf16,f32),[bias],[gelu]
///        fc: (bf16,u8,f32),dq,[bias],[gelu]
///        fc: (bf16,u8,bf16),dq,[bias],[gelu]
///
status_t fc_kernel_create(fc_kernel** mm, const fc_create_param* param);
void fc_kernel_destroy(fc_kernel* mm);
// when fc_create_param.dt_b==bf16, dt_b is in [bf16, f32]
// when fc_create_param.dt_b==u8, dt_b is in [bf16, f32]
void fc_kernel_pack_weight(fc_kernel* mm, void* ptr_b, data_type_t dt_b, size_t N, size_t K, size_t stride_b, size_t n_start, size_t n_end);
void fc_kernel_pack_weight_to_dst(fc_kernel* mm, void* src_b, void* dst_b, data_type_t dt_b, size_t N, size_t K, size_t stride_b, size_t n_start, size_t n_end);
// ptr_b may be null if using fc_kernel_pack_weight to pack into internal buffer
// if ptr_b is not null, its layout is [N/32, 32*rndup(K,32|64)]
void fc_kernel_execute(fc_kernel* mm,
        void* ptr_a, void* ptr_b, void* ptr_c, size_t stride_a, size_t stride_c,
        size_t M, size_t N, size_t K, size_t n_start, size_t n_end,
        float* dq=nullptr, float* q=nullptr, float* bias=nullptr);

/// Generates a fc based on param
class fc {
public:
    fc();
    ~fc();

    bool init(const fc_create_param& param);
    void pack_weight(const tensor& w);
    status_t exec(const tensor& input, const tensor& output, const tensor& dq, const tensor& q, const tensor& bias);

    struct impl {
        virtual ~impl() {}
        virtual bool init(const fc_create_param& param) = 0;
        virtual void pack_weight(const tensor& w) = 0;
        virtual status_t exec(const tensor& input, const tensor& output, const tensor& dq, const tensor& q, const tensor& bias) = 0;
    };
protected:
    impl* _impl;
};

}  // namespace llmdnn
