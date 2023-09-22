// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

namespace llmdnn {

/// Data type specification
typedef enum {
    /// Undefined data type, used for empty memory descriptors.
    llmdnn_data_type_undef = 0,
    /// 16-bit/half-precision floating point.
    llmdnn_f16 = 1,
    /// non-standard 16-bit (bfloat16 w/ 7 bit mantissa) floating point.
    llmdnn_bf16 = 2,
    /// 32-bit/single-precision floating point.
    llmdnn_f32 = 3,
    /// 32-bit signed integer.
    llmdnn_s32 = 4,
    /// 8-bit signed integer.
    llmdnn_s8 = 5,
    /// 8-bit unsigned integer.
    llmdnn_u8 = 6,
    /// 64-bit/double-precision floating point.
    llmdnn_f64 = 7,

    /// Parameter to allow internal only data_types without undefined behavior.
    /// This parameter is chosen to be valid for so long as sizeof(int) >= 2.
    llmdnn_data_type_max = 0x7fff,
} data_type_t;

typedef enum {
    status_ok,
    status_invalid_arguments,
    status_unimplemented,
    status_fail = 10
} status_t;

}  // namespace llmdnn
