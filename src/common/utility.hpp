// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <initializer_list>
#include <vector>
#include <string>
#include "llm_types.hpp"

#ifndef OV_DECL_ALIGNED
#  ifdef __GNUC__
#    define OV_DECL_ALIGNED(x) __attribute__ ((aligned (x)))
#  elif defined _MSC_VER
#    define OV_DECL_ALIGNED(x) __declspec(align(x))
#  else
#    define OV_DECL_ALIGNED(x)
#  endif
#endif // OV_DECL_ALIGNED

namespace llmdnn {

inline size_t get_precision_size(data_type_t type) {
    switch(type) {
        case llmdnn_f16:
        case llmdnn_bf16:
            return 2;
        case llmdnn_f32:
        case llmdnn_s32:
            return 4;
        case llmdnn_s8:
        case llmdnn_u8:
            return 1;
        case llmdnn_f64:
            return 8;
        default:
            assert(false && "unknown data type");
            return 0;
    }
}

inline data_type_t get_dt_from_str(const std::string& name) {
    static std::pair<const char*, data_type_t> name2type[] = {
        { "f16", llmdnn_f16 },
        { "bf16", llmdnn_bf16 },
        { "f32", llmdnn_f32 },
        { "s32", llmdnn_s32 },
        { "i32", llmdnn_s32 },
        { "s8", llmdnn_s8 },
        { "i8", llmdnn_s8 },
        { "u8", llmdnn_u8 },
        { "f64", llmdnn_f64 },
    };
    for (size_t i = 0; i < sizeof(name2type) / sizeof(name2type[0]); i++) {
        if (name == name2type[i].first)
            return name2type[i].second;
    }

    return llmdnn_data_type_undef;
}

}  // namespace llmdnn
