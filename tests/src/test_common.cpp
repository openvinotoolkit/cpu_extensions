// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>
#include <chrono>
#include <iostream>
#include <omp.h>
#include "common/simple_parallel.hpp"
#include "test_common.hpp"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE         /* See feature_test_macros(7) */
#endif
#include <unistd.h>
#include <sys/syscall.h>   /* For SYS_xxx definitions */

#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

using namespace std;
using namespace llmdnn;

std::string dtype_to_str(data_type_t type) {
    switch (type) {
        case llmdnn_data_type_undef: return "undef";
        case llmdnn_f16: return "f16";
        case llmdnn_bf16: return "bf16";
        case llmdnn_f32: return "f32";
        case llmdnn_s32: return "s32";
        case llmdnn_s8: return "s8";
        case llmdnn_u8: return "u8";
        case llmdnn_f64: return "f64";
        default: return "unkown";
    }
}

bool initXTILE() {
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status) return false;
    if (bitmask & XFEATURE_MASK_XTILEDATA) return true;

    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (0 != status)
        return false; // XFEATURE_XTILEDATA setup is failed, TMUL usage is not allowed
    status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);

    // XFEATURE_XTILEDATA setup is failed, can't use TMUL
    if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA)) return false;

    // XFEATURE_XTILEDATA set successfully, TMUL usage is allowed
    return true;
}

namespace llmdnn {

size_t get_total_threads() {
    return omp_get_max_threads();
}

void simple_parallel_for(const size_t total, const std::function<void(size_t)>& fn) {
    #pragma omp parallel for
    for(size_t i = 0; i < total; i++) {
        fn(i);
    }
}

}