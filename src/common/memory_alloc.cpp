// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sched.h>
#include <numa.h>
#include "memory_alloc.hpp"

void* llmdnn_alloc(size_t aligned_size, size_t size, bool hint_numa) {
    if (hint_numa && numa_available() != -1) {
        int cur_cpu = sched_getcpu();
        auto cur_numa_node = numa_node_of_cpu(cur_cpu);
        return numa_alloc_onnode(size, cur_numa_node);
    } else {
        return aligned_alloc(aligned_size, size);
    }
}

void llmdnn_free(void* p, size_t size, bool hint_numa) {
    if (hint_numa && numa_available() != -1) {
        numa_free(p, size);
    } else {
        ::free(p);
    }
}
