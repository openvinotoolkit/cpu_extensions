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
#include "common/simple_parallel.hpp"

static bool llmdnn_use_numa() {
    if (numa_available() == -1)
        return false;

    static bool init = false;
    static bool use_numa = true;
    if (!init) {
        init = true;
        auto p = std::getenv("LLMDNN_USE_NUMA");
        if (p) {
            use_numa = p[0] != '0';
        }
    }
    return use_numa;
}

void* llmdnn_alloc(size_t aligned_size, size_t size, bool hint_numa) {
    if (hint_numa && llmdnn_use_numa()) {
        int cur_cpu = sched_getcpu();
        auto cur_numa_node = numa_node_of_cpu(cur_cpu);
        return numa_alloc_onnode(size, cur_numa_node);
    } else {
        return aligned_alloc(aligned_size, size);
    }
}

void llmdnn_free(void* p, size_t size, bool hint_numa) {
    if (hint_numa && llmdnn_use_numa()) {
        numa_free(p, size);
    } else {
        ::free(p);
    }
}

int llmdnn_get_numa_id_for_cur_task() {
    if (llmdnn_use_numa()) {
        int cur_cpu = sched_getcpu();
        return numa_node_of_cpu(cur_cpu);
    } else {
        return 0;
    }
}

llm_vector<int> llmdnn_get_numa_nodes() {
    llm_vector<int> numa_nodes;
    if (llmdnn_use_numa()) {
        auto thread_nums = llmdnn::get_total_threads();
        llm_vector<int> numa_nodes_list;
        numa_nodes_list.resize(thread_nums);
        llmdnn::parallel_for(thread_nums, [&] (size_t id) {
            int cur_cpu = sched_getcpu();
            numa_nodes_list[id] = numa_node_of_cpu(cur_cpu);
        });
        for (auto numa_node : numa_nodes_list) {
            if (std::find(numa_nodes.begin(), numa_nodes.end(), numa_node) == numa_nodes.end()) {
                numa_nodes.push_back(numa_node);
            }
        }
        std::sort(numa_nodes.begin(), numa_nodes.end());
    } else {
        numa_nodes.push_back(0);
    }
    return numa_nodes;
}

void* llmdnn_alloc_on(size_t aligned_size, size_t size, int numa_id) {
    if (llmdnn_use_numa()) {
        return numa_alloc_onnode(size, static_cast<size_t>(numa_id));
    } else {
        return aligned_alloc(aligned_size, size);
    }
}

void llmdnn_free_on(void* p, size_t size) {
    if (llmdnn_use_numa()) {
        numa_free(p, size);
    } else {
        ::free(p);
    }
}
