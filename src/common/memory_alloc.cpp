// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sched.h>
#include <dlfcn.h>

#include "memory_alloc.hpp"
#include "common/simple_parallel.hpp"

struct numa_funcs {
    numa_funcs() {
        _numa_handle = dlopen(libnuma_path, RTLD_NOW);
        if (_numa_handle) {
            _numa_available = reinterpret_cast<decltype(_numa_available)>(dlsym(_numa_handle, "numa_available"));
            _numa_node_of_cpu = reinterpret_cast<decltype(_numa_node_of_cpu)>(dlsym(_numa_handle, "numa_node_of_cpu"));
            _numa_alloc_onnode = reinterpret_cast<decltype(_numa_alloc_onnode)>(dlsym(_numa_handle, "numa_alloc_onnode"));
            _numa_free = reinterpret_cast<decltype(_numa_free)>(dlsym(_numa_handle, "numa_free"));
        }
    }

    ~numa_funcs() {
        if (_numa_handle) {
            dlclose(_numa_handle);
        }
    }

    static numa_funcs& get() {
        static numa_funcs funcs;
        return funcs;
    }

    int numa_available() {
        if (_numa_available) {
            return _numa_available();
        } else {
            return -1;
        }
    }

    int numa_node_of_cpu(int cpu) {
        if (_numa_node_of_cpu) {
            return _numa_node_of_cpu(cpu);
        } else {
            return 0;
        }
    }

    void *numa_alloc_onnode(size_t size, int node) {
        if (_numa_alloc_onnode) {
            return _numa_alloc_onnode(size, node);
        } else {
            return aligned_alloc(64, size);
        }
    }

    void numa_free(void *mem, size_t size) {
        if (_numa_free) {
            _numa_free(mem, size);
        } else {
            ::free(mem);
        }
    }

private:
    constexpr static const char* libnuma_path = "libnuma.so.1";
    void* _numa_handle = nullptr;
    int (*_numa_available)(void) = nullptr;
    int (*_numa_node_of_cpu)(int cpu) = nullptr;
    void *(*_numa_alloc_onnode)(size_t size, int node) = nullptr;
    void (*_numa_free)(void *mem, size_t size) = nullptr;
};

static bool llmdnn_use_numa() {
    struct init_numa_flag {
        init_numa_flag() {
            auto p = std::getenv("LLMDNN_USE_NUMA");
            if (p) {
                use_numa = p[0] != '0';
            }
            if (use_numa) {
                use_numa = numa_funcs::get().numa_available() != -1;
            }
        }
    
        bool use_numa = true;
    };

    static init_numa_flag flag;

    return flag.use_numa;
}

void* llmdnn_alloc(size_t aligned_size, size_t size, bool hint_numa) {
    if (hint_numa && llmdnn_use_numa()) {
        int cur_cpu = sched_getcpu();
        auto cur_numa_node = numa_funcs::get().numa_node_of_cpu(cur_cpu);
        return numa_funcs::get().numa_alloc_onnode(size, cur_numa_node);
    } else {
        return aligned_alloc(aligned_size, size);
    }
}

void llmdnn_free(void* p, size_t size, bool hint_numa) {
    if (hint_numa && llmdnn_use_numa()) {
        numa_funcs::get().numa_free(p, size);
    } else {
        ::free(p);
    }
}

int llmdnn_get_numa_id_for_cur_task() {
    if (llmdnn_use_numa()) {
        int cur_cpu = sched_getcpu();
        return numa_funcs::get().numa_node_of_cpu(cur_cpu);
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
            numa_nodes_list[id] = numa_funcs::get().numa_node_of_cpu(cur_cpu);
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
        return numa_funcs::get().numa_alloc_onnode(size, static_cast<size_t>(numa_id));
    } else {
        return aligned_alloc(aligned_size, size);
    }
}

void llmdnn_free_on(void* p, size_t size) {
    if (llmdnn_use_numa()) {
        numa_funcs::get().numa_free(p, size);
    } else {
        ::free(p);
    }
}
