// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

#include "common/log.hpp"
#include "common/simple_parallel.hpp"
#include "common/tensor2d.hpp"
#include "common/utility.hpp"
#include "common/compatible.hpp"
#include "common/memory_alloc.hpp"
#include "llm_types.hpp"
#include "utility_kernel_avx512.hpp"
#include "mm_kernel_common_amx.hpp"
#include "softmax_kernel_avx512.hpp"
#include "transpose_kernel_avx512.hpp"
#include "llm_fc.hpp"
#include "fc_amx.hpp"

namespace llmdnn {

struct fc_impl_amx : public fc::impl {
    fc_impl_amx() = default;
    ~fc_impl_amx();

    bool init(const fc_create_param& param) override;
    void pack_weight(const tensor& w) override;
    status_t exec(const tensor& input, const tensor& output, const tensor& dq, const tensor& q, const tensor& bias) override;
    void build_thread_infos(const llm_vector<int>& numa_nodes);

    fc_create_param _create_param;
    llm_vector<fc_kernel*> _kernel;      // one kernel for each numa node
    llm_vector<int8_t*> _weights;        // one weight for each numa node
    llm_vector<size_t> _weight_sizes;    // one weight size for each numa node
    size_t _thread_nums;                 // thread numbers
    size_t _N_in_one_numa;              // N on each numa node
    llm_vector<int> _thread_nums_in_one_numa;     // thread numbers in one numa node
    size_t _K_align;
    struct work_info {
        int numa_id;                     // numa node id, use to index in _numa_nodes
        size_t thread_no_in_one_numa;    // sequence no in one numa node
    };
    llm_vector<work_info> _work_infos;   // map thread id to numa node id and thread no in one numa node
};

fc_impl_amx::~fc_impl_amx() {
    for (size_t i = 0; i < _kernel.size(); i++) {
        fc_kernel_destroy(_kernel[i]);
    }
    for (size_t i = 0; i < _weight_sizes.size(); i++) {
        llmdnn_free_on(_weights[i], _weight_sizes[i]);
    }
}

bool fc_impl_amx::init(const fc_create_param& param) {
    _create_param = param;
    _thread_nums = get_total_threads();
    _kernel.resize(_thread_nums);
    std::atomic_bool ret{true};
    parallel_for(_thread_nums, [&] (size_t id) {
        if (fc_kernel_create(&_kernel[id], &param) != llmdnn::status_ok) {
            ret = false;
        }
    });

    return ret;
}

void fc_impl_amx::build_thread_infos(const llm_vector<int>& numa_nodes) {
    _work_infos.resize(_thread_nums);
    struct int_atomic {
        std::atomic_int v{0};
    };
    llm_vector<int_atomic> thread_id_in_one_numa(numa_nodes.size());
    // the real numa id may not be continuous, but we need a number to index _numa_nodes
    parallel_for(_thread_nums, [&] (size_t id) {
        auto cur_numa_id = llmdnn_get_numa_id_for_cur_task();
        for (int i = 0; i < static_cast<int>(numa_nodes.size()); i++) {
            if (numa_nodes[i] == cur_numa_id) {
                _work_infos[id].numa_id = i;
                _work_infos[id].thread_no_in_one_numa = thread_id_in_one_numa[i].v.fetch_add(1);
                break;
            }
        }
    });

    // check: the index is stable in another loop
    std::mutex m;
    parallel_for(_thread_nums, [&] (size_t id) {
        auto cur_numa_id = llmdnn_get_numa_id_for_cur_task();
        for (int i = 0; i < static_cast<int>(numa_nodes.size()); i++) {
            if (numa_nodes[i] == cur_numa_id) {
                if (_work_infos[id].numa_id != i) {
                    std::lock_guard<std::mutex> l(m);
                    DEBUG_LOG << "index test fail: cur numa index of thread no " << id << " is " << i << ", prev index " << _work_infos[id].numa_id << "\n";
                }
                break;
            }
        }
    });

    // check: each numa should have same thread numbers
    _thread_nums_in_one_numa.resize(numa_nodes.size());
    int actual_threads = thread_id_in_one_numa[0].v;
    _thread_nums_in_one_numa[0] = thread_id_in_one_numa[0].v;
    for (size_t i = 1; i < thread_id_in_one_numa.size(); i++) {
        if (thread_id_in_one_numa[0].v != thread_id_in_one_numa[i].v) {
            DEBUG_LOG << "numa test fail: thread number of numa " << i << " is " << thread_id_in_one_numa[i].v << ", not equal to numa 0 thread numbers: " << thread_id_in_one_numa[0].v << "\n";
        }
        actual_threads += thread_id_in_one_numa[i].v;
        _thread_nums_in_one_numa[i] = thread_id_in_one_numa[i].v;
    }
    // check: actual threads number should equal to _thread_nums
    if (static_cast<int>(_thread_nums) != actual_threads) {
        DEBUG_LOG << "thread number test fail: actual threads number: " << actual_threads << ", not equal to _thread_nums " << _thread_nums << "\n";
    }
}

void fc_impl_amx::pack_weight(const tensor& w) {
    auto N = w.m_dims[_create_param.b_is_trans ? 0 : 1];
    auto K = w.m_dims[_create_param.b_is_trans ? 1 : 0];
    // will allocate memory on different numa nodes:
    //   1, get numa nodes number, allocate memory on each numa node
    //   2, get cores number, compute each cores area and pack each area simultaneously
    // for omp, GOMP_CPU_AFFINITY=0-95 numactl -C0-95 will bind threads to cores
    auto numa_nodes = llmdnn_get_numa_nodes();
    auto numa_nodes_nums = numa_nodes.size();
    auto N_blocks = rndup(N, 32) / 32;
    // NOTE: assuming memory/thread is evenly distributed across mutiple numas. Need to support unbalanced numa?
    _N_in_one_numa = (N_blocks + numa_nodes_nums - 1) / numa_nodes_nums * 32;
    if (_create_param.dt_b == data_type_t::llmdnn_bf16) {
        _K_align = rndup(K, 32);
    } else {
        _K_align = rndup(K, 64);
    }
    _weights.resize(numa_nodes_nums);
    _weight_sizes.resize(numa_nodes_nums);
    // allocate memory
    for (size_t i = 0; i < numa_nodes_nums; i++) {
        auto size = _K_align * _N_in_one_numa * get_precision_size(_create_param.dt_b);
        _weights[i] = reinterpret_cast<int8_t*>(llmdnn_alloc_on(64, size + 4096, numa_nodes[i]));
        _weight_sizes[i] = size + 4096;
        memset(_weights[i] + size, 0, 4096);
    }
    build_thread_infos(numa_nodes);
    auto work_amount_in_one_numa = _N_in_one_numa / 32;
    parallel_for(_thread_nums, [&] (size_t id) {
        auto numa_id = _work_infos[id].numa_id;
        auto thread_no_in_one_numa = _work_infos[id].thread_no_in_one_numa;
        size_t start, end;
        splitter(work_amount_in_one_numa, static_cast<size_t>(_thread_nums_in_one_numa[numa_id]), thread_no_in_one_numa, start, end);
        size_t n0_in_one_numa = start * 32;
        size_t n1_in_one_numa = std::min(end * 32, _N_in_one_numa);
        if (n0_in_one_numa >= _N_in_one_numa) return;
        auto n0 = n0_in_one_numa + _N_in_one_numa * numa_id;
        auto n1 = n1_in_one_numa + _N_in_one_numa * numa_id;
        n1 = std::min(n1, N);
        if (n0 >= n1) return;

        auto dst = _weights[numa_id] + n0_in_one_numa * _K_align * get_precision_size(_create_param.dt_b);
        fc_kernel_pack_weight_to_dst(_kernel[id], w.data<int8_t>(), dst, w.m_dtype, N, K, w.stride(0), n0, n1);
    });
}

status_t fc_impl_amx::exec(const tensor& input, const tensor& output, const tensor& dq, const tensor& q, const tensor& bias) {
    if (input.m_rank != 2 || output.m_rank != 2 || bias.m_rank != 2) {
        DEBUG_LOG << "input,output,bias rank should be 2.\n";
        return status_t::status_invalid_arguments;
    }

    auto M = input.size(0);
    auto N = output.size(1);
    auto K = input.size(1);
    auto work_amount_in_one_numa = _N_in_one_numa / 32;
    parallel_for(_thread_nums, [&](size_t id) {
        auto numa_id = _work_infos[id].numa_id;
        auto thread_no_in_one_numa = _work_infos[id].thread_no_in_one_numa;
        size_t start, end;
        splitter(work_amount_in_one_numa, static_cast<size_t>(_thread_nums_in_one_numa[numa_id]), thread_no_in_one_numa, start, end);
        size_t n0_in_one_numa = start * 32;
        size_t n1_in_one_numa = std::min(end * 32, _N_in_one_numa);
        if (n0_in_one_numa >= _N_in_one_numa) return;
        auto n0 = n0_in_one_numa + _N_in_one_numa * numa_id;
        auto n1 = n1_in_one_numa + _N_in_one_numa * numa_id;
        n1 = std::min(n1, N);
        if (n0 >= n1) return;

        auto weight = _weights[numa_id] + n0_in_one_numa * _K_align * get_precision_size(_create_param.dt_b);
        fc_kernel_execute(_kernel[id], input.data<int8_t>(), weight, output.data<int8_t>(), input.stride(0),
            output.stride(0), M, N, K, n0, n1, dq.data<float>(), q.data<float>(), bias.data<float>());
    });

    return status_t::status_ok;
}

fc::impl* new_fc_impl_amx() {
    return new fc_impl_amx();
}

}  // namespace llmdnn
