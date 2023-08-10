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
#include "llm_fc.hpp"
#include "common/tensor2d.hpp"
#include "common/tensor2d_helper.hpp"
#include "test_common.hpp"
#include "timeit.hpp"

// timeit timer({{PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, "task_clock"}});
timeit timer;

using namespace std;
using namespace llmdnn;

inline int omp_thread_count()
{
    int n = 0;
#pragma omp parallel reduction(+ : n)
    n += 1;
    return n;
}
static int OMP_NT = omp_thread_count();

//=================================================== parallel_nt
template <typename F>
void parallel_nt(const F &func)
{
#pragma omp parallel for
    for (int idx = 0; idx < OMP_NT; idx++)
    {
        func(OMP_NT, idx);
    }
}

template <typename T, typename U>
inline typename remove_reference<T>::type div_up(const T a, const U b)
{
    assert(b);
    return static_cast<typename remove_reference<T>::type>((a + b - 1) / b);
}

template <typename T, typename U>
inline void balance211(T n, U team, U tid, T &n_start, T &n_end)
{
    T n_min = 1;
    T &n_my = n_end;
    if (team <= 1 || n == 0)
    {
        n_start = 0;
        n_my = n;
    }
    else if (n_min == 1)
    {
        // team = T1 + T2
        // n = T1*n1 + T2*n2  (n1 - n2 = 1)
        T n1 = div_up(n, (T)team);
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_my = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}

struct Fullyconnect
{
    std::vector<std::shared_ptr<llmdnn::fc_kernel>> fcLLMs;

    bool get_range(size_t nthreads, size_t tid, size_t N, size_t &n0, size_t &n1)
    {
        auto nblocks = (N + 31) / 32;
        size_t start{0}, end{0};
        balance211(nblocks, nthreads, tid, start, end);
        n0 = start * 32;
        n1 = std::min(end * 32, N);
        return n0 < N;
    }

    bool prepare(fc_create_param &param, void *ptr_B, size_t ldb, size_t M, size_t N, size_t K)
    {
        fcLLMs.resize(OMP_NT);
        for (size_t i = 0; i < OMP_NT; i++)
        {
            llmdnn::fc_kernel *fc;
            if (fc_kernel_create(&fc, &param) != llmdnn::status_ok)
                return false;
            fcLLMs[i] = std::shared_ptr<llmdnn::fc_kernel>(fc, [](llmdnn::fc_kernel *p)
                                                           { fc_kernel_destroy(p); });
        }
        parallel_nt([&](size_t nt, size_t idx)
                    {
                        size_t n0, n1;
                        if (get_range(nt, idx, N, n0, n1))
                            fc_kernel_pack_weight(fcLLMs[idx].get(), ptr_B, N, K, ldb, n0, n1);
                        // fc_kernel_pack_weight(fcLLMs[idx].get(), ptr_B, N, K, ldb, 0, N);
                    });
        return true;
    }

    void execute(size_t M, size_t N, size_t K, void *pA, size_t ldA, void *pC, size_t ldC, float *dq, float *q, float *bias)
    {
        parallel_nt([&](size_t nt, size_t idx)
                    {
            size_t n0 = 0, n1 = N;
            size_t m0 = 0, m1 = M;
            if (get_range(nt, idx, N, n0, n1))
            {
                // get_range(OMP_NT, idx, M, m0, m1);
                fc_kernel_execute(fcLLMs[idx].get(), pA + m0 * ldA, pC + m0 * ldC, ldA, ldC, m1 - m0, N, K, n0, n1, dq, q, bias);
            } });
    }
};

template <typename TA, typename TB, typename TC>
void do_test(fc_create_param &param, size_t M, size_t N, size_t K)
{
    Fullyconnect fc;

    tensor2D<TA> A(M, K, true);
    tensor2D<TB> B(K, N, true);
    tensor2D<TC> C(M, N, true);
    tensor2D<TC> C_Ref(M, N, true);
    tensor2D<float> dq(1, N);
    tensor2D<float> q(1, N);
    tensor2D<float> bias(1, N);

    fill_rnd(A);
    fill_rnd(B);
    dq = 2;
    q = 2;
    fill_rnd(bias);
    bias = 1;

    tensor2D<TB> BT = B.Tr();
    TB *ptr_B;
    size_t ldb;
    if (param.b_is_trans)
    {
        ptr_B = BT.data;
        ldb = BT.stride;
    }
    else
    {
        ptr_B = B.data;
        ldb = B.stride;
    }

    assert(fc.prepare(param, ptr_B, ldb, M, N, K));

    timer.tag(__func__, M, N, K, TypeName<TA>::get())(-1000, [&]()
                                                      { fc.execute(M, N, K, A.data, A.stride, C.data, C.stride, dq.data, q.data, bias.data); });
    C_Ref = 0;
    float *ptr_dq = nullptr;
    float *ptr_q = nullptr;
    float *ptr_bias = nullptr;
    func_act act = func_act();
    if (param.postops_type & DEQUANT)
        ptr_dq = dq.data;
    if (param.postops_type & QUANT)
        ptr_q = q.data;
    if (param.postops_type & BIAS)
        ptr_bias = bias.data;
    if (param.postops_type & GELU)
    {
        act = [](float x)
        {
            return x * 0.5 * (1 + std::erf(x / std::sqrt(2)));
        };
    }
    if (param.postops_type & GELU_TANH)
    {
        act = [](float x)
        {
            return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / 3.1415926f) * x * (1 + 0.044715f * x * x)));
        };
    }

    matmul(A, B, C_Ref, ptr_dq, ptr_bias, act, ptr_q);
    float thresh = 0.0001f;
    if (std::is_same<TA, int8_t>::value || std::is_same<TA, uint8_t>::value)
        thresh = 1.1f;
    if (std::is_same<TA, ov::bfloat16>::value)
        thresh = 0.01f;
    assert(compare(C, C_Ref, thresh));
}

int main()
{
    initXTILE();
    std::cout << "OMP_NT=" << OMP_NT << std::endl;
    fc_create_param param;
    param.dt_a = llmdnn_bf16;
    param.dt_b = llmdnn_bf16;
    param.dt_c = llmdnn_bf16;
    param.b_is_trans = false;
    param.postops_type = NONE;

    do_test<ov::bfloat16, ov::bfloat16, ov::bfloat16>(param, 640, 2 * 1024, 640);
    do_test<ov::bfloat16, ov::bfloat16, ov::bfloat16>(param, 2 * 1024, 640, 640);
}
