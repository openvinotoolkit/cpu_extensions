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
#include "gtest/gtest.h"
#include "llm_fc.hpp"
#include "common/tensor2d.hpp"
#include "common/tensor2d_helper.hpp"
#include "test_common.hpp"

using namespace std;
using namespace llmdnn;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

using FCKernelTestShape = std::tuple<size_t, size_t, size_t>;
using FCKernelTestDTPost = std::tuple<data_type_t, data_type_t, data_type_t, postops_types>;
using FCKernelTestParamSet = std::tuple<
        FCKernelTestDTPost,                          // a, b, c data type, postops
        bool,                                        // b needs transpose
        FCKernelTestShape                            // M, N, K
        >;

class FCKernelTest : public TestWithParam<FCKernelTestParamSet> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FCKernelTestParamSet>& obj) {
        FCKernelTestDTPost types;
        bool is_transpose;
        postops_types postops_type;
        data_type_t dt_a, dt_b, dt_c;
        FCKernelTestShape shape;
        int M, N, K;
        std::tie(types, is_transpose, shape) = obj.param;
        std::tie(M, N, K) = shape;
        std::tie(dt_a, dt_b, dt_c, postops_type) = types;

        std::ostringstream result;
        result << "A_" << dtype_to_str(dt_a) << "_B_" << dtype_to_str(dt_b)
               << "_C_" << dtype_to_str(dt_c) << (is_transpose ? "_transpose" : "")
               << "_postops_" << postops_type << "_M_" << M << "_N_" << N << "_K_" << K;
        return result.str();
    }

protected:
    virtual void SetUp() override {
        initXTILE();

        FCKernelTestShape shape;
        FCKernelTestDTPost types;
        std::tie(types, _is_transpose, shape) = GetParam();
        std::tie(_M, _N, _K) = shape;
        std::tie(_dt_a, _dt_b, _dt_c, _postops_type) = types;
    };

    template<typename TA, typename TB, typename TC>
    void do_test() {
        fc_kernel* fc;
        fc_create_param param = {
            _dt_a, _dt_b, _dt_c,
            _is_transpose, _postops_type
        };
        ASSERT_TRUE(fc_kernel_create(&fc, &param) == llmdnn::status_ok);
        auto gemm = std::shared_ptr<fc_kernel>(fc, [](fc_kernel* p) { fc_kernel_destroy(p); });

        tensor2D<TA> A(_M, _K, true);
        tensor2D<TB> B(_K, _N, true);
        tensor2D<TC> C(_M, _N, true);
        tensor2D<TC> C_Ref(_M, _N, true);
        tensor2D<float> dq(1, _N);
        tensor2D<float> q(1, _N);
        tensor2D<float> bias(1, _N);

        fill_rnd(A);
        fill_rnd(B);
        dq = 2;
        q = 2;
        fill_rnd(bias);
        bias = 1;

        tensor2D<TB> BT = B.Tr();
        TB* ptr_B;
        size_t ldb;
        if (_is_transpose) {
            ptr_B = BT.data;
            ldb = BT.stride;
        } else {
            ptr_B = B.data;
            ldb = B.stride;
        }
        fc_kernel_pack_weight(gemm.get(), ptr_B, _N, _K, ldb, 0, _N);
        fc_kernel_execute(gemm.get(), A.data, C.data, A.stride,
            C.stride, _M, _N, _K, 0, _N, dq.data, q.data, bias.data);
        C_Ref = 0;
        float* ptr_dq = nullptr;
        float* ptr_q = nullptr;
        float* ptr_bias = nullptr;
        func_act act = func_act(); 
        if (_postops_type & DEQUANT) {
            ptr_dq = dq.data;
        }
        if (_postops_type & QUANT) {
            ptr_q = q.data;
        }
        if (_postops_type & BIAS) {
            ptr_bias = bias.data;
        }
        if (_postops_type & GELU) {
            act = [] (float x) {
                return x * 0.5 * (1 + std::erf(x / std::sqrt(2)));
            };
        }
        if (_postops_type & GELU_TANH) {
            act = [] (float x) {
                return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / 3.1415926f) * x * (1 + 0.044715f * x * x)));
            };
        }

        matmul(A, B, C_Ref, ptr_dq, ptr_bias, act, ptr_q);
        float thresh = 0.0001f;
        if (std::is_same<TA, int8_t>::value || std::is_same<TA, uint8_t>::value)
            thresh = 1.1f;
        if (std::is_same<TA, ov::bfloat16>::value)
            thresh = 0.01f;
        ASSERT_TRUE(compare(C, C_Ref, thresh));
    }

    int _M, _N, _K;
    bool _is_transpose;
    postops_types _postops_type;
    data_type_t _dt_a, _dt_b, _dt_c;
};

TEST_P(FCKernelTest, Func) {
    if (_dt_a == llmdnn_s8 && _dt_b == llmdnn_s8 && _dt_c == llmdnn_s8) {
        do_test<int8_t, int8_t, int8_t>();
    } else if (_dt_a == llmdnn_s8 && _dt_b == llmdnn_s8 && _dt_c == llmdnn_bf16) {
        do_test<int8_t, int8_t, ov::bfloat16>();
    } else if (_dt_a == llmdnn_s8 && _dt_b == llmdnn_s8 && _dt_c == llmdnn_f32) {
        do_test<int8_t, int8_t, float>();
    } else if (_dt_a == llmdnn_bf16 && _dt_b == llmdnn_bf16 && _dt_c == llmdnn_bf16) {
        do_test<ov::bfloat16, ov::bfloat16, ov::bfloat16>();
    } else if (_dt_a == llmdnn_bf16 && _dt_b == llmdnn_bf16 && _dt_c == llmdnn_f32) {
        do_test<ov::bfloat16, ov::bfloat16, float>();
    } else if (_dt_a == llmdnn_bf16 && _dt_b == llmdnn_f32 && _dt_c == llmdnn_bf16) {
        do_test<ov::bfloat16, float, ov::bfloat16>();
    } else if (_dt_a == llmdnn_bf16 && _dt_b == llmdnn_f32 && _dt_c == llmdnn_f32) {
        do_test<ov::bfloat16, float, float>();
    } else if (_dt_a == llmdnn_bf16 && _dt_b == llmdnn_s8 && _dt_c == llmdnn_f32) {
        do_test<ov::bfloat16, int8_t, float>();
    } else if (_dt_a == llmdnn_bf16 && _dt_b == llmdnn_s8 && _dt_c == llmdnn_bf16) {
        do_test<ov::bfloat16, int8_t, ov::bfloat16>();
    } else {
        ASSERT_TRUE(false);
    }
}

// supported:
//  (s8,s8,s8),dq,[bias],[gelu],q
//  (s8,s8,bf16),dq,[bias],[gelu]
//  (s8,s8,f32),dq,[bias],[gelu]
//  (bf16,bf16,bf16),[bias],[gelu]
//  (bf16,bf16,f32),[bias],[gelu]
//  (bf16,f32,bf16),[bias],[gelu]
//  (bf16,f32,f32),[bias],[gelu]
//  (bf16,s8,f32),dq,[bias],[gelu]
//  (bf16,s8,bf16),dq,[bias],[gelu]
const std::vector<FCKernelTestDTPost> types = {
    { llmdnn_s8, llmdnn_s8, llmdnn_s8, DEQUANT_QUANT },
    { llmdnn_s8, llmdnn_s8, llmdnn_s8, DEQUANT_BIAS_QUANT },
    { llmdnn_s8, llmdnn_s8, llmdnn_s8, DEQUANT_GELU_QUANT },
    { llmdnn_s8, llmdnn_s8, llmdnn_s8, DEQUANT_BIAS_GELU_QUANT },
    { llmdnn_s8, llmdnn_s8, llmdnn_s8, DEQUANT_GELU_TANH_QUANT },
    { llmdnn_s8, llmdnn_s8, llmdnn_s8, DEQUANT_BIAS_GELU_TANH_QUANT },
    { llmdnn_s8, llmdnn_s8, llmdnn_bf16, DEQUANT },
    { llmdnn_s8, llmdnn_s8, llmdnn_bf16, DEQUANT_BIAS },
    { llmdnn_s8, llmdnn_s8, llmdnn_bf16, DEQUANT_GELU },
    { llmdnn_s8, llmdnn_s8, llmdnn_bf16, DEQUANT_BIAS_GELU },
    { llmdnn_s8, llmdnn_s8, llmdnn_bf16, DEQUANT_GELU_TANH },
    { llmdnn_s8, llmdnn_s8, llmdnn_bf16, DEQUANT_BIAS_GELU_TANH },
    { llmdnn_s8, llmdnn_s8, llmdnn_f32, DEQUANT },
    { llmdnn_s8, llmdnn_s8, llmdnn_f32, DEQUANT_BIAS },
    { llmdnn_s8, llmdnn_s8, llmdnn_f32, DEQUANT_GELU },
    { llmdnn_s8, llmdnn_s8, llmdnn_f32, DEQUANT_BIAS_GELU },
    { llmdnn_s8, llmdnn_s8, llmdnn_f32, DEQUANT_GELU_TANH },
    { llmdnn_s8, llmdnn_s8, llmdnn_f32, DEQUANT_BIAS_GELU_TANH },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, NONE },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, BIAS },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, GELU },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, BIAS_GELU },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, GELU_TANH },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, BIAS_GELU_TANH },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_f32, NONE },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_f32, BIAS },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_f32, GELU },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_f32, BIAS_GELU },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_f32, GELU_TANH },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_f32, BIAS_GELU_TANH },
    { llmdnn_bf16, llmdnn_f32, llmdnn_bf16, NONE },
    { llmdnn_bf16, llmdnn_f32, llmdnn_bf16, BIAS },
    { llmdnn_bf16, llmdnn_f32, llmdnn_bf16, GELU },
    { llmdnn_bf16, llmdnn_f32, llmdnn_bf16, BIAS_GELU },
    { llmdnn_bf16, llmdnn_f32, llmdnn_bf16, GELU_TANH },
    { llmdnn_bf16, llmdnn_f32, llmdnn_bf16, BIAS_GELU_TANH },
    { llmdnn_bf16, llmdnn_f32, llmdnn_f32, NONE },
    { llmdnn_bf16, llmdnn_f32, llmdnn_f32, BIAS },
    { llmdnn_bf16, llmdnn_f32, llmdnn_f32, GELU },
    { llmdnn_bf16, llmdnn_f32, llmdnn_f32, BIAS_GELU },
    { llmdnn_bf16, llmdnn_f32, llmdnn_f32, GELU_TANH },
    { llmdnn_bf16, llmdnn_f32, llmdnn_f32, BIAS_GELU_TANH },
    // TODO: support weight compression
    // { llmdnn_bf16, llmdnn_s8, llmdnn_f32, DEQUANT },
    // { llmdnn_bf16, llmdnn_s8, llmdnn_f32, DEQUANT_BIAS },
    // { llmdnn_bf16, llmdnn_s8, llmdnn_f32, DEQUANT_GELU },
    // { llmdnn_bf16, llmdnn_s8, llmdnn_f32, DEQUANT_BIAS_GELU },
    // { llmdnn_bf16, llmdnn_s8, llmdnn_bf16, DEQUANT },
    // { llmdnn_bf16, llmdnn_s8, llmdnn_bf16, DEQUANT_BIAS },
    // { llmdnn_bf16, llmdnn_s8, llmdnn_bf16, DEQUANT_GELU },
    // { llmdnn_bf16, llmdnn_s8, llmdnn_bf16, DEQUANT_BIAS_GELU },
};

// M, N, K
const std::vector<FCKernelTestShape> shapes = {
    // normal
    {256, 48, 448},
    // k tail
    {256, 48, 449},
    // M tail == unroll 8
    {256 + 8, 48, 449},
    // M tail == unroll 8 + 2
    {256 + 10, 48, 449},
    // N tail
    {256, 40, 448},
    // all tail
    {256 + 9, 47, 449},
    // gemv, K <= 64(32)*6
    {256, 1, 80},
};

INSTANTIATE_TEST_SUITE_P(smoke_FCKernel, FCKernelTest,
    ::testing::Combine(ValuesIn(types),
                       Values(true, false),
                       ValuesIn(shapes)),
    FCKernelTest::getTestCaseName);
