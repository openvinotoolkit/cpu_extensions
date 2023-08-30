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
#include "llm_tensor.hpp"
#include "llm_types.hpp"
#include "test_common.hpp"

using namespace std;
using namespace llmdnn;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

using FCTestShape = std::tuple<size_t, size_t, size_t>;
using FCTestDTPost = std::tuple<data_type_t,      // dt_a
                                data_type_t,      // dt_b
                                data_type_t,      // dt_c
                                data_type_t,      // type of pack weight
                                postops_types>;
using FCTestParamSet = std::tuple<
        FCTestDTPost,                          // a, b, c data type, postops
        bool,                                  // b needs transpose
        FCTestShape                            // M, N, K
        >;

class FCTest : public TestWithParam<FCTestParamSet> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FCTestParamSet>& obj) {
        FCTestDTPost types;
        bool is_transpose;
        postops_types postops_type;
        data_type_t dt_a, dt_b, dt_c, dt_weight;
        FCTestShape shape;
        int M, N, K;
        std::tie(types, is_transpose, shape) = obj.param;
        std::tie(M, N, K) = shape;
        std::tie(dt_a, dt_b, dt_c, dt_weight, postops_type) = types;

        std::ostringstream result;
        result << "A_" << dtype_to_str(dt_a) << "_B_" << dtype_to_str(dt_b)
               << "_C_" << dtype_to_str(dt_c) << "_WEIGHT_" << dtype_to_str(dt_weight)
               << (is_transpose ? "_transpose" : "")
               << "_postops_" << postops_type << "_M_" << M << "_N_" << N << "_K_" << K;
        return result.str();
    }

protected:
    virtual void SetUp() override {
        initXTILE();

        FCTestShape shape;
        FCTestDTPost types;
        std::tie(types, _is_transpose, shape) = GetParam();
        std::tie(_M, _N, _K) = shape;
        std::tie(_dt_a, _dt_b, _dt_c, _dt_weight, _postops_type) = types;
    };

    template<typename TA, typename TB, typename TC>
    void do_test() {
        fc_create_param param = {
            _dt_a, _dt_b, _dt_c,
            _is_transpose, _postops_type
        };
        llmdnn::fc fc;
        ASSERT_TRUE(fc.init(param));

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

        tensor2D<TB> BT = B.Tr(true);
        TB* ptr_B;
        size_t ldb;
        tensor weight;
        if (_is_transpose) {
            ptr_B = BT.data;
            ldb = BT.stride;
            weight.resize({ static_cast<size_t>(BT.dims[0]), static_cast<size_t>(BT.dims[1]) }, static_cast<TB*>(ptr_B));
        } else {
            ptr_B = B.data;
            ldb = B.stride;
            weight.resize({ static_cast<size_t>(B.dims[0]), static_cast<size_t>(B.dims[1]) }, static_cast<TB*>(ptr_B));
        }
        fc.pack_weight(weight);
        tensor input, output, bias_t, q_t, dq_t;
        input.resize({ static_cast<size_t>(A.dims[0]), static_cast<size_t>(A.dims[1]) }, static_cast<TA*>(A.data));
        output.resize({ static_cast<size_t>(C.dims[0]), static_cast<size_t>(C.dims[1]) }, static_cast<TC*>(C.data));
        dq_t.resize({ static_cast<size_t>(dq.dims[0]), static_cast<size_t>(dq.dims[1]) }, dq.data);
        q_t.resize({ static_cast<size_t>(q.dims[0]), static_cast<size_t>(q.dims[1]) }, q.data);
        bias_t.resize({ static_cast<size_t>(bias.dims[0]), static_cast<size_t>(bias.dims[1]) }, bias.data);
        ASSERT_TRUE(fc.exec(input, output, dq_t, q_t, bias_t) == llmdnn::status_ok);
        C_Ref = 0;
        float* ptr_dq = nullptr;
        float* ptr_q = nullptr;
        float* ptr_bias = nullptr;
        func_act act = func_act(); 
        if ((_postops_type & DEQUANT) && _dt_a == llmdnn::llmdnn_s8) {
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
    data_type_t _dt_a, _dt_b, _dt_c, _dt_weight;
};

TEST_P(FCTest, Func) {
    if (_dt_a == llmdnn_s8 && _dt_weight == llmdnn_s8 && _dt_c == llmdnn_s8) {
        do_test<int8_t, int8_t, int8_t>();
    } else if (_dt_a == llmdnn_s8 && _dt_weight == llmdnn_s8 && _dt_c == llmdnn_bf16) {
        do_test<int8_t, int8_t, ov::bfloat16>();
    } else if (_dt_a == llmdnn_s8 && _dt_weight == llmdnn_s8 && _dt_c == llmdnn_f32) {
        do_test<int8_t, int8_t, float>();
    } else if (_dt_a == llmdnn_bf16 && _dt_weight == llmdnn_bf16 && _dt_c == llmdnn_bf16) {
        do_test<ov::bfloat16, ov::bfloat16, ov::bfloat16>();
    } else if (_dt_a == llmdnn_bf16 && _dt_weight == llmdnn_bf16 && _dt_c == llmdnn_f32) {
        do_test<ov::bfloat16, ov::bfloat16, float>();
    } else if (_dt_a == llmdnn_bf16 && _dt_weight == llmdnn_f32 && _dt_c == llmdnn_bf16) {
        do_test<ov::bfloat16, float, ov::bfloat16>();
    } else if (_dt_a == llmdnn_bf16 && _dt_weight == llmdnn_f32 && _dt_c == llmdnn_f32) {
        do_test<ov::bfloat16, float, float>();
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
//  (bf16,s8,f32),dq,[bias],[gelu]
//  (bf16,s8,bf16),dq,[bias],[gelu]
const std::vector<FCTestDTPost> types = {
    { llmdnn_s8, llmdnn_s8, llmdnn_s8, llmdnn_s8, DEQUANT_QUANT },
    { llmdnn_s8, llmdnn_s8, llmdnn_s8, llmdnn_s8, DEQUANT_BIAS_QUANT },
    { llmdnn_s8, llmdnn_s8, llmdnn_s8, llmdnn_s8, DEQUANT_GELU_QUANT },
    { llmdnn_s8, llmdnn_s8, llmdnn_s8, llmdnn_s8, DEQUANT_BIAS_GELU_QUANT },
    { llmdnn_s8, llmdnn_s8, llmdnn_s8, llmdnn_s8, DEQUANT_GELU_TANH_QUANT },
    { llmdnn_s8, llmdnn_s8, llmdnn_s8, llmdnn_s8, DEQUANT_BIAS_GELU_TANH_QUANT },
    { llmdnn_s8, llmdnn_s8, llmdnn_bf16, llmdnn_s8, DEQUANT },
    { llmdnn_s8, llmdnn_s8, llmdnn_bf16, llmdnn_s8, DEQUANT_BIAS },
    { llmdnn_s8, llmdnn_s8, llmdnn_bf16, llmdnn_s8, DEQUANT_GELU },
    { llmdnn_s8, llmdnn_s8, llmdnn_bf16, llmdnn_s8, DEQUANT_BIAS_GELU },
    { llmdnn_s8, llmdnn_s8, llmdnn_bf16, llmdnn_s8, DEQUANT_GELU_TANH },
    { llmdnn_s8, llmdnn_s8, llmdnn_bf16, llmdnn_s8, DEQUANT_BIAS_GELU_TANH },
    { llmdnn_s8, llmdnn_s8, llmdnn_f32, llmdnn_s8, DEQUANT },
    { llmdnn_s8, llmdnn_s8, llmdnn_f32, llmdnn_s8, DEQUANT_BIAS },
    { llmdnn_s8, llmdnn_s8, llmdnn_f32, llmdnn_s8, DEQUANT_GELU },
    { llmdnn_s8, llmdnn_s8, llmdnn_f32, llmdnn_s8, DEQUANT_BIAS_GELU },
    { llmdnn_s8, llmdnn_s8, llmdnn_f32, llmdnn_s8, DEQUANT_GELU_TANH },
    { llmdnn_s8, llmdnn_s8, llmdnn_f32, llmdnn_s8, DEQUANT_BIAS_GELU_TANH },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, NONE },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, BIAS },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, GELU },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, BIAS_GELU },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, GELU_TANH },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, BIAS_GELU_TANH },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_f32, llmdnn_bf16, NONE },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_f32, llmdnn_bf16, BIAS },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_f32, llmdnn_bf16, GELU },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_f32, llmdnn_bf16, BIAS_GELU },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_f32, llmdnn_bf16, GELU_TANH },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_f32, llmdnn_bf16, BIAS_GELU_TANH },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, llmdnn_f32, NONE },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, llmdnn_f32, BIAS },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, llmdnn_f32, GELU },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, llmdnn_f32, BIAS_GELU },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, llmdnn_f32, GELU_TANH },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_bf16, llmdnn_f32, BIAS_GELU_TANH },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_f32, llmdnn_f32, NONE },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_f32, llmdnn_f32, BIAS },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_f32, llmdnn_f32, GELU },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_f32, llmdnn_f32, BIAS_GELU },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_f32, llmdnn_f32, GELU_TANH },
    { llmdnn_bf16, llmdnn_bf16, llmdnn_f32, llmdnn_f32, BIAS_GELU_TANH },
    // weight compression
    { llmdnn_bf16, llmdnn_s8, llmdnn_f32, llmdnn_bf16, DEQUANT },
    { llmdnn_bf16, llmdnn_s8, llmdnn_f32, llmdnn_bf16, DEQUANT_BIAS },
    { llmdnn_bf16, llmdnn_s8, llmdnn_f32, llmdnn_bf16, DEQUANT_GELU },
    { llmdnn_bf16, llmdnn_s8, llmdnn_f32, llmdnn_bf16, DEQUANT_BIAS_GELU },
    { llmdnn_bf16, llmdnn_s8, llmdnn_bf16, llmdnn_bf16, DEQUANT },
    { llmdnn_bf16, llmdnn_s8, llmdnn_bf16, llmdnn_bf16, DEQUANT_BIAS },
    { llmdnn_bf16, llmdnn_s8, llmdnn_bf16, llmdnn_bf16, DEQUANT_GELU },
    { llmdnn_bf16, llmdnn_s8, llmdnn_bf16, llmdnn_bf16, DEQUANT_BIAS_GELU },
    { llmdnn_bf16, llmdnn_s8, llmdnn_f32, llmdnn_f32, DEQUANT },
    { llmdnn_bf16, llmdnn_s8, llmdnn_f32, llmdnn_f32, DEQUANT_BIAS },
    { llmdnn_bf16, llmdnn_s8, llmdnn_f32, llmdnn_f32, DEQUANT_GELU },
    { llmdnn_bf16, llmdnn_s8, llmdnn_f32, llmdnn_f32, DEQUANT_BIAS_GELU },
    { llmdnn_bf16, llmdnn_s8, llmdnn_bf16, llmdnn_f32, DEQUANT },
    { llmdnn_bf16, llmdnn_s8, llmdnn_bf16, llmdnn_f32, DEQUANT_BIAS },
    { llmdnn_bf16, llmdnn_s8, llmdnn_bf16, llmdnn_f32, DEQUANT_GELU },
    { llmdnn_bf16, llmdnn_s8, llmdnn_bf16, llmdnn_f32, DEQUANT_BIAS_GELU },
};

// M, N, K
const std::vector<FCTestShape> shapes = {
    // normal
    {256, 128, 448},
    // k tail
    {256, 48, 449},
    // M tail == unroll 8
    {256 + 8, 48, 449},
    // M tail == unroll 8 + 2
    {256 + 10, 48, 449},
    // N tail
    {256, 95, 448},
    // all tail
    {256 + 9, 47, 449},
    // gemv, K <= 64(32)*6
    {256, 1, 80},
};

INSTANTIATE_TEST_SUITE_P(smoke_FC, FCTest,
    ::testing::Combine(ValuesIn(types),
                       Values(true, false),
                       ValuesIn(shapes)),
    FCTest::getTestCaseName);
