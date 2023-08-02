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
#include <string>
#include "gtest/gtest.h"
#include "common/tensor2d.hpp"
#include "common/tensor2d_helper.hpp"
#include "gelu_kernel_avx512.hpp"
#include "test_common.hpp"

using namespace std;
using namespace llmdnn;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

using GeluTestParamSet = std::tuple<
        std::string                                // data type
        >;

class GeluTest : public TestWithParam<GeluTestParamSet> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GeluTestParamSet>& obj) {
        std::string types;
        std::tie(types) = obj.param;

        std::ostringstream result;
        result << types;
        return result.str();
    }

protected:
    virtual void SetUp() override {
        std::tie(_types) = GetParam();
    };

    void gelu_ref(float* s, float* d, int n) {
        if (_types == "tanh") {
            for (int i = 0; i < n; i++) {
                auto x = s[i];
                d[i] = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / 3.1415926f) * x * (1 + 0.044715f * x * x)));
            }
        } else {
            for (int i = 0; i < n; i++) {
                auto x = s[i];
                d[i] = 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
            }
        }
    }

    void test(float thresh) {
        tensor2D<float> src(1, 16, true);
        tensor2D<float> dst(1, 16, true);
        tensor2D<float> ref(1, 16, true);
        for (int i = 0; i < 16; i++) {
            src[i] = std::sqrt(i) - 2;
        }
        __m512 s = _mm512_loadu_ps(src.data);
        __m512 d;
        if (_types == "tanh")
            d = gelu_tanh_avx512(s);
        else
            d = gelu_erf_minmax_approx_avx512(s);
        _mm512_storeu_ps(dst.data, d);
        gelu_ref(src.data, ref.data, 16);
        for (int i = 0; i < src.dims[1]; i++) {
            float r = ref[i];
            float c = dst[i];
            if (std::abs(r - c) > thresh) {
                FAIL() << " cur is not equal, pos: " << i << " opt: " << c << " ref: " << r;
            }
        }
    }

    std::string _types;
};

TEST_P(GeluTest, Gelu) {
    test(0.01f);
}

const std::vector<std::string> types = {
    "tanh", "erf"
};

INSTANTIATE_TEST_SUITE_P(smoke_Gelu, GeluTest,
    ::testing::Combine(ValuesIn(types)),
    GeluTest::getTestCaseName);
