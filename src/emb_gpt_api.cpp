// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "emb_gpt_avx512.hpp"

namespace llmdnn {

void emb_gpt(const tensor& q_src,
             const tensor& k_src,
             const tensor& v_src,
             const tensor& k_past,
             const tensor& v_past,
             const tensor& q_dst,
             const tensor& k_dst,
             const tensor& v_dst,
             const tensor& cos,
             const tensor& sin,
             const tensor& position2d_ids) {
    emb_gpt_avx512(q_src, k_src, v_src, k_past, v_past, q_dst, k_dst, v_dst, cos, sin, position2d_ids);
}

}