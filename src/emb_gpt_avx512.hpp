// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "llm_types.hpp"
#include "llm_emb_gpt.hpp"

namespace llmdnn {

status_t emb_gpt_avx512(const tensor& q_src,
                        const tensor& k_src,
                        const tensor& v_src,
                        const tensor& k_past,
                        const tensor& v_past,
                        const tensor& q_dst,
                        const tensor& k_dst,
                        const tensor& v_dst,
                        const tensor& cos,
                        const tensor& sin,
                        const tensor& position2d_ids);
}  // namespace llmdnn
