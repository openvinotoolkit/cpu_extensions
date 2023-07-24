// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "mha_gpt_amx.hpp"

namespace llmdnn {

// interface
mha_gpt::mha_gpt(): _impl(new_impl_amx()) {
}

mha_gpt::~mha_gpt() {
    delete _impl;
}

void mha_gpt::exec(const tensor& q, const tensor& k, const tensor& v, const tensor& output, const tensor& attn_mask, const tensor& alibi, const tensor& causal_mask, bool select_nfltmax_at_0, float normal_factor, bool use_causal_mask) {
    _impl->exec(q, k, v, output, attn_mask, alibi, causal_mask, select_nfltmax_at_0, normal_factor, use_causal_mask);
}

}