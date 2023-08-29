// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "fc_amx.hpp"

namespace llmdnn {

// interface
fc::fc(): _impl(new_fc_impl_amx()) {
}

fc::~fc() {
    delete _impl;
}

bool fc::init(const fc_create_param& param) {
    return _impl->init(param);
}

void fc::pack_weight(const tensor& w) {
    return _impl->pack_weight(w);
}

status_t fc::exec(const tensor& input, const tensor& output, const tensor& dq, const tensor& q, const tensor& bias) {
    return _impl->exec(input, output, dq, q, bias);
}

}  // namespace llmdnn
