// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "llm_types.hpp"
#include "llm_fc.hpp"

namespace llmdnn {

fc::impl* new_fc_impl_amx();

}  // namespace llmdnn
