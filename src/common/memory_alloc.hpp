// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory.h>

void* llmdnn_alloc(size_t aligned_size, size_t size, bool hint_numa = true);
void llmdnn_free(void* p, size_t size, bool hint_numa = true);
