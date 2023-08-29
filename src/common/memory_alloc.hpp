// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory.h>
#include "compatible.hpp"

void* llmdnn_alloc(size_t aligned_size, size_t size, bool hint_numa = true);
void llmdnn_free(void* p, size_t size, bool hint_numa = true);

llm_vector<int> llmdnn_get_numa_nodes();
void* llmdnn_alloc_on(size_t aligned_size, size_t size, int numa_id);
void llmdnn_free_on(void* p, size_t size);
int llmdnn_get_numa_id_for_cur_task();