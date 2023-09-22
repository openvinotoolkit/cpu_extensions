// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef ENABLE_LOG
    #define DEBUG_LOG std::cout
#else
    #define DEBUG_LOG if (0) std::cout
#endif
