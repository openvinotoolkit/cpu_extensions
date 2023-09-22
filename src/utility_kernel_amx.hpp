// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdio.h>
#include <stdint.h>
#include <initializer_list>
#include <vector>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

struct tileconfig_t {
    uint8_t palette_id;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t cols[16];
    uint8_t rows[16];
    tileconfig_t() = default;

    tileconfig_t(int palette, int _startRow, const std::initializer_list<std::pair<int, int>> &_rows_columnsBytes) {
        palette_id = palette;
        startRow = _startRow;
        int i;
        for(i = 0; i < 14; i++) {
            reserved[i] = 0;
        }
        i = 0;
        for (const auto& ele : _rows_columnsBytes) {
            rows[i] = ele.first;
            cols[i] = ele.second;
            i++;
        }
        for(; i < 16; i++) {
            cols[i] = 0;
            rows[i] = 0;
        }
        load();
    }

    tileconfig_t(int palette, int _startRow, const std::initializer_list<int> &_rows, int columnsBytes) {
        palette_id = palette;
        startRow = _startRow;
        int i;
        for(i = 0; i < 14; i++) {
            reserved[i] = 0;
        }
        i = 0;
        for (const auto ele : _rows) {
            rows[i] = ele;
            cols[i] = columnsBytes;
            i++;
        }
        for(; i < 16; i++) {
            cols[i] = 0;
            rows[i] = 0;
        }
        load();
    }

    tileconfig_t(int palette, int _startRow, int numTiles, int _rows, int columnsBytes) {
        palette_id = palette;
        startRow = _startRow;
        int i;
        for(i = 0; i < 14; i++) {
            reserved[i] = 0;
        }
        for(i = 0; i < numTiles; i++) {
            rows[i] = _rows;
            cols[i] = columnsBytes;
        }
        for(; i < 16; i++) {
            cols[i] = 0;
            rows[i] = 0;
        }
        load();
    }

    ~tileconfig_t() {
        _tile_release();
    }

    void __attribute__((noinline)) load() {
        _tile_loadconfig(this);
    }

    void store() {
        _tile_storeconfig(this);
    }
} __attribute__ ((__packed__));
