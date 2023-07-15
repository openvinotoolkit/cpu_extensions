// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory.h>
#include <vector>
#include <string>

// gcc 9 does not recognize 'std::__throw_bad_array_new_length()' which is imported by
//   gcc 11. The symbol exists in std::allocator::allocate, use custom to wa.
template<typename T>
class custom_allocator {
public:
    using value_type = T;
    custom_allocator() noexcept = default;
    template <class U>
    custom_allocator (const custom_allocator<U>&) noexcept {}
    inline T* allocate(std::allocator<void>::size_type cnt, typename std::allocator<void>::const_pointer = 0) {
        return static_cast<T*>(::operator new(cnt * sizeof(T)));
    }
    void deallocate (T* p, std::size_t n) {
        ::operator delete(p);
    }
};

template <class T, class U>
bool operator==(custom_allocator<T> const&, custom_allocator<U> const&) noexcept {
    return true;
}

template <class T, class U>
bool operator!=(custom_allocator<T> const& x, custom_allocator<U> const& y) noexcept {
    return !(x == y);
}

template<typename T>
using llm_vector = std::vector<T, custom_allocator<T>>;
