#pragma once

#ifdef __linux__
#include <sys/mman.h>
#endif
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

inline void secure_zero_memory(void *ptr, size_t size) {
    if (ptr != nullptr && size > 0) {
#if defined(__STDC_LIB_EXT1__)
        memset_s(ptr, size, 0, size);
#else
        volatile unsigned char *p = (volatile unsigned char *)ptr;
        while (size--) {
            *p++ = 0;
        }
#endif
    }
}

inline int secure_mlock(void *ptr, size_t size) {
#ifdef __linux__
    if (ptr != nullptr && size > 0) {
        return mlock(ptr, size);
    }
#endif
    return -1;
}

inline int secure_munlock(void *ptr, size_t size) {
#ifdef __linux__
    if (ptr != nullptr && size > 0) {
        return munlock(ptr, size);
    }
#endif
    return -1;
}

#ifdef __cplusplus
}
#endif
