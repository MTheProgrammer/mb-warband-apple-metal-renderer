#pragma once

// dyld interposing: every image except this one that calls REPLACEE gets
// REPLACEMENT instead. Calls to REPLACEE from inside the shim reach the original.
#define WB_INTERPOSE(REPLACEMENT, REPLACEE)                                        \
    __attribute__((used)) static const struct {                                    \
        const void *replacement;                                                   \
        const void *replacee;                                                      \
    } wb_interpose_##REPLACEE __attribute__((section("__DATA,__interpose"))) = {   \
        (const void *)(unsigned long)&REPLACEMENT,                                 \
        (const void *)(unsigned long)&REPLACEE}
