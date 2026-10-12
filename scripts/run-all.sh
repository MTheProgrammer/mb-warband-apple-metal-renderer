#!/bin/sh
# Starts the game with every optional rendering improvement switched on, on top
# of the ones that are on by default (soft shadows, 4x multisampling, reversed
# depth). Anything set in the environment already is kept, so one option can be
# turned off for a comparison:   WBM_AO=0 scripts/run-all.sh
export WBM_PBR=${WBM_PBR:-2}          # microfacet highlights, metalness, environment reflection
export WBM_SRGB=${WBM_SRGB:-1}        # hardware sRGB texture decoding
export WBM_AO=${WBM_AO:-1}            # ambient occlusion
export WBM_TONEMAP=${WBM_TONEMAP:-1}  # filmic tone map
export WBM_HDR=${WBM_HDR:-1}          # extended-range output on HDR displays
exec "$(dirname "$0")/run.sh" "$@"
