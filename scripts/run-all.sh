#!/bin/sh
# Starts the game with every optional rendering improvement switched on, on top
# of the ones that are on by default (soft shadows, 4x multisampling, reversed
# depth). Anything set in the environment already is kept, so one option can be
# turned off for a comparison:   WBM_AO=0 scripts/run-all.sh
export WBM_SHADOWS=${WBM_SHADOWS:-2}  # shadows that soften with distance (PCSS)
export WBM_DYNAMIC_SHADOWS=${WBM_DYNAMIC_SHADOWS:-1}  # characters cast shadows on everything
export WBM_PBR=${WBM_PBR:-3}          # microfacet highlights, metalness, reflection of the scene's sky
export WBM_SRGB=${WBM_SRGB:-1}        # hardware sRGB texture decoding
export WBM_AO=${WBM_AO:-1}            # ambient occlusion
export WBM_CONTACT_SHADOWS=${WBM_CONTACT_SHADOWS:-1}  # screen-space contact shadows
export WBM_LIGHT_SHAFTS=${WBM_LIGHT_SHAFTS:-1}  # light shafts from the sun
export WBM_MOTION_BLUR=${WBM_MOTION_BLUR:-1}    # blur from the camera's movement
export WBM_SKY_AMBIENT=${WBM_SKY_AMBIENT:-1}    # ambient light tinted by the sky, with ground bounce
export WBM_MATERIALS=${WBM_MATERIALS:-1}        # metalness and roughness by material class
export WBM_TONEMAP=${WBM_TONEMAP:-1}  # filmic tone map
export WBM_HDR=${WBM_HDR:-1}          # extended-range output on HDR displays
exec "$(dirname "$0")/run.sh" "$@"
