#!/bin/sh
# Starts the game with every optional rendering improvement switched on, on top
# of the ones that are on by default (soft shadows, 4x multisampling, reversed
# depth). Anything set in the environment already is kept, so one option can be
# turned off for a comparison:   WBM_AO=0 scripts/run-all.sh
export WBM_MSAA=${WBM_MSAA:-1}        # no multisampling: it is the costliest setting for the GPU (4 restores it)
export WBM_FXAA=${WBM_FXAA:-1}        # edges smoothed after the fact instead
# The frame rate is the game's to set (max_framerate in its rgl_config.txt). For a cooler,
# quieter machine a lower limit can be put on top of it: WBM_MAX_FPS=60 scripts/run-all.sh,
# and WBM_MENU_FPS for menus and loading screens. The lower of the limits applies.
export WBM_SHADOWS=${WBM_SHADOWS:-2}  # shadows that soften with distance (PCSS)
export WBM_DYNAMIC_SHADOWS=${WBM_DYNAMIC_SHADOWS:-1}  # characters cast shadows on everything
export WBM_PBR=${WBM_PBR:-3}          # microfacet highlights, metalness, reflection of the scene's sky
export WBM_SRGB=${WBM_SRGB:-1}        # hardware sRGB texture decoding
export WBM_AO=${WBM_AO:-1}            # ambient occlusion
export WBM_CONTACT_SHADOWS=${WBM_CONTACT_SHADOWS:-1}  # screen-space contact shadows
export WBM_BOUNCE_LIGHT=${WBM_BOUNCE_LIGHT:-1}  # lit surfaces throw a little light and colour on what is near (a strength; 0: none)
export WBM_LIGHT_SHAFTS=${WBM_LIGHT_SHAFTS:-1}  # light shafts from the sun
export WBM_MOTION_BLUR=${WBM_MOTION_BLUR:-0}    # blur from the camera's movement and of things that move by themselves (off for now; 1 turns it on)
export WBM_SKY_AMBIENT=${WBM_SKY_AMBIENT:-1}    # ambient light tinted by the sky, with ground bounce
export WBM_MATERIALS=${WBM_MATERIALS:-1}        # metalness and roughness by material class
export WBM_WIND=${WBM_WIND:-1}                  # trees, bushes and grass sway (a strength; 0 for none)
export WBM_FOLIAGE=${WBM_FOLIAGE:-0.6}          # sunlight through leaves seen from their far side
export WBM_FUR=${WBM_FUR:-1}                    # a soft, velvety coat on horses (needs coat maps: build/texture_tool pbr)
export WBM_FUR_SHELLS=${WBM_FUR_SHELLS:-12}     # layers of fur on horses within a few metres, and long hair where a hair map says (0: none)
export WBM_HAIR_STRANDS=${WBM_HAIR_STRANDS:-1} # mane and tail as strands in place of layers (0: layers)
export WBM_PEOPLE_HAIR=${WBM_PEOPLE_HAIR:-1}   # loose strands over the hair and beards of people within a few metres
export WBM_PARALLAX=${WBM_PARALLAX:-0.05}     # stones stand out from their joints on walls and buildings nearby (0: none)
export WBM_EYES=${WBM_EYES:-1}                  # a highlight and a reflection in the eyes of faces
export WBM_SSS=${WBM_SSS:-1}                    # light scattering in the skin of faces
# For speed, not looks, so not set here: WBM_UPSCALE=0.67 draws the scene smaller and lets MetalFX enlarge it;
# WBM_SHARPEN=0.3 sharpens the picture. Set either before calling this script.
export WBM_TONEMAP=${WBM_TONEMAP:-1}  # filmic tone map
export WBM_HDR=${WBM_HDR:-1}          # extended-range output on HDR displays
exec "$(dirname "$0")/run.sh" "$@"
