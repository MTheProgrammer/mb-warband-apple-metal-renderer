// Ambient occlusion, contact shadows, light shafts and bounce light (src/backend/ambient_occlusion.mm):
// full-screen passes over the scene buffer that s_last describes, worked out from its depth and
// colour once the scene's last pass has ended (see end_pass).
#pragma once

#include <cstdint>

namespace wbm __attribute__((visibility("hidden"))) {

// Runs the passes on the scene buffer described by s_last. Called between render passes, with none open.
void apply_ambient_occlusion();

// For the statistics (mtl_dump_stats).
extern uint64_t s_bounce_frames;

} // namespace wbm
