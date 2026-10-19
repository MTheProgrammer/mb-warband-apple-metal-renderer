// Command buffers, render targets and passes; uniform storage.
// Included by metal_backend.mm, inside its anonymous namespace.

// ---- command buffer and passes -----------------------------------------------------

// The time the GPU spends on each command buffer, added up (for the summary: how heavy a frame is for the
// GPU, which wall-clock time over a run cannot say on a machine doing other things).
std::atomic<double> s_gpu_seconds{0};
std::atomic<uint64_t> s_gpu_buffers{0};
id<MTLCommandBuffer> timed(id<MTLCommandBuffer> cmd)
{
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> done) {
      double t = done.GPUEndTime - done.GPUStartTime;
      if (t > 0 && t < 1) {
          double was = s_gpu_seconds.load();
          while (!s_gpu_seconds.compare_exchange_weak(was, was + t)) {}
          s_gpu_buffers++;
      }
    }];
    return cmd;
}

id<MTLCommandBuffer> command_buffer()
{
    if (!s_cmd) s_cmd = timed([s_queue commandBuffer]);
    return s_cmd;
}

void end_pass()
{
    if (!s_pass) return;
    [s_pass endEncoding];
    s_pass = nil;
    s_pass_color = s_pass_depth = nil;
}

id<MTLTexture> new_render_texture(MTLPixelFormat format, NSUInteger width, NSUInteger height, int samples)
{
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; // ambient occlusion reads the depth buffer
    d.storageMode = MTLStorageModePrivate;
    if (samples > 1) {
        d.textureType = MTLTextureType2DMultisample;
        d.sampleCount = (NSUInteger)samples;
    }
    return [s_device newTextureWithDescriptor:d];
}

id<MTLTexture> new_depth_texture(NSUInteger width, NSUInteger height, int samples)
{
    return new_render_texture(MTLPixelFormatDepth32Float, width, height, samples);
}

struct Target {
    id<MTLTexture> color, depth; // color is what later passes sample; with multisampling it receives the resolve
    id<MTLTexture> ms_color;     // what is rendered into when multisampled
    bool screen = false;
    bool scene = false; // the engine's HDR scene buffer: screen-sized, RGBA16F, with depth
    bool prepass = false; // the engine's depth pre-pass: screen-sized, R16F, with depth
    bool shadow_map = false; // the engine's own shadow map of the scenery: large, square, R32F
    double scale = 1;
    int samples = 1;
};

// The attachments the bound framebuffer currently describes. Returns false
// when there is nothing to draw into.
bool resolve_target(Target &t)
{
    if (s_fbo == 0) {
        if (!s_drawable) {
            if (!s_layer) return false;
            s_drawable = [s_layer nextDrawable];
            if (!s_drawable) return false;
        }
        t.color = s_drawable.texture;
        // When the engine renders the 3D scene into its own HDR buffer (this frame
        // or the last), all the screen receives is one full-screen image and the
        // UI. Multisampling that gains nothing and costs a full-screen resolve,
        // so the screen is then single-sampled.
        int samples = (s_scene_frame + 1 >= s_frame_serial) ? 1 : s_msaa;
        if (!s_screen_depth || s_screen_depth.width != t.color.width || s_screen_depth.height != t.color.height ||
            (int)s_screen_depth.sampleCount != samples) {
            s_screen_depth = new_depth_texture(t.color.width, t.color.height, samples);
            s_screen_ms = samples > 1 ? new_render_texture(t.color.pixelFormat, t.color.width, t.color.height, samples) : nil;
        }
        t.depth = s_screen_depth;
        t.ms_color = s_screen_ms;
        t.samples = samples;
        t.screen = true;
        t.scale = s_scale;
        return true;
    }
    Framebuffer *fb = find(s_framebuffers, s_fbo);
    if (!fb) return false;
    if (Texture *tex = find(s_textures, fb->color_texture)) {
        t.color = tex->mtl;
        t.scale = tex->scale;
        if (tex->mtl && tex->screen_sized && s_msaa > 1) {
            if (!tex->ms) tex->ms = new_render_texture(tex->mtl.pixelFormat, tex->mtl.width, tex->mtl.height, s_msaa);
            t.ms_color = tex->ms;
            t.samples = s_msaa;
        }
    }
    if (Renderbuffer *rb = find(s_renderbuffers, fb->depth_renderbuffer)) {
        t.depth = rb->mtl;
        if (!t.color) t.scale = rb->scale;
    } else if (Texture *tex = find(s_textures, fb->depth_texture)) {
        t.depth = tex->mtl;
        if (!t.color) t.scale = tex->scale;
    }
    if (!t.color && t.depth) t.samples = (int)t.depth.sampleCount;
    if (Texture *tex = find(s_textures, fb->color_texture))
        t.scene = tex->screen_sized && tex->format == MTLPixelFormatRGBA16Float && t.depth != nil;
    if (Texture *tex = find(s_textures, fb->color_texture))
        t.prepass = tex->screen_sized && tex->format == MTLPixelFormatR16Float && t.depth != nil;
    if (Texture *tex = find(s_textures, fb->color_texture))
        t.shadow_map = !tex->screen_sized && tex->format == MTLPixelFormatR32Float && tex->mtl.width >= 2048 && tex->mtl.width == tex->mtl.height;
    if (t.color && t.depth && (int)t.depth.sampleCount != t.samples) {
        complain("framebuffer mixes a %d-sample colour target with a %d-sample depth buffer; depth is ignored", t.samples,
                 (int)t.depth.sampleCount);
        t.depth = nil;
    }
    return t.color || t.depth;
}

// Opens a render pass on the target. clear_mask selects what the pass clears
// as it starts; everything else is loaded.
void start_pass(const Target &t, GLbitfield clear_mask)
{
    end_pass();
    // The engine has drawn the scene and is moving to another target: this is
    // where post-processing starts, and the point to darken the scene first.
    // (A scene is told from other uses of its buffer by the number of draws; a previewed model is a scene of few.)
    if ((s_ao || s_contact_shadows || s_light_shafts || s_bounce_light > 0 || s_room_probe || s_motion_blur || s_fxaa || s_upscale || s_sharpen > 0) && s_last.scene && s_last.color != t.color && (s_last.draws > 100 || s_preview_in_scene || (s_last.draws > 8 && s_prepass_frame == s_frame_serial)) &&
        s_last.occluded_in_frame != s_frame_serial) {
        s_last.occluded_in_frame = s_frame_serial;
        if (s_ao || s_contact_shadows || s_light_shafts || s_bounce_light > 0) apply_ambient_occlusion();
        // Indoors is where the engine gives the sun no colour. On leaving, the room is forgotten.
        s_interior = s_room_probe && s_sun_valid && s_sun_color[0] + s_sun_color[1] + s_sun_color[2] < 0.01f;
        if (s_interior) update_room_probe();
        else s_room_atlas = nil;
        if (s_motion_blur) apply_motion_blur();
        if (s_fxaa && !s_temporal) apply_fxaa(); // last: edges as they will be shown (the temporal scaler smooths them itself)
        id<MTLTexture> shown = nil;
        if (@available(macOS 13.0, *)) {
            if (s_temporal) shown = apply_temporal_upscale();
            if (!shown && s_upscale) shown = apply_upscale(); // s_temporal is off by now if its scaler could not be made
        }
        if (s_sharpen > 0) apply_sharpen(shown ? shown : s_last.color);
    }
    MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
    if (t.color) {
        bool clear = clear_mask & GL_COLOR_BUFFER_BIT;
        d.colorAttachments[0].loadAction = clear ? MTLLoadActionClear : MTLLoadActionLoad;
        if (t.ms_color) { // keep the samples for later passes on this target, and resolve for anything that reads it
            d.colorAttachments[0].texture = t.ms_color;
            d.colorAttachments[0].resolveTexture = t.color;
            d.colorAttachments[0].storeAction = MTLStoreActionStoreAndMultisampleResolve;
        } else {
            d.colorAttachments[0].texture = t.color;
            d.colorAttachments[0].storeAction = MTLStoreActionStore;
        }
        d.colorAttachments[0].clearColor = s_debug_clear && t.screen ? MTLClearColorMake(0.10, 0.30, 0.60, 1.0) : s_clear_color;
    }
    bool with_motion = s_temporal && s_shader_options.motion_vectors && t.scene && t.color && !t.ms_color;
    if (with_motion) {
        if (!s_scene_motion || s_scene_motion.width != t.color.width || s_scene_motion.height != t.color.height) {
            MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:t.color.width height:t.color.height mipmapped:NO];
            td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            td.storageMode = MTLStorageModePrivate;
            s_scene_motion = [s_device newTextureWithDescriptor:td];
            s_scene_motion_frame = ~0ull;
        }
        d.colorAttachments[1].texture = s_scene_motion;
        d.colorAttachments[1].loadAction = s_scene_motion_frame == s_frame_serial ? MTLLoadActionLoad : MTLLoadActionClear;
        d.colorAttachments[1].storeAction = MTLStoreActionStore;
        d.colorAttachments[1].clearColor = MTLClearColorMake(0, 0, -1, 0);
        s_scene_motion_frame = s_frame_serial;
    }
    if (t.depth) {
        bool clear = (clear_mask & GL_DEPTH_BUFFER_BIT) && s_depth_mask;
        d.depthAttachment.texture = t.depth;
        d.depthAttachment.loadAction = clear ? MTLLoadActionClear : MTLLoadActionLoad;
        d.depthAttachment.storeAction = MTLStoreActionStore;
        d.depthAttachment.clearDepth = s_reverse_depth ? 0.0 : 1.0; // the far plane
    }
    s_pass = [command_buffer() renderCommandEncoderWithDescriptor:d];
    reset_pass_state();
    s_pass_color = t.color;
    s_pass_depth = t.depth;
    s_pass_screen = t.screen;
    s_pass_motion = with_motion;
    s_pass_scale = t.scale;
    s_pass_samples = t.samples;
    if (t.scene) s_scene_frame = s_frame_serial;
    s_pass_scene = t.scene;
    s_pass_prepass = t.prepass;
    if (t.prepass) s_prepass_frame = s_frame_serial;
    s_pass_shadow_map = t.shadow_map;
    if (s_last.color != t.color) s_last.draws = 0; // a clear restarts the pass on the same target; keep counting
    s_last.color = t.color;
    s_last.ms_color = t.ms_color;
    s_last.depth = t.depth;
    s_last.scene = t.scene;
    s_last.samples = t.samples;
}

// WBM_CALLSITES=1: where in the game's own code the work comes from. At chosen
// moments (a pass begun on some kind of target, a draw, a matrix handed to a
// shader, a buffer filled, the frame shown) the addresses the call came through
// are noted, those inside the game's executable only, as the addresses that
// file has on disk (what a disassembler shows), innermost first. The summary
// lists each distinct path with how often it was taken. This is the way into a
// binary that carries no names: whoever hands over matWorldArray draws what has
// bones; whoever begins the pass on the shadow map draws the shadows. See
// docs/engine-notes.md.
bool s_callsites;
struct CallSite {
    uint64_t count = 0, first_frame = 0;
};
std::map<std::string, CallSite> s_callsite_counts;

void note_callsite(const char *what, const char *detail = "")
{
    if (!s_callsites) return;
    static const void *game = _dyld_get_image_header(0);
    static const intptr_t slide = _dyld_get_image_vmaddr_slide(0);
    void *stack[24];
    int depth = backtrace(stack, 24), kept = 0;
    char line[512];
    int at = snprintf(line, sizeof line, "%s%s%s:", what, *detail ? " " : "", detail);
    for (int i = 1; i < depth && kept < 7 && at < (int)sizeof line - 20; i++) {
        Dl_info info;
        if (!dladdr(stack[i], &info) || info.dli_fbase != game) continue;
        at += snprintf(line + at, sizeof line - (size_t)at, " %llx", (unsigned long long)((intptr_t)stack[i] - slide));
        kept++;
    }
    if (!kept || s_callsite_counts.size() > 20000) return;
    CallSite &c = s_callsite_counts[line];
    if (!c.count++) c.first_frame = s_frame_serial;
}

const char *pass_kind()
{
    return s_pass_screen ? "the screen" : s_pass_scene ? "the scene" : s_pass_prepass ? "the depth pre-pass" : s_pass_shadow_map ? "the shadow map" : "another target";
}

bool ensure_pass()
{
    Target t;
    if (!resolve_target(t)) return false;
    if (!s_pass || s_pass_color != t.color || s_pass_depth != t.depth) {
        start_pass(t, 0);
        if (s_callsites) note_callsite("pass begun on", pass_kind());
    }
    return true;
}

// ---- uniform storage -----------------------------------------------------------------

// Copies bytes into this frame's uniform storage and returns where they went.
id<MTLBuffer> push_uniforms(const std::vector<uint8_t> &data, NSUInteger *offset)
{
    size_t need = (data.size() + 255) & ~(size_t)255;
    if (!s_frame_chunks) s_frame_chunks = [NSMutableArray array];
    if (s_frame_chunks.count == 0 || s_chunk_used + need > UNIFORM_CHUNK) {
        id<MTLBuffer> chunk;
        {
            std::lock_guard<std::mutex> lock(s_chunk_mu);
            if (!s_free_chunks.empty()) {
                chunk = s_free_chunks.back();
                s_free_chunks.pop_back();
            }
        }
        if (!chunk) chunk = [s_device newBufferWithLength:UNIFORM_CHUNK options:MTLResourceStorageModeShared];
        [s_frame_chunks addObject:chunk];
        s_chunk_used = 0;
    }
    id<MTLBuffer> chunk = s_frame_chunks.lastObject;
    memcpy((uint8_t *)chunk.contents + s_chunk_used, data.data(), data.size());
    *offset = s_chunk_used;
    s_chunk_used += need;
    return chunk;
}
