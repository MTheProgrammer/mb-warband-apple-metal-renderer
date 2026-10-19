// Motion blur, and the movement of the scene for it and for temporal upscaling.
// Included by metal_backend.mm, inside its anonymous namespace.

// ---- motion blur ------------------------------------------------------------------------
//
// Blur from the camera's own movement. A pixel's depth gives its place in the
// world; the previous frame's camera says where on screen that place was a
// frame ago; the scene is smeared along the line between the two. Things that
// move by themselves are not known to move, so a sword's swing is not blurred,
// and whatever travels with the camera (the player's character, a horse) would
// be blurred as if it stood still in the world. That is avoided by distance:
// nothing closer than a few metres is blurred or smeared into what is.

const char k_blur_source[] = R"(
#include <metal_stdlib>
using namespace metal;

struct Params {
    float4x4 reproject; // this frame's clip space to the previous frame's
    float4 inverse_w;   // dot with a clip-space point at w = 1: one over its distance from the camera
    float2 size;        // of the scene, in pixels
    float reversed;
    float exposure;     // in frames
    float near_start, near_end; // distances between which the blur fades in
    float max_length;   // in screen heights
    float has_objects;  // 1 when the third texture holds this frame's moving objects (see render_object_velocities)
    float has_swayed;   // 1 when the fourth holds what the swaying shaders wrote of their own movement (see s_scene_motion)
    float spare0, spare1, spare2;
};

#if WBM_MSAA
typedef depth2d_ms<float> DepthTexture;
#else
typedef depth2d<float> DepthTexture;
#endif

struct VertexOut {
    float4 position [[position]];
};

vertex VertexOut blur_vertex(uint id [[vertex_id]])
{
    VertexOut out;
    out.position = float4(float2((id << 1) & 2, id & 2) * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

// The pixel as a point in OpenGL clip space with w = 1. Textures are stored in
// OpenGL's row order, so v grows with clip-space y.
static float4 clip_point(DepthTexture depth, float2 pixel, Params p)
{
    uint2 c = uint2(clamp(pixel, float2(0.0), p.size - 1.0));
    float d = depth.read(c, 0);
    return float4(pixel / p.size * 2.0 - 1.0, p.reversed > 0.5 ? 1.0 - 2.0 * d : 2.0 * d - 1.0, 1.0);
}

static float blurred(float4 n, Params p)
{
    float inverse = dot(p.inverse_w, n);
    float distance = inverse > 1.0e-9 ? 1.0 / inverse : 1.0e9;
    return smoothstep(p.near_start, p.near_end, distance);
}

// Things that move by themselves (a sword swung, a horse's legs) are drawn once
// more into a small texture, each pixel holding how far it moved across the
// screen since the last frame (x, y, in clip space) and how far off it is (z;
// below 0 where nothing moving was drawn). See render_object_velocities.
struct VelocityStatic {
    float4x4 now, before; // object to clip space, this frame and the last
};
struct VelocitySkinned {
    float4x4 now, before;
    float4x4 bones_now[32], bones_before[32];
};
struct VelocityStaticIn {
    float3 position [[attribute(0)]];
};
struct VelocitySkinnedIn {
    float3 position [[attribute(0)]];
    float4 weights [[attribute(7)]];
    float4 indices [[attribute(8)]];
};
struct VelocityOut {
    float4 position [[position]];
    float4 now, before;
};
static VelocityOut velocity_out(float4 now, float4 before)
{
    VelocityOut out;
    out.position = float4(now.x, -now.y, (now.z + now.w) * 0.5, now.w); // rows as the scene's textures have them
    out.now = now, out.before = before;
    return out;
}
vertex VelocityOut velocity_static(VelocityStaticIn in [[stage_in]], constant VelocityStatic &u [[buffer(0)]])
{
    float4 p = float4(in.position, 1.0);
    return velocity_out(u.now * p, u.before * p);
}
vertex VelocityOut velocity_skinned(VelocitySkinnedIn in [[stage_in]], constant VelocitySkinned &u [[buffer(0)]])
{
    float4 p = float4(in.position, 1.0);
    int4 i = int4(in.indices);
    float4 a = u.bones_now[i.x] * p * in.weights.x + u.bones_now[i.y] * p * in.weights.y + u.bones_now[i.z] * p * in.weights.z + u.bones_now[i.w] * p * in.weights.w;
    float4 b = u.bones_before[i.x] * p * in.weights.x + u.bones_before[i.y] * p * in.weights.y + u.bones_before[i.z] * p * in.weights.z + u.bones_before[i.w] * p * in.weights.w;
    return velocity_out(u.now * a, u.before * b);
}
fragment float4 velocity_fragment(VelocityOut in [[stage_in]])
{
    return float4(in.now.xy / in.now.w - in.before.xy / max(in.before.w, 1.0e-4), in.now.w, 1.0);
}

// How much of a thing's own movement is blurred. Far off, all of it. Near the
// camera, where the player's own character is, only what is fast: a blade
// swung, not the body that swings it or its legs as it runs. `extra` is the
// movement over a frame, in clip space; the speeds are in screen heights to the exposure.
static float own_share(float2 extra, float amount, Params p)
{
    float speed = length(extra * float2(p.size.x / p.size.y, 1.0)) * 0.5 * p.exposure;
    return mix(smoothstep(0.012, 0.035, speed), 1.0, amount);
}

fragment float4 blur_fragment(VertexOut in [[stage_in]], texture2d<float> scene [[texture(0)]], DepthTexture depth [[texture(1)]],
                              texture2d<float> objects [[texture(2)]], constant Params &p [[buffer(0)]])
{
    constexpr sampler smooth(filter::linear, address::clamp_to_edge);
    constexpr sampler nearest(filter::nearest, address::clamp_to_edge);
    float2 uv = in.position.xy / p.size;
    float4 centre = scene.sample(smooth, uv);
    float4 n = clip_point(depth, in.position.xy, p);
    float amount = blurred(n, p);
    float4 before = p.reproject * n;
    float2 camera = before.w > 1.0e-6 ? n.xy - before.xy / before.w : float2(0.0); // what the camera's own movement does to this pixel
    float2 moved = camera * amount;
    // A thing that moves by itself is blurred by its own movement however near it is:
    // what it does over and above the camera's. It must be the thing seen at this pixel, not one hidden behind it.
    float own = 0.0;
    if (p.has_objects > 0.5) {
        float4 o = objects.sample(nearest, uv);
        float inverse = dot(p.inverse_w, n);
        float distance = inverse > 1.0e-9 ? 1.0 / inverse : 1.0e9;
        float2 extra = o.xy - camera;
        if (o.z > 0.0 && abs(o.z - distance) < 0.06 * distance + 0.15 && length(extra * p.size) > 3.0) {
            float keep = own_share(extra, amount, p);
            moved += extra * keep, own = step(0.02, keep);
        }
        if (own <= 0.0) {
            // Beside a moving thing that is nearer: its streak reaches over this pixel, if it is moving this way fast enough.
            float most = 0.0;
            float spin = fract(52.9829189 * fract(dot(in.position.xy, float2(0.06711056, 0.00583715)))); // differently at each pixel, or the reach shows as steps
            for (int k = 0; k < 12; k++) {
                float angle = (float(k) + spin) * 0.5235988, reach = ((k & 1) == 0 ? 7.0 : 18.0) * (0.75 + 0.5 * fract(spin * 7.0)); // pixels
                float2 off = float2(cos(angle), sin(angle)) * reach;
                float4 t = objects.sample(nearest, uv + off / p.size);
                float2 theirs = (t.xy - camera) * 0.25 * p.exposure * p.size; // half the streak, in pixels
                float along = abs(dot(theirs, normalize(off)));
                if (t.z > 0.0 && t.z < distance - 0.1 && along > reach && along > most) most = along, extra = t.xy - camera;
            }
            if (most > 0.0) {
                float keep = own_share(extra, amount, p);
                moved += extra * keep, own = step(0.02, keep);
            }
        }
    }
    if (amount <= 0.0 && own <= 0.0) return centre;
    float2 velocity = moved * 0.5 * p.exposure; // in uv, per exposure
    float aspect = p.size.x / p.size.y;
    float length_ = length(velocity * float2(aspect, 1.0));
    if (length_ * p.size.y < 0.75) return centre;
    velocity *= min(1.0, p.max_length * (1.0 + own) / length_);

    const int samples = 10;
    float noise = fract(52.9829189 * fract(dot(in.position.xy, float2(0.06711056, 0.00583715))));
    float3 sum = centre.rgb;
    float weight = 1.0;
    for (int i = 0; i < samples; i++) {
        float t = (float(i) + noise) / float(samples) - 0.5;
        float2 q = uv + velocity * t;
        if (any(q < float2(0.0)) || any(q > float2(1.0))) continue;
        float w = max(blurred(clip_point(depth, q * p.size, p), p), own); // near things keep their colour to themselves
        sum += scene.sample(smooth, q).rgb * w;
        weight += w;
    }
    return float4(sum / weight, centre.a);
}

// For temporal upscaling: where each pixel was a frame ago, as a step in
// texture coordinates from where it is now. The camera's movement says it for
// everything that stands still; things that moved by themselves have it in
// `objects`, where one is seen at this pixel.
fragment float4 motion_fragment(VertexOut in [[stage_in]], DepthTexture depth [[texture(1)]], texture2d<float> objects [[texture(2)]], texture2d<float> swayed [[texture(3)]],
                                constant Params &p [[buffer(0)]])
{
    constexpr sampler nearest(filter::nearest, address::clamp_to_edge);
    float4 n = clip_point(depth, in.position.xy, p);
    float4 before = p.reproject * n;
    float2 moved = before.w > 1.0e-6 ? n.xy - before.xy / before.w : float2(0.0);
    float inverse = dot(p.inverse_w, n);
    float distance = inverse > 1.0e-9 ? 1.0 / inverse : 1.0e9;
    // What a swaying shader wrote holds only while that surface is still the one seen here.
    if (p.has_swayed > 0.5) {
        float4 o = swayed.read(uint2(in.position.xy));
        if (o.z > 0.0 && abs(o.z - distance) < 0.06 * distance + 0.15) moved = o.xy;
    }
    if (p.has_objects > 0.5) {
        float4 o = objects.sample(nearest, in.position.xy / p.size);
        if (o.z > 0.0 && abs(o.z - distance) < 0.06 * distance + 0.15) moved = o.xy;
    }
    return float4(-0.5 * moved, 0.0, 0.0); // clip space is two wide
}

// WBM_DEBUG_MOTION: the movement texture as a picture. Grey where nothing moved; redder where a pixel came
// from farther right, greener from farther down; full colour at a fiftieth of the picture's width a frame.
fragment float4 motion_show(VertexOut in [[stage_in]], texture2d<float> motion [[texture(0)]], constant Params &p [[buffer(0)]])
{
    constexpr sampler nearest(filter::nearest, address::clamp_to_edge);
    float2 m = motion.sample(nearest, in.position.xy / p.size).xy;
    return float4(0.5 + m * 25.0, 0.5, 1.0);
}
)";

struct BlurParams { // must match Params in k_blur_source
    float reproject[16];
    float inverse_w[4];
    float size[2];
    float reversed, exposure, near_start, near_end, max_length, has_objects, has_swayed, spare[3];
};
id<MTLTexture> render_object_velocities(const double *reproject, NSUInteger w, NSUInteger h);

id<MTLLibrary> s_blur_library;
std::map<int, id<MTLRenderPipelineState>> s_blur_pipelines; // by colour format
id<MTLTexture> s_blur_texture;
double s_blur_time;
uint64_t s_blur_frames;

// The shaders of motion blur and of the movement texture, compiled when first wanted.
bool blur_library()
{
    static bool failed;
    if (s_blur_library || failed) return s_blur_library != nil;
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.preprocessorMacros = @{@"WBM_MSAA" : @(s_last.depth.sampleCount > 1 ? 1 : 0)};
    NSError *error = nil;
    s_blur_library = [s_device newLibraryWithSource:@(k_blur_source) options:options error:&error];
    if (!s_blur_library) complain("motion shaders did not compile: %s", error.localizedDescription.UTF8String), failed = true;
    return s_blur_library != nil;
}

// Called between render passes, once the engine has finished the scene, after
// the other effects on it.
bool objects_recorded();
void apply_motion_blur()
{
    double now = CACurrentMediaTime(), frame_time = now - s_blur_time;
    s_blur_time = now;
    // Needs this frame's camera and the previous frame's; after a gap (a menu,
    // a loading screen) the two have nothing to do with each other.
    if (s_camera.view_proj_frame != s_frame_serial || s_camera.last_view_proj_frame + 1 != s_frame_serial || !s_last.depth || !s_last.color) return;
    if (frame_time <= 0 || frame_time > 0.1) return;
    double inverse[16], reproject[16];
    if (!mat_inverse(s_camera.view_proj, inverse)) return;
    mat_mul(s_camera.last_view_proj, inverse, reproject);
    // Nothing to do while the camera is still. And a jump (a cut to another
    // view) is no movement: the far point straight ahead would cross the screen.
    double change = 0;
    for (int i = 0; i < 16; i++) change = std::max(change, fabs(reproject[i] - (i % 5 == 0 ? reproject[15] : 0.0)));
    bool still = change < 1e-7 * fabs(reproject[15]);
    if (still && !objects_recorded()) return;
    double qw = reproject[11] + reproject[15];
    if (qw <= 1e-9) return;
    double jump = hypot((reproject[8] + reproject[12]) / qw, (reproject[9] + reproject[13]) / qw);
    if (jump > 0.5) return;

    if (!blur_library()) {
        s_motion_blur = false;
        return;
    }
    id<MTLRenderPipelineState> __strong &pipeline = s_blur_pipelines[(int)s_last.color.pixelFormat];
    if (!pipeline) {
        MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = [s_blur_library newFunctionWithName:@"blur_vertex"];
        d.fragmentFunction = [s_blur_library newFunctionWithName:@"blur_fragment"];
        d.colorAttachments[0].pixelFormat = s_last.color.pixelFormat;
        NSError *error = nil;
        pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
        if (!pipeline) {
            complain("motion blur pipeline failed: %s", error.localizedDescription.UTF8String);
            s_motion_blur = false;
            return;
        }
    }
    NSUInteger w = s_last.color.width, h = s_last.color.height;
    if (s_last.depth.width != w || s_last.depth.height != h) return;
    if (!s_blur_texture || s_blur_texture.width != w || s_blur_texture.height != h || s_blur_texture.pixelFormat != s_last.color.pixelFormat) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:s_last.color.pixelFormat width:w height:h mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        s_blur_texture = [s_device newTextureWithDescriptor:td];
    }

    BlurParams params;
    for (int i = 0; i < 16; i++) params.reproject[i] = (float)reproject[i];
    for (int c = 0; c < 4; c++) params.inverse_w[c] = (float)inverse[c * 4 + 3];
    params.size[0] = (float)w, params.size[1] = (float)h;
    params.reversed = s_reverse_depth ? 1.0f : 0.0f;
    // The exposure is a length of time, not a share of the frame, so that the
    // blur looks the same at any frame rate.
    params.exposure = (float)(s_motion_blur_strength / 48.0 / std::max(frame_time, 1.0 / 240.0));
    // Scenes differ in scale (see apply_ambient_occlusion); distances follow the far plane.
    double far_inverse = inverse[11] + inverse[15], far = fabs(far_inverse) > 1e-9 ? fabs(1.0 / far_inverse) : 1250.0;
    float scale = (float)std::min(1.0, std::max(0.15, far / 1250.0));
    params.near_start = 5.0f * scale;
    params.near_end = 10.0f * scale;
    params.max_length = 0.04f;
    id<MTLTexture> objects = render_object_velocities(reproject, w, h);
    params.has_objects = objects ? 1.0f : 0.0f;
    if (still && !objects) return;

    MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
    d.colorAttachments[0].texture = s_blur_texture;
    d.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    d.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> pass = [command_buffer() renderCommandEncoderWithDescriptor:d];
    [pass setRenderPipelineState:pipeline];
    [pass setFragmentTexture:s_last.color atIndex:0];
    [pass setFragmentTexture:s_last.depth atIndex:1];
    [pass setFragmentTexture:objects ? objects : s_white atIndex:2];
    [pass setFragmentBytes:&params length:sizeof params atIndex:0];
    [pass drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [pass endEncoding];
    id<MTLBlitCommandEncoder> blit = [command_buffer() blitCommandEncoder];
    [blit copyFromTexture:s_blur_texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1)
                toTexture:s_last.color destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    s_blur_frames++;
}
