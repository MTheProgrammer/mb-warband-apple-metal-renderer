// ---- ambient occlusion ---------------------------------------------------------------

// Three full-screen passes written directly in Metal's shading language.
//
// ao_linearize, at half resolution: turn the depth buffer into distances. The
// depth buffer is multisampled and at Retina size; reading it dozens of times
// per pixel is what made the first version of this halve the frame rate.
//
// ao_compute, at half resolution: for each pixel, rebuild its position in view
// space, estimate the surface normal from neighbouring distances, and look at
// 16 nearby points on a rotated spiral. Points that rise
// above the surface's tangent plane within the radius occlude it; the closer,
// the more. This is the "Alchemy" estimator, a horizon-based method that needs
// only depth.
//
// ao_blur, at half resolution: blur the result over 3x3 texels, ignoring
// texels at a different depth so that occlusion does not leak across
// silhouettes.
//
// ao_apply, at full resolution: multiply the scene's colour by it.
const char k_ao_source[] = R"(
#include <metal_stdlib>
using namespace metal;

struct Params {
    float a, b;        // clip z = a * w + b, from the camera's projection
    float p00, p11;    // projection scale in x and y
    float2 depth_size; // of the depth buffer, in pixels
    float2 ao_size;    // of the half-resolution textures
    float radius, strength, reversed, fade_start;
    float4 sun;        // xyz: direction to the sun in view space; w: distance within which contact shadows are left out, or 0
    float shadow_strength, shadow_length;
    float2 shaft_size; // of the light-shaft textures, in pixels
    float4 shaft;      // xy: the sun's position on screen (0..1), z: how visible it is, w: strength
    float4 bounce;     // x: strength, y: radius in metres, z: how many texels of the half-resolution textures one of its own covers
};

#if WBM_MSAA
typedef depth2d_ms<float> DepthTexture;
#else
typedef depth2d<float> DepthTexture;
#endif

// Distance in front of the camera for a pixel of the depth buffer; a very
// large value for the sky.
//
// With multisampling a pixel holds several depths. Along cut-out edges (hair,
// leaves, grass) they differ from sample to sample, and taking any single one
// makes thin shapes flicker in and out of the occlusion. The farthest is used:
// a pixel only counts as covered by a thin shape once all of it is.
static float view_z(DepthTexture t, int2 c, Params p)
{
    c = clamp(c, int2(0), int2(p.depth_size) - 1);
#if WBM_MSAA
    float d = t.read(uint2(c), 0);
    for (uint i = 1; i < t.get_num_samples(); i++) {
        float e = t.read(uint2(c), i);
        d = p.reversed > 0.5 ? min(d, e) : max(d, e);
    }
#else
    float d = t.read(uint2(c), 0);
#endif
    bool sky = p.reversed > 0.5 ? d <= 0.0 : d >= 1.0;
    if (sky) return 1.0e6;
    float ndc = p.reversed > 0.5 ? 1.0 - 2.0 * d : 2.0 * d - 1.0;
    return p.b / (ndc - p.a);
}

struct VertexOut {
    float4 position [[position]];
};

vertex VertexOut ao_vertex(uint id [[vertex_id]])
{
    float2 corner = float2((id << 1) & 2, id & 2); // one triangle that covers the screen
    VertexOut out;
    out.position = float4(corner * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

// Pass 1, at the occlusion resolution (half the depth buffer's, or all of it
// with WBM_AO_FULL): distances, so that the passes after it read a small
// single-sample texture instead of the multisampled depth buffer.
fragment float4 ao_linearize(VertexOut in [[stage_in]], DepthTexture depth [[texture(0)]], constant Params &p [[buffer(0)]])
{
    return float4(view_z(depth, int2(in.position.xy * p.depth_size / p.ao_size), p));
}

static float load_z(texture2d<float> z, int2 c, Params p)
{
    return z.read(uint2(clamp(c, int2(0), int2(p.ao_size) - 1))).x;
}

static float3 view_position(texture2d<float> z, int2 c, Params p)
{
    float w = load_z(z, c, p);
    float2 ndc = (float2(c) + 0.5) / p.ao_size * 2.0 - 1.0;
    return float3(ndc.x * w / p.p00, ndc.y * w / p.p11, w);
}

// Pass 2, half resolution: occlusion.
fragment float4 ao_compute(VertexOut in [[stage_in]], texture2d<float> z [[texture(0)]], constant Params &p [[buffer(0)]])
{
    int2 c = int2(in.position.xy);
    float3 P = view_position(z, c, p);
    if (P.z > p.fade_start * 2.0) return float4(1.0); // far away, or sky

    // The normal, from whichever neighbour on each axis is closer in depth, so
    // that it is not taken across an edge.
    float3 right = view_position(z, c + int2(1, 0), p) - P, left = P - view_position(z, c - int2(1, 0), p);
    float3 down = view_position(z, c + int2(0, 1), p) - P, up = P - view_position(z, c - int2(0, 1), p);
    float3 dx = abs(right.z) < abs(left.z) ? right : left;
    float3 dy = abs(down.z) < abs(up.z) ? down : up;
    float3 n = normalize(cross(dy, dx));
    if (dot(n, P) > 0.0) n = -n; // towards the camera

    float wanted_px = p.radius * abs(p.p00) / P.z * 0.5 * p.ao_size.x;
    float radius_px = clamp(wanted_px, 2.0, 96.0 * p.ao_size.x / p.depth_size.x);
    // Close to the camera the radius does not fit on screen and is cut short.
    // The distance it then really covers is what the weights below go by, and
    // the bias with it; kept at the full radius, a face seen close up had its
    // eye sockets and the sides of its nose blackened as if they were pits.
    float radius = p.radius * min(1.0, radius_px / max(wanted_px, 0.001));
    float noise = fract(52.9829189 * fract(dot(in.position.xy, float2(0.06711056, 0.00583715))));
    float bias = (0.02 + 0.004 * P.z) * radius / p.radius;
    float sum = 0.0;
    const int directions = 4, steps = 4;
    for (int i = 0; i < directions; i++) {
        float angle = (float(i) + noise) * (6.2831853 / float(directions));
        float2 dir = float2(cos(angle), sin(angle));
        for (int s = 0; s < steps; s++) {
            float t = (float(s) + fract(noise * 7.0 + 0.37 * float(i)) + 0.2) / float(steps);
            float3 v = view_position(z, c + int2(dir * radius_px * t), p) - P;
            float dd = dot(v, v);
            sum += max(0.0, dot(v, n) - bias) / (dd + 0.0001 * radius * radius) * saturate(1.0 - dd / (radius * radius));
        }
    }
    float occlusion = sum * radius * 2.0 / float(directions * steps);
    float ao = saturate(1.0 - p.strength * occlusion);

    // Contact shadows: walk a short way from the surface towards the sun and see
    // whether something on screen is in the way. This catches what the shadow
    // map is too coarse for: feet on the ground, gear against a body, grass.
    // Only surfaces facing the sun are tested, since the others are unlit
    // already, and a hit counts only if the blocker is thin enough to plausibly
    // be in front of the ray instead of merely nearer to the camera.
    if (p.shadow_strength > 0.0) {
        float3 L = normalize(p.sun.xyz);
        float facing = dot(n, L);
        if (facing > 0.05) {
            float3 origin = P + n * (0.015 + 0.004 * P.z);
            float hit = 0.0;
            const int shadow_steps = 10;
            for (int s = 0; s < shadow_steps; s++) {
                float t = (float(s) + noise) / float(shadow_steps) * p.shadow_length;
                float3 S = origin + L * t;
                if (S.z <= 0.05) break;
                float2 ndc = float2(S.x * p.p00, S.y * p.p11) / S.z;
                int2 sc = int2((ndc * 0.5 + 0.5) * p.ao_size);
                if (any(sc < int2(0)) || any(sc >= int2(p.ao_size))) break;
                float behind = S.z - z.read(uint2(sc)).x;
                if (behind > 0.02 + 0.003 * S.z && behind < 0.25 + 0.01 * S.z) {
                    hit = 1.0 - 0.5 * t / p.shadow_length;
                    break;
                }
            }
            // Where the per-frame shadow map's fine window reaches, things already
            // cast exact shadows; this estimate next to them reads as a second one.
            float beyond = p.sun.w > 0.0 ? smoothstep(p.sun.w * 0.6, p.sun.w, P.z) : 1.0;
            ao *= 1.0 - p.shadow_strength * hit * smoothstep(0.05, 0.3, facing) * beyond;
        }
    }
    return float4(mix(ao, 1.0, smoothstep(p.fade_start, p.fade_start * 2.0, P.z)));
}

// Pass 3, half resolution: blur over 3x3 texels, skipping those at a different
// distance so that occlusion does not leak across silhouettes.
fragment float4 ao_blur(VertexOut in [[stage_in]], texture2d<float> ao [[texture(0)]], texture2d<float> z [[texture(1)]],
                        constant Params &p [[buffer(0)]])
{
    int2 c = int2(in.position.xy);
    float zc = load_z(z, c, p);
    float total = 0.0, weight = 0.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            int2 t = clamp(c + int2(i, j), int2(0), int2(p.ao_size) - 1);
            float w = exp(-abs(load_z(z, t, p) - zc) / (0.03 * zc + 0.02)) / (1.0 + abs(float(i)) + abs(float(j)));
            total += ao.read(uint2(t)).x * w;
            weight += w;
        }
    }
    return float4(weight > 0.0001 ? total / weight : 1.0);
}

// Pass 4, full resolution, blended onto the scene: one filtered sample. All the
// per-pixel work was done at half resolution; this pass touches four times as
// many pixels and has to stay cheap.
fragment float4 ao_apply(VertexOut in [[stage_in]], texture2d<float> ao [[texture(0)]], constant Params &p [[buffer(0)]])
{
    constexpr sampler smooth(filter::linear, address::clamp_to_edge);
    float a = ao.sample(smooth, in.position.xy / p.depth_size).x;
    return float4(a, a, a, 1.0);
}

// Light shafts ("god rays"). Sky near the sun is bright; anything in front of
// it is not. Smearing that bright-sky mask along lines towards the sun's
// position on screen makes streaks of light pass the edges of whatever blocks
// it, which is how shafts through trees and past towers look.
//
// shaft_mask, quarter resolution: the scene's colour where the pixel is sky,
// weighted towards the sun. shaft_blur: 40 steps from each pixel towards the
// sun, fading with distance, jittered per pixel. shaft_apply: added to the scene.
fragment float4 shaft_mask(VertexOut in [[stage_in]], texture2d<float> scene [[texture(0)]], texture2d<float> z [[texture(1)]],
                           constant Params &p [[buffer(0)]])
{
    constexpr sampler smooth(filter::linear, address::clamp_to_edge);
    float2 uv = in.position.xy / p.shaft_size;
    float sky = z.sample(smooth, uv).x > 1.0e5 ? 1.0 : 0.0;
    float2 ndc = uv * 2.0 - 1.0;
    float3 dir = normalize(float3(ndc.x / p.p00, ndc.y / p.p11, 1.0));
    float near_sun = pow(saturate(dot(dir, normalize(p.sun.xyz))), 24.0);
    return float4(scene.sample(smooth, uv).rgb * sky * (0.1 + 0.9 * near_sun), 1.0);
}

fragment float4 shaft_blur(VertexOut in [[stage_in]], texture2d<float> mask [[texture(0)]], constant Params &p [[buffer(0)]])
{
    constexpr sampler smooth(filter::linear, address::clamp_to_edge);
    const int samples = 40;
    float2 uv = in.position.xy / p.shaft_size;
    float2 step = (p.shaft.xy - uv) * (0.85 / float(samples));
    float noise = fract(52.9829189 * fract(dot(in.position.xy, float2(0.06711056, 0.00583715))));
    float2 q = uv + step * noise;
    float3 sum = float3(0.0);
    float weight = 1.0;
    for (int i = 0; i < samples; i++) {
        sum += mask.sample(smooth, q).rgb * weight;
        weight *= 0.965;
        q += step;
    }
    return float4(sum * (p.shaft.z * p.shaft.w / float(samples)), 1.0);
}

fragment float4 shaft_apply(VertexOut in [[stage_in]], texture2d<float> shafts [[texture(0)]], constant Params &p [[buffer(0)]])
{
    constexpr sampler smooth(filter::linear, address::clamp_to_edge);
    return float4(shafts.sample(smooth, in.position.xy / p.depth_size).rgb, 0.0);
}

// Bounce light: what is lit throws some of its light, in its own colour, on
// what is near it; a red cloak warms the wall beside it, sunlit ground lights
// the underside of a shield. From the picture itself: for each pixel, a ring
// of others within a metre or two is asked how bright they are and whether
// they lie on the side its surface faces, the nearer the more.
//
// bounce_compute, quarter resolution (the light is soft, and at half this cost
// more than twice as much): the light arriving, linear. bounce_blur:
// smoothed over 5x5 texels of like depth. bounce_apply: the scene's colour is
// multiplied by one plus it, which takes the surface's own colour for what it
// reflects.
fragment float4 bounce_compute(VertexOut in [[stage_in]], texture2d<float> z [[texture(0)]], texture2d<float> scene [[texture(1)]], constant Params &p [[buffer(0)]])
{
    constexpr sampler smooth(filter::linear, address::clamp_to_edge);
    int2 c = int2(in.position.xy * p.bounce.z);
    float3 P = view_position(z, c, p);
    if (P.z > p.fade_start * 2.0) return float4(0.0);
    float3 right = view_position(z, c + int2(1, 0), p) - P, left = P - view_position(z, c - int2(1, 0), p);
    float3 down = view_position(z, c + int2(0, 1), p) - P, up = P - view_position(z, c - int2(0, 1), p);
    float3 dx = abs(right.z) < abs(left.z) ? right : left;
    float3 dy = abs(down.z) < abs(up.z) ? down : up;
    float3 n = normalize(cross(dy, dx));
    if (dot(n, P) > 0.0) n = -n;
    float radius = p.bounce.y;
    float radius_px = clamp(radius * abs(p.p00) / P.z * 0.5 * p.ao_size.x, 4.0, 0.2 * p.ao_size.x);
    float noise = fract(52.9829189 * fract(dot(in.position.xy, float2(0.06711056, 0.00583715))));
    float3 sum = float3(0.0);
    const int directions = 6, steps = 3;
    for (int i = 0; i < directions; i++) {
        float angle = (float(i) + noise) * (6.2831853 / float(directions));
        float2 dir = float2(cos(angle), sin(angle));
        for (int s = 0; s < steps; s++) {
            float t = (float(s) + fract(noise * 7.0 + 0.37 * float(i)) + 0.3) / float(steps);
            int2 q = c + int2(dir * radius_px * t);
            if (any(q < int2(0)) || any(q >= int2(p.ao_size))) continue;
            float3 Q = view_position(z, q, p);
            if (Q.z > 1.0e5) continue; // the sky lights by other means
            float3 v = Q - P;
            float dd = dot(v, v);
            float facing = max(dot(v, n) * rsqrt(dd + 1e-6) - 0.08, 0.0);          // it lies on the side this surface faces
            float reach = saturate(1.0 - dd / (radius * radius * 4.0));             // and not too far off
            float3 colour = scene.sample(smooth, (float2(q) + 0.5) / p.ao_size, level(1.0)).rgb;
            colour = pow(max(colour, 0.0), float3(2.2));                            // the scene holds colours as shown
            sum += min(colour, float3(4.0)) * (facing * reach);
        }
    }
    float fade = 1.0 - smoothstep(p.fade_start, p.fade_start * 2.0, P.z);
    return float4(sum * (fade / float(directions * steps)), 1.0);
}

fragment float4 bounce_blur(VertexOut in [[stage_in]], texture2d<float> light [[texture(0)]], texture2d<float> z [[texture(1)]], constant Params &p [[buffer(0)]])
{
    int2 c = int2(in.position.xy);
    int2 size = int2(light.get_width(), light.get_height());
    float zc = load_z(z, int2(in.position.xy * p.bounce.z), p);
    float3 total = float3(0.0);
    float weight = 0.0;
    int stride = p.bounce.z > 1.5 ? 1 : 2; // the same reach on screen either way
    for (int j = -2; j <= 2; j++) {
        for (int i = -2; i <= 2; i++) {
            int2 t = clamp(c + int2(i, j) * stride, int2(0), size - 1);
            float w = exp(-abs(load_z(z, int2((float2(t) + 0.5) * p.bounce.z), p) - zc) / (0.03 * zc + 0.02));
            total += light.read(uint2(t)).rgb * w;
            weight += w;
        }
    }
    return float4(weight > 0.0001 ? total / weight : float3(0.0), 1.0);
}

fragment float4 bounce_apply(VertexOut in [[stage_in]], texture2d<float> light [[texture(0)]], constant Params &p [[buffer(0)]])
{
    constexpr sampler smooth(filter::linear, address::clamp_to_edge);
    float3 l = light.sample(smooth, in.position.xy / p.depth_size).rgb * p.bounce.x;
    return float4(pow(1.0 + l, float3(1.0 / 2.2)) - 1.0, 0.0); // the scene is multiplied by one plus this, in its own encoding
}
)";

struct AOParams { // must match Params in k_ao_source
    float a, b, p00, p11;
    float depth_size[2], ao_size[2];
    float radius, strength, reversed, fade_start;
    float sun[4];
    float shadow_strength, shadow_length, shaft_size[2];
    float shaft[4];
    float bounce[4];
};

id<MTLLibrary> s_ao_library;
id<MTLRenderPipelineState> s_ao_linearize_pipeline, s_ao_compute_pipeline, s_ao_blur_pipeline;
id<MTLTexture> s_ao_z_texture, s_ao_blurred_texture;
id<MTLRenderPipelineState> s_shaft_mask_pipeline, s_shaft_blur_pipeline, s_shaft_apply_pipeline;
id<MTLRenderPipelineState> s_bounce_compute_pipeline, s_bounce_blur_pipeline;
std::map<int, id<MTLRenderPipelineState>> s_bounce_apply_pipelines; // by colour format
id<MTLTexture> s_bounce_texture, s_bounce_blurred_texture;
uint64_t s_bounce_frames;
id<MTLTexture> s_shaft_mask_texture, s_shaft_texture;
std::map<std::pair<int, int>, id<MTLRenderPipelineState>> s_ao_apply_pipelines; // by colour format and sample count
id<MTLTexture> s_ao_texture;

bool ao_setup()
{
    if (s_ao_library) return true;
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.preprocessorMacros = @{@"WBM_MSAA" : @(s_msaa > 1 ? 1 : 0)};
    NSError *error = nil;
    s_ao_library = [s_device newLibraryWithSource:@(k_ao_source) options:options error:&error];
    if (!s_ao_library) {
        complain("ambient occlusion shaders did not compile: %s", error.localizedDescription.UTF8String);
        return s_ao = s_contact_shadows = s_light_shafts = false;
    }
    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [s_ao_library newFunctionWithName:@"ao_vertex"];
    d.fragmentFunction = [s_ao_library newFunctionWithName:@"ao_linearize"];
    d.colorAttachments[0].pixelFormat = MTLPixelFormatR32Float;
    s_ao_linearize_pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    d.fragmentFunction = [s_ao_library newFunctionWithName:@"ao_compute"];
    d.colorAttachments[0].pixelFormat = MTLPixelFormatR8Unorm;
    s_ao_compute_pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    d.fragmentFunction = [s_ao_library newFunctionWithName:@"ao_blur"];
    s_ao_blur_pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    d.fragmentFunction = [s_ao_library newFunctionWithName:@"shaft_mask"];
    d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
    s_shaft_mask_pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    d.fragmentFunction = [s_ao_library newFunctionWithName:@"shaft_blur"];
    s_shaft_blur_pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    d.fragmentFunction = [s_ao_library newFunctionWithName:@"shaft_apply"];
    d.colorAttachments[0].blendingEnabled = YES; // added to the scene's colour; alpha untouched
    d.colorAttachments[0].sourceRGBBlendFactor = d.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
    d.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorZero;
    d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
    s_shaft_apply_pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    d.colorAttachments[0].blendingEnabled = NO;
    d.fragmentFunction = [s_ao_library newFunctionWithName:@"bounce_compute"];
    s_bounce_compute_pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    d.fragmentFunction = [s_ao_library newFunctionWithName:@"bounce_blur"];
    s_bounce_blur_pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    if (!s_bounce_compute_pipeline || !s_bounce_blur_pipeline) {
        complain("bounce light pipeline failed: %s", error.localizedDescription.UTF8String);
        s_bounce_light = 0;
    }
    if (!s_ao_linearize_pipeline || !s_ao_compute_pipeline || !s_ao_blur_pipeline || !s_shaft_mask_pipeline ||
        !s_shaft_blur_pipeline || !s_shaft_apply_pipeline) {
        complain("ambient occlusion pipeline failed: %s", error.localizedDescription.UTF8String);
        return s_ao = s_contact_shadows = s_light_shafts = false;
    }
    return true;
}

id<MTLRenderPipelineState> ao_apply_pipeline(MTLPixelFormat format, int samples)
{
    id<MTLRenderPipelineState> __strong &state = s_ao_apply_pipelines[{(int)format, samples}];
    if (!state) {
        MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = [s_ao_library newFunctionWithName:@"ao_vertex"];
        d.fragmentFunction = [s_ao_library newFunctionWithName:@"ao_apply"];
        d.rasterSampleCount = (NSUInteger)samples;
        MTLRenderPipelineColorAttachmentDescriptor *c = d.colorAttachments[0];
        c.pixelFormat = format;
        c.blendingEnabled = YES; // colour = colour already there * occlusion; alpha untouched
        c.sourceRGBBlendFactor = MTLBlendFactorDestinationColor;
        c.destinationRGBBlendFactor = MTLBlendFactorZero;
        c.sourceAlphaBlendFactor = MTLBlendFactorZero;
        c.destinationAlphaBlendFactor = MTLBlendFactorOne;
        NSError *error = nil;
        state = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
        if (!state) complain("ambient occlusion apply pipeline failed: %s", error.localizedDescription.UTF8String);
    }
    return state;
}

// Runs the two passes on the scene buffer described by s_last. Called between
// render passes, with none open.
void apply_ambient_occlusion()
{
    // Without this frame's camera there is nothing sound to compute with; a
    // camera left over from another scene is worse than no effect.
    if (s_camera.projection_frame != s_frame_serial || !s_last.depth || !ao_setup()) return;
    AOParams params;
    params.a = (float)s_camera.a;
    params.b = (float)s_camera.b;
    params.p00 = (float)s_camera.p00;
    params.p11 = (float)s_camera.p11;
    // Distances are in the scene's own units, and scenes differ in scale: a
    // battlefield is life-size with a far plane around 1250, the world map a
    // miniature with one around 300. The radius, fade distance and shadow
    // length are scaled by the far plane so that they stay in proportion.
    double far = fabs(1.0 - s_camera.a) > 1e-9 ? fabs(s_camera.b / (1.0 - s_camera.a)) : 1250.0;
    float scale = (float)std::min(1.0, std::max(0.15, far / 1250.0));
    NSUInteger w = s_last.depth.width, h = s_last.depth.height;
    NSUInteger aw = s_ao_full ? w : (w + 1) / 2, ah = s_ao_full ? h : (h + 1) / 2;
    params.depth_size[0] = (float)w, params.depth_size[1] = (float)h;
    params.ao_size[0] = (float)aw, params.ao_size[1] = (float)ah;
    params.radius = s_ao_radius * scale;
    params.strength = s_ao ? s_ao_strength : 0.0f;
    // vSunDir points the way the light travels, so the direction to the sun is
    // its negative. No sun (night, interiors) means no contact shadows or shafts.
    memset(params.sun, 0, sizeof params.sun);
    memset(params.shaft, 0, sizeof params.shaft);
    params.shadow_strength = 0;
    params.shadow_length = s_contact_length * scale;
    NSUInteger sw = std::max<NSUInteger>(1, w / 4), sh = std::max<NSUInteger>(1, h / 4);
    params.shaft_size[0] = (float)sw, params.shaft_size[1] = (float)sh;
    float sun_brightness = s_sun_color[0] + s_sun_color[1] + s_sun_color[2];
    bool has_sun = s_sun_valid && sun_brightness > 0.05f && s_camera.view_frame == s_frame_serial;
    bool shafts = false;
    if (has_sun) {
        for (int r = 0; r < 3; r++)
            params.sun[r] = (float)-(s_camera.view[r][0] * s_sun_dir[0] + s_camera.view[r][1] * s_sun_dir[1] + s_camera.view[r][2] * s_sun_dir[2]);
        if (s_contact_shadows) params.shadow_strength = s_contact_strength;
        if (s_dyn_shadows && s_dyn_near[2] > 0) params.sun[3] = s_dyn_near_range * 1.5f; // as far ahead as the fine window reaches
        // (Not around a previewed model on an empty stage: with nothing but sky, the whole picture would be shaft.)
        if (s_light_shafts && params.sun[2] > 0.05f && !(s_previewing && !s_preview_keeps_scene)) { // the sun is in front of the camera
            float nx = params.sun[0] * params.p00 / params.sun[2], ny = params.sun[1] * params.p11 / params.sun[2];
            float off_screen = std::max(fabsf(nx), fabsf(ny));
            float visible = 1.0f - std::min(1.0f, std::max(0.0f, (off_screen - 1.0f) / 0.8f)); // fades out as the sun leaves the view
            params.shaft[0] = nx * 0.5f + 0.5f;
            params.shaft[1] = ny * 0.5f + 0.5f;
            params.shaft[2] = visible;
            params.shaft[3] = s_shaft_strength;
            shafts = visible > 0.01f;
        }
    }
    bool occlude = s_ao || params.shadow_strength > 0;
    bool bounce = s_bounce_light > 0 && s_bounce_compute_pipeline && s_bounce_blur_pipeline;
    NSUInteger bk = s_bounce_half ? 1 : 2, bw = (aw + bk - 1) / bk, bh = (ah + bk - 1) / bk;
    params.bounce[0] = s_bounce_light * 2.5f, params.bounce[1] = 1.6f * scale, params.bounce[2] = (float)bk, params.bounce[3] = 0;
    if (!occlude && !shafts && !bounce) return;
    params.reversed = s_reverse_depth ? 1.0f : 0.0f;
    params.fade_start = 60.0f * scale; // beyond this the effect fades out, and is gone at twice it

    if (!s_ao_texture || s_ao_texture.width != aw || s_ao_texture.height != ah) {
        auto make = [&](MTLPixelFormat format) {
            MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:aw height:ah mipmapped:NO];
            td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            td.storageMode = MTLStorageModePrivate;
            return [s_device newTextureWithDescriptor:td];
        };
        s_ao_texture = make(MTLPixelFormatR8Unorm);
        s_ao_blurred_texture = make(MTLPixelFormatR8Unorm);
        s_ao_z_texture = make(MTLPixelFormatR32Float);
    }
    if (!s_shaft_texture || s_shaft_texture.width != sw || s_shaft_texture.height != sh) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:sw height:sh mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        s_shaft_mask_texture = [s_device newTextureWithDescriptor:td];
        s_shaft_texture = [s_device newTextureWithDescriptor:td];
    }
    id<MTLRenderPipelineState> apply = ao_apply_pipeline(s_last.color.pixelFormat, 1);
    if (!apply) return;

    auto half_pass = [&](id<MTLTexture> target, id<MTLRenderPipelineState> pipeline, id<MTLTexture> input0, id<MTLTexture> input1) {
        MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
        d.colorAttachments[0].texture = target;
        d.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        d.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> pass = [command_buffer() renderCommandEncoderWithDescriptor:d];
        [pass setRenderPipelineState:pipeline];
        [pass setFragmentTexture:input0 atIndex:0];
        if (input1) [pass setFragmentTexture:input1 atIndex:1];
        [pass setFragmentBytes:&params length:sizeof params atIndex:0];
        [pass drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [pass endEncoding];
    };
    // Both `onto` passes blend onto the resolved image, which is what
    // post-processing reads. The multisampled copy is left as it was: writing
    // there would blend every sample and resolve the whole buffer again, and the
    // engine does not draw into the scene once post-processing has begun.
    auto onto_scene = [&](id<MTLRenderPipelineState> pipeline, id<MTLTexture> input) {
        MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
        d.colorAttachments[0].texture = s_last.color;
        d.colorAttachments[0].loadAction = MTLLoadActionLoad;
        d.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> pass = [command_buffer() renderCommandEncoderWithDescriptor:d];
        [pass setRenderPipelineState:pipeline];
        [pass setFragmentTexture:input atIndex:0];
        [pass setFragmentBytes:&params length:sizeof params atIndex:0];
        [pass drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [pass endEncoding];
    };
    half_pass(s_ao_z_texture, s_ao_linearize_pipeline, s_last.depth, nil);
    if (bounce) { // before occlusion, which then darkens this light with the rest
        id<MTLRenderPipelineState> __strong &add = s_bounce_apply_pipelines[(int)s_last.color.pixelFormat];
        if (!add) {
            MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
            pd.vertexFunction = [s_ao_library newFunctionWithName:@"ao_vertex"];
            pd.fragmentFunction = [s_ao_library newFunctionWithName:@"bounce_apply"];
            MTLRenderPipelineColorAttachmentDescriptor *c = pd.colorAttachments[0];
            c.pixelFormat = s_last.color.pixelFormat;
            c.blendingEnabled = YES; // colour = colour already there * (1 + light); alpha untouched
            c.sourceRGBBlendFactor = MTLBlendFactorDestinationColor;
            c.destinationRGBBlendFactor = MTLBlendFactorOne;
            c.sourceAlphaBlendFactor = MTLBlendFactorZero;
            c.destinationAlphaBlendFactor = MTLBlendFactorOne;
            NSError *error = nil;
            add = [s_device newRenderPipelineStateWithDescriptor:pd error:&error];
            if (!add) complain("bounce light apply pipeline failed: %s", error.localizedDescription.UTF8String);
        }
        if (!s_bounce_texture || s_bounce_texture.width != bw || s_bounce_texture.height != bh) {
            MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:bw height:bh mipmapped:NO];
            td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            td.storageMode = MTLStorageModePrivate;
            s_bounce_texture = [s_device newTextureWithDescriptor:td];
            s_bounce_blurred_texture = [s_device newTextureWithDescriptor:td];
        }
        if (add) {
            half_pass(s_bounce_texture, s_bounce_compute_pipeline, s_ao_z_texture, s_last.color);
            half_pass(s_bounce_blurred_texture, s_bounce_blur_pipeline, s_bounce_texture, s_ao_z_texture);
            onto_scene(add, s_bounce_blurred_texture);
            s_bounce_frames++;
        }
    }
    if (occlude) {
        half_pass(s_ao_texture, s_ao_compute_pipeline, s_ao_z_texture, nil);
        half_pass(s_ao_blurred_texture, s_ao_blur_pipeline, s_ao_texture, s_ao_z_texture);
        onto_scene(apply, s_ao_blurred_texture);
    }
    if (shafts) { // after occlusion, so that the light is not darkened by it
        half_pass(s_shaft_mask_texture, s_shaft_mask_pipeline, s_last.color, s_ao_z_texture);
        half_pass(s_shaft_texture, s_shaft_blur_pipeline, s_shaft_mask_texture, nil);
        onto_scene(s_shaft_apply_pipeline, s_shaft_texture);
    }
}
