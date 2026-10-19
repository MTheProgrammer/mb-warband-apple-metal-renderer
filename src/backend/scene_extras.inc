// Foliage that the engine hands over in batches, many plants in one buffer and
// already placed in the world, carries nothing that says how high above its
// own foot a vertex is, which is what the wind needs. It is worked out here,
// once per buffer: a batch repeats one mesh, so the texture coordinates repeat
// with the mesh's vertex count, which gives the plants; a vertex's height above
// the lowest point of its plant, in decimetres, is then written into the alpha
// of its colour, which foliage leaves at full. The vertex shaders read it from
// there and take the alpha as full (see k_wind_function). For leaves that hang
// from a crown (palms) the lowest point is a leaf's tip, so there the distance
// from the plant's middle is used instead.
bool weigh_foliage(Buffer &b, bool radial)
{
    if (b.wind_weights == (__bridge void *)b.mtl) return true;
    const Attrib &pos = s_attribs[0], &col = s_attribs[2], &tex = s_attribs[4];
    if (!b.mtl || !pos.enabled || !col.enabled || !tex.enabled || pos.stride <= 0 || col.buffer != pos.buffer || tex.buffer != pos.buffer ||
        pos.type != GL_FLOAT || col.type != GL_UNSIGNED_BYTE || tex.type != GL_FLOAT)
        return false;
    size_t stride = (size_t)pos.stride, n = b.mtl.length / stride;
    if (n < 3 || std::max({pos.offset + 12, col.offset + 4, tex.offset + 8}) > stride) return false;
    uint8_t *base = (uint8_t *)b.mtl.contents;
    auto uv = [&](size_t i) { return base + i * stride + tex.offset; };
    auto z = [&](size_t i) {
        float v;
        memcpy(&v, base + i * stride + pos.offset + 8, 4);
        return v;
    };
    // The shortest period of the texture coordinates, tried only where the first vertex's recur.
    size_t period = n;
    for (size_t c = 3; c <= n / 2; c++) {
        if (n % c || memcmp(uv(0), uv(c), 8)) continue;
        bool repeats = true;
        for (size_t i = c; i < n && repeats; i++) repeats = !memcmp(uv(i), uv(i - c), 8);
        if (repeats) {
            period = c;
            break;
        }
    }
    for (size_t first = 0; first < n; first += period) {
        float lowest = 1e30f;
        double middle[2] = {0, 0};
        for (size_t i = first; i < first + period; i++) {
            float xy[2];
            memcpy(xy, base + i * stride + pos.offset, 8);
            lowest = std::min(lowest, z(i));
            middle[0] += xy[0] / (double)period, middle[1] += xy[1] / (double)period;
        }
        for (size_t i = first; i < first + period; i++) {
            float metres = z(i) - lowest;
            if (radial) { // leaves that hang from a crown are held at the middle and free at their tips, wherever those are
                float xy[2];
                memcpy(xy, base + i * stride + pos.offset, 8);
                metres = 1.6f * (float)hypot(xy[0] - middle[0], xy[1] - middle[1]);
            }
            base[i * stride + col.offset + 3] = (uint8_t)std::min(255.0f, std::max(0.0f, metres * 10.0f + 0.5f));
        }
    }
    b.wind_weights = (__bridge void *)b.mtl;
    return true;
}

// Where a pair of eyes is turned at a time: held still for a second or two,
// about half the time straight ahead, then moved to the next place in a
// twelfth of a second, as eyes do. In the face texture's coordinates.
void eye_gaze(double time, float phase, float *out)
{
    if (const char *fixed = getenv("WBM_GAZE")) { // "x y", for looking at the effect
        sscanf(fixed, "%f %f", &out[0], &out[1]);
        return;
    }
    auto hash = [](double n) {
        double v = sin(n) * 43758.5453;
        return (float)(v - floor(v));
    };
    auto target = [&](double step, float *xy) {
        bool ahead = hash(step * 1.7 + phase) < 0.5f;
        xy[0] = ahead ? 0.0f : (hash(step * 3.1 + phase + 4.0) - 0.5f) * 0.016f * s_gaze_strength;
        xy[1] = ahead ? 0.0f : (hash(step * 5.3 + phase + 9.0) - 0.5f) * 0.005f * s_gaze_strength;
    };
    double period = 1.1 + 1.6 * hash(phase + 2.0), at = time / period + hash(phase), step = floor(at);
    float now[2], before[2];
    target(step, now), target(step - 1, before);
    float t = (float)std::min(1.0, (at - step) * period / 0.08);
    t = t * t * (3 - 2 * t);
    out[0] = before[0] + (now[0] - before[0]) * t;
    out[1] = before[1] + (now[1] - before[1]) * t;
}

// Holds the frame rate to a limit by waiting at the end of a frame.
void limit_frame_rate()
{
    static const double start = CACurrentMediaTime();
    s_wind_time_before = s_wind_time;
    s_wind_time = fmod(CACurrentMediaTime() - start, 3600.0); // for the frame about to begin
    static std::chrono::steady_clock::time_point next;
    double limit = s_scene_frame + 1 >= s_frame_serial ? s_max_fps : (s_menu_fps > 0 ? s_menu_fps : s_max_fps); // s_frame_serial was just advanced
    auto now = std::chrono::steady_clock::now();
    if (limit <= 0) {
        next = now;
        return;
    }
    auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / limit));
    next += period;
    if (next < now - period * 4) next = now; // fell behind: do not try to catch up
    if (next > now) std::this_thread::sleep_until(next);
}
