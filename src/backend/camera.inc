// The scene camera, worked out from the draws' matrices.
// Included by metal_backend.mm, inside its anonymous namespace.

// ---- the scene camera ------------------------------------------------------------------

// Column-major 4x4 matrices, as OpenGL passes them: element (row r, column c) is m[c * 4 + r].
void mat_mul(const double *x, const double *y, double *out)
{
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++) {
            double v = 0;
            for (int k = 0; k < 4; k++) v += x[k * 4 + r] * y[c * 4 + k];
            out[c * 4 + r] = v;
        }
}

bool mat_inverse(const double *m, double *out)
{
    double inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (fabs(det) < 1e-30) return false;
    for (int i = 0; i < 16; i++) out[i] = inv[i] / det;
    return true;
}

// The current value of a mat4 uniform of the vertex stage, or false if the program has none by that name.
bool vertex_matrix(Program &p, const char *name, double *out)
{
    auto it = p.location_by_name.find(name);
    if (it == p.location_by_name.end()) return false;
    const XlUniform *u = p.locations[(size_t)it->second].uniform[0];
    if (!u || u->columns != 4 || u->components != 4) return false;
    float m[16];
    memcpy(m, p.uniform_data[0].data() + u->offset, sizeof m);
    for (int i = 0; i < 16; i++) out[i] = m[i];
    return true;
}

// Called for draws into the scene buffer until the camera is known for this
// frame. The engine gives each program combined matrices, never the projection
// on its own, and which combinations exist depends on the scene (the world map
// never sets a view-projection, for one). Two that are nearly always there:
//
//   projection = (world-view-projection) x inverse(world-view)
//   view       = (world-view) x inverse(world)
//
// Anything that does not come out as a perspective projection is ignored and
// the next draw is tried.
void observe_camera(Program &p)
{
    double wvp[16], wv[16], w[16], inv[16], m[16];
    // (Played back with the camera moved, the sky's matrices are not the scene's: see prepare_draw.)
    if ((s_motion_blur || s_temporal) && s_camera.view_proj_frame != s_frame_serial && (!s_replaying || s_depth_test) && vertex_matrix(p, "matWorldViewProj", wvp) &&
        vertex_matrix(p, "matWorld", w) && mat_inverse(w, inv)) {
        memcpy(s_camera.last_view_proj, s_camera.view_proj, sizeof s_camera.view_proj);
        s_camera.last_view_proj_frame = s_camera.view_proj_frame;
        mat_mul(wvp, inv, s_camera.view_proj);
        s_camera.view_proj_frame = s_frame_serial;
    }
    if (!vertex_matrix(p, "matWorldView", wv) || !mat_inverse(wv, inv)) return;
    if (s_camera.projection_frame != s_frame_serial && vertex_matrix(p, "matWorldViewProj", wvp)) {
        mat_mul(wvp, inv, m);
        // Rows 2 and 3 of a perspective projection are (0, 0, A, B) and (0, 0, C, 0) with C = +-1.
        double A = m[10], B = m[14], C = m[11], D = m[15];
        bool perspective = fabs(fabs(C) - 1.0) < 0.01 && fabs(D) < 0.01 && fabs(m[3]) + fabs(m[7]) < 0.01 && fabs(B) > 1e-6 &&
                           fabs(m[0]) > 1e-3 && fabs(m[5]) > 1e-3;
        if (perspective) {
            s_camera.a = A / C;
            s_camera.b = B - s_camera.a * D;
            s_camera.p00 = m[0];
            s_camera.p11 = m[5];
            s_camera.projection_frame = s_frame_serial;
            // The third view axis is the direction in which w grows.
            s_camera.view[2][0] = C;
        }
    }
    if (s_camera.view_frame != s_frame_serial && s_camera.projection_frame == s_frame_serial && vertex_matrix(p, "matWorld", w) &&
        mat_inverse(w, inv)) {
        mat_mul(wv, inv, m);
        double forward_sign = s_camera.view[2][0];
        s_camera.forward_sign = forward_sign < 0 ? -1.0 : 1.0;
        for (int r = 0; r < 3; r++) {
            double x = m[r], y = m[4 + r], z = m[8 + r], len = sqrt(x * x + y * y + z * z);
            if (len < 1e-9) return;
            double sign = r == 2 ? forward_sign : 1.0;
            s_camera.view[r][0] = sign * x / len, s_camera.view[r][1] = sign * y / len, s_camera.view[r][2] = sign * z / len;
        }
        s_camera.view_frame = s_frame_serial;
    }
}
