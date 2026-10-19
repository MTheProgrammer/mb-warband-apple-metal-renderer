// ---- frame capture and playback (see frame_capture.inc) -------------------------------------------

void mtl_request_capture() { s_capture_requested = 1; }

bool mtl_replay_open(const char *path, int *width, int *height)
{
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    s_replay.file.resize((size_t)std::max(size, 0L));
    bool read = fread(s_replay.file.data(), 1, s_replay.file.size(), f) == s_replay.file.size();
    fclose(f);
    size_t at = 0;
    uint32_t kind;
    Rd r{};
    if (!read || !next_record(s_replay.file, &at, &kind, &r) || kind != CAP_HEADER || r.i() != CAP_VERSION) return false;
    s_replaying = true;
    s_replay.width = *width = (int)r.i();
    s_replay.height = *height = (int)r.i();
    s_replay.state_at = at;
    return r.ok;
}

bool mtl_replay_restore()
{
    if (!restore_state()) return false;
    find_camera();
    return true;
}

void mtl_replay_orbit(double yaw, double pitch, double pan_x, double pan_y, double forward, double pivot)
{
    const double o[6] = {yaw, pitch, pan_x, pan_y, forward, pivot};
    memcpy(s_replay.orbit, o, sizeof o);
}

void mtl_replay_show_interface(bool show) { s_replay.hide_ui = !show; }

bool mtl_preview_model(const char *name, int lod, bool keep_scene, double *distance)
{
    s_preview.keep_scene = s_preview_keeps_scene = keep_scene;
    s_previewing = false;
    if (!name || !*name) {
        s_preview.active = false;
        return true;
    }
    bool ok = preview_load(name, lod);
    s_previewing = ok;
    if (ok && distance) *distance = preview_distance();
    return ok;
}

bool mtl_preview_gallery(const char *path, bool keep_scene)
{
    s_preview.keep_scene = s_preview_keeps_scene = keep_scene;
    std::string problem;
    s_previewing = preview_load_gallery(path, &problem);
    if (!s_previewing) fprintf(stderr, "%s\n", problem.c_str());
    return s_previewing;
}

bool mtl_preview_action(const char *action, double held)
{
    s_preview.m.held = held;
    return preview_set_action(action ? action : "");
}

const char *mtl_preview_next_action(const char *action)
{
    size_t n = sizeof k_horse_actions / sizeof k_horse_actions[0];
    for (size_t i = 0; i < n; i++)
        if (action && !strcmp(action, k_horse_actions[i].name)) return i + 1 < n ? k_horse_actions[i + 1].name : "";
    return k_horse_actions[0].name;
}

void mtl_preview_material(const char *name) { s_preview.m.material = name ? name : ""; }

// The mesh a name stands for: itself if it is a mesh; the first mesh of a
// resource file of that name otherwise; null if neither.
const char *mtl_preview_resolve(const char *name)
{
    preview_read_library();
    static std::string result;
    std::string want = lower_case(name ? name : "");
    if (want.size() > 4 && !want.compare(want.size() - 4, 4, ".brf")) want.resize(want.size() - 4);
    auto &meshes = s_preview.library.meshes;
    if (meshes.count(want)) return (result = want).c_str();
    if (!s_preview.library.parts_of(want, 0).empty()) return (result = want).c_str(); // a model made of parts only (name.1, name.2 ...)
    for (size_t f = 0; f < s_preview.library.files.size(); f++) {
        const std::string &path = s_preview.library.files[f];
        std::string base = lower_case(path.substr(path.rfind('/') + 1));
        if (base != want + ".brf") continue;
        for (auto &m : meshes)
            if (m.second.file == (int)f && m.first.find(".lod") == std::string::npos) return (result = m.first).c_str();
    }
    return nullptr;
}

int mtl_preview_list(const char *pattern, FILE *out)
{
    preview_read_library();
    std::string want = lower_case(pattern ? pattern : "");
    int found = 0;
    for (auto &m : s_preview.library.meshes) {
        if (m.first.find(".lod") != std::string::npos || m.first.find(want) == std::string::npos) continue;
        const std::string &file = s_preview.library.files[(size_t)m.second.file];
        fprintf(out, "%-44s %s\n", m.first.c_str(), file.substr(file.rfind('/') + 1).c_str());
        found++;
    }
    for (auto &m : s_preview.library.materials) {
        if (m.first.find(want) == std::string::npos) continue;
        fprintf(out, "%-44s material: shader %s, textures %s %s %s %s\n", m.first.c_str(), m.second.shader.c_str(), m.second.diffuse.c_str(), m.second.diffuse2.c_str(),
                m.second.normal.c_str(), m.second.specular.c_str());
    }
    return found;
}

// The mesh `by` places after (or before) the named one in its resource file's list; lods and further parts passed over.
const char *mtl_preview_neighbour(const char *name, int by)
{
    preview_read_library();
    static std::string result;
    auto &meshes = s_preview.library.meshes;
    auto it = meshes.find(lower_case(name ? name : ""));
    if (it == meshes.end()) return nullptr;
    int file = it->second.file;
    std::vector<std::string> names;
    for (auto &m : meshes) {
        size_t dot = m.first.rfind('.');
        bool part = dot != std::string::npos && dot + 1 < m.first.size() && isdigit((unsigned char)m.first[dot + 1]);
        if (m.second.file == file && m.first.find(".lod") == std::string::npos && !part) names.push_back(m.first);
    }
    auto at = std::find(names.begin(), names.end(), it->first);
    if (at == names.end() || names.empty()) return nullptr;
    long index = ((at - names.begin()) + by) % (long)names.size();
    if (index < 0) index += (long)names.size();
    result = names[(size_t)index];
    return result.c_str();
}

void mtl_replay_frame(bool capture, uint32_t frame)
{
    replay_calls();
    mtl_present(capture, frame);
}
