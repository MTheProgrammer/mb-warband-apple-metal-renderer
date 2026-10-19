// Showing one of the game's models in a played-back frame. Included by
// metal_backend.mm, inside namespace wbm, after frame_capture.inc.
//
// The captured frame is the stage: its sky, sun, fog, shadow settings and
// post-processing stay, what the game drew into the scene is left out, and a
// mesh read from the game's resource files (brf.h) is drawn in its place with
// the game's own shader for the mesh's material, through the same entry points
// the game uses. Everything the renderer adds therefore applies to the model
// as it would in the game.

struct PreviewVertex { // the layout the model is handed over in
    float position[3], normal[3];
    uint8_t colour0[4], colour1[4];
    float uv[2], tangent[3], binormal[3], weight[4], bone[4];
};

struct PreviewPart {
    GLuint vertices = 0, indices = 0, program[2] = {0, 0}; // program: for the scene, for the depth pre-pass
    GLsizei count = 0;
    GLuint diffuse = 0, diffuse2 = 0, normal = 0, specular = 0, environment = 0;
    BrfMaterial material;
    bool skinned = false;
    float alpha_test = 0.5f;
};

// One model shown: a mesh with its parts, what it plays, and (in a gallery) where it stands.
struct PreviewModel {
    std::string name;
    // An action to play on a skinned model: frames `from` to `to` of an animation, taking `seconds`, over and over.
    const BrfAnimation *animation = nullptr;
    const BrfSkeleton *skeleton = nullptr;
    std::string action;
    double from = 0, to = 0, seconds = 1, held = -1; // held: a fixed place in the action, 0 to 1, or below 0 to let it run
    int most_bones = 0;   // one more than the highest bone any vertex of the model hangs on
    std::string material; // when set, used in place of the one the mesh names (heads name none: the game picks a face's by its own rules)
    std::vector<PreviewPart> parts;
    float centre[3] = {0, 0, 0}, radius = 1;
    // In a gallery: metres to the right of the stage's camera, up, and ahead of it (level, whichever way it tilts),
    // and a turn about the upright, degrees.
    double at[3] = {0, 0, 0}, turn = 0;
    double tilt[2] = {0, 0}; // first, in the model's own space: degrees about its x axis, then about its y (a head's mesh lies on its side)
    // For a ball: the part of its texture wrapped round it (left, top, right, bottom, as fractions of the picture
    // measured from its top left), so that a ball can show one material of a texture that holds many.
    float window[4] = {0, 0, 1, 1};
    bool windowed = false;
    bool grounded = false; // `up` is where its lowest point goes, not its middle (a plant standing on the ground)
    float lowest = 0;      // its lowest point, in its own space
};

struct Preview {
    bool active = false, keep_scene = false, library_read = false;
    BrfLibrary library;
    std::string root;
    PreviewModel m;                     // the model shown, or the one being loaded into a gallery
    std::vector<PreviewModel> gallery;  // several at once (mtl_preview_gallery); empty for one
    GLuint prepass_texture = 0; // the target of the engine's depth pre-pass in the frame being played
    GLuint next_id = 0x40000000; // far from anything a capture numbers
    std::map<std::string, GLuint> programs, textures;
};
Preview s_preview;

std::string game_folder()
{
    if (const char *game = getenv("WBM_GAME")) return game;
    const char *home = getenv("HOME");
    return std::string(home ? home : "") + "/Library/Application Support/Steam/steamapps/common/MountBlade Warband";
}

std::string lower_case(std::string s)
{
    for (char &c : s) c = (char)tolower((unsigned char)c);
    return s;
}

void preview_read_library()
{
    Preview &pv = s_preview;
    if (pv.library_read) return;
    pv.library_read = true;
    pv.root = game_folder();
    pv.library.scan(pv.root + "/CommonRes");
    // The module asked for (WBM_MODULE, as scripts/run.sh takes it), or else the one the game last played.
    const char *module_env = getenv("WBM_MODULE");
    std::string module = module_env ? module_env : "";
    if (const char *home = getenv("HOME"); home && module.empty()) {
        if (FILE *f = fopen((std::string(home) + "/Library/Application Support/MBWarband/last_module_warband").c_str(), "r")) {
            char text[512];
            size_t n = fread(text, 1, sizeof text - 1, f);
            fclose(f);
            module.assign(text, n);
            while (!module.empty() && isspace((unsigned char)module.back())) module.pop_back();
        }
    }
    if (!module.empty()) pv.library.scan(pv.root + "/Modules/" + module + "/Resource");
    for (const std::string &problem : pv.library.problems) wb_note("preview: %s", problem.c_str());
}

bool read_text(const std::string &path, std::string *out)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char buf[65536];
    size_t n;
    out->clear();
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) out->append(buf, n);
    fclose(f);
    return true;
}

// A program from the game's shader files for a technique, or 0.
GLuint preview_program(const std::string &technique)
{
    Preview &pv = s_preview;
    auto it = pv.programs.find(technique);
    if (it != pv.programs.end()) return it->second;
    GLuint id = 0;
    std::string vs, fs, dir = pv.root + "/GLShadersOptimized/";
    if (read_text(dir + "vs_" + technique + ".glsl", &vs) && read_text(dir + "fs_" + technique + ".glsl", &fs)) {
        id = pv.next_id++;
        Program &p = s_programs[id];
        for (int stage = 0; stage < 2; stage++) {
            auto shader = std::make_shared<Shader>();
            shader->type = stage ? GL_FRAGMENT_SHADER : GL_VERTEX_SHADER;
            shader->source = stage ? fs : vs;
            p.shaders.push_back(shader);
        }
        // The names and numbers the game binds its vertex attributes to.
        const char *const names[9] = {"inPosition", "inNormal", "inColor0", "inColor1", "inTexCoord", "inTangent", "inBinormal", "inBlendWeight", "inBlendIndices"};
        for (int n = 0; n < 9; n++) p.attribs[names[n]] = n;
        link_program(id, p);
        if (!p.linked) {
            complain("preview: the shaders of technique %s did not translate", technique.c_str());
            s_programs.erase(id);
            id = 0;
        } else {
            // What the engine sets for every program alike comes from the stage.
            for (const UniformLocation &l : p.locations) {
                auto u = s_stage_uniforms.find(l.name);
                if (u == s_stage_uniforms.end()) continue;
                SavedUniform value = u->second;
                value.unit = l.unit;
                restore_uniform(p, value);
            }
        }
    }
    pv.programs[technique] = id;
    return id;
}

// The first technique of the list whose shader files exist.
GLuint preview_program_of(std::initializer_list<std::string> techniques)
{
    for (const std::string &t : techniques)
        if (GLuint id = preview_program(t)) return id;
    return 0;
}

// One of the game's textures by name, loaded as the game would have it; 0 for "none".
GLuint preview_texture(const std::string &texture_name)
{
    Preview &pv = s_preview;
    std::string name = lower_case(texture_name);
    if (name.size() > 4 && !name.compare(name.size() - 4, 4, ".dds")) name.resize(name.size() - 4);
    if (name.empty() || name == "none") return 0;
    auto it = pv.textures.find(name);
    if (it == pv.textures.end()) {
        std::string path;
        auto pack = s_pack_files.find(name);
        if (pack != s_pack_files.end()) path = pack->second;
        for (const char *folder : {"/TexturesHD/", "/Textures/"}) {
            std::string candidate = pv.root + folder + name + ".dds";
            if (path.empty() && access(candidate.c_str(), R_OK) == 0) path = candidate;
        }
        for (const char *folder : {"/TexturesHD/", "/Textures/"}) { // an animated texture is a run of files: the first will do
            std::string candidate = pv.root + folder + name + "_0.dds";
            if (path.empty() && access(candidate.c_str(), R_OK) == 0) path = candidate;
        }
        GLuint id = 0;
        if (!path.empty()) {
            id = pv.next_id++;
            Texture &t = s_textures[id];
            if (load_replacement(t, path, true)) {
                t.material = (uint8_t)classify_texture_name(name);
                t.radial_wind = name.find("palm") != std::string::npos;
                t.woman = is_womans_face(name);
                horse_eye_for(name, t.horse_eye);
                t.min_filter = GL_LINEAR_MIPMAP_LINEAR;
                t.anisotropy = 16;
                auto map = s_pack_files.find(name + "_pbr");
                Texture pbr, hair, depth;
                if (map != s_pack_files.end() && load_replacement(pbr, map->second, true)) t.pbr = pbr.mtl;
                map = s_pack_files.find(name + "_depth");
                if (map != s_pack_files.end() && load_replacement(depth, map->second, true)) t.depth = depth.mtl;
                Texture sheen;
                map = s_pack_files.find(name + "_sheen");
                if (map != s_pack_files.end() && load_replacement(sheen, map->second, true)) t.sheen = sheen.mtl;
                map = s_pack_files.find(name + "_hair");
                if (map != s_pack_files.end() && load_replacement(hair, map->second, true)) {
                    t.hair = hair.mtl;
                    read_hair_colour(map->second, t.hair_colour);
                }
            } else {
                s_textures.erase(id);
                id = 0;
            }
        } else {
            complain("preview: no texture file named %s", name.c_str());
        }
        it = pv.textures.emplace(name, id).first;
    }
    return it->second;
}

// A ball 0.7 m across, for showing a material on (mesh name @sphere): smooth, with its texture wrapped
// twice round and once from pole to pole, and the tangents that normal maps need.
// With a window, its texture is only that part of the picture, and it runs to and fro round the ball so that no
// seam shows.
void preview_sphere(BrfMesh *mesh, const float *window)
{
    const int around = 96, down = 48;
    const float radius = 0.35f;
    mesh->skinned = false;
    mesh->vertices.clear(), mesh->indices.clear();
    for (int j = 0; j <= down; j++) {
        float v = (float)j / down, theta = v * (float)M_PI;
        for (int i = 0; i <= around; i++) {
            float u = (float)i / around, phi = u * 2.0f * (float)M_PI;
            BrfVertex x = {};
            float n[3] = {sinf(theta) * cosf(phi), sinf(theta) * sinf(phi), cosf(theta)};
            for (int k = 0; k < 3; k++) x.normal[k] = n[k], x.position[k] = n[k] * radius;
            x.tangent[0] = -sinf(phi), x.tangent[1] = cosf(phi), x.tangent[2] = 0;
            x.handedness = 1;
            x.uv[0] = u * 2.0f, x.uv[1] = v;
            if (window) { // the meshes' texture coordinates count upwards from the bottom of the picture
                float across = 1.0f - fabsf(2.0f * u - 1.0f);
                x.uv[0] = window[0] + (window[2] - window[0]) * across;
                x.uv[1] = 1.0f - (window[1] + (window[3] - window[1]) * v);
            }
            for (int k = 0; k < 4; k++) x.colour[k] = 255;
            mesh->vertices.push_back(x);
        }
    }
    for (int j = 0; j < down; j++)
        for (int i = 0; i < around; i++) {
            uint32_t a = (uint32_t)(j * (around + 1) + i), b = a + 1, c = a + (uint32_t)around + 1, d = c + 1;
            mesh->indices.insert(mesh->indices.end(), {a, c, b, b, c, d});
        }
}

// A flat square of ground 80 m across, its texture repeated every 5 m (mesh name @plane).
void preview_plane(BrfMesh *mesh)
{
    const int cells = 16;
    const float half = 40.0f;
    mesh->skinned = false;
    mesh->vertices.clear(), mesh->indices.clear();
    for (int j = 0; j <= cells; j++)
        for (int i = 0; i <= cells; i++) {
            BrfVertex x = {};
            x.position[0] = -half + 2.0f * half * (float)i / cells, x.position[1] = -half + 2.0f * half * (float)j / cells;
            x.normal[2] = 1, x.tangent[0] = 1, x.handedness = 1;
            x.uv[0] = (float)i, x.uv[1] = (float)j;
            for (int k = 0; k < 4; k++) x.colour[k] = 255;
            mesh->vertices.push_back(x);
        }
    for (int j = 0; j < cells; j++)
        for (int i = 0; i < cells; i++) {
            uint32_t a = (uint32_t)(j * (cells + 1) + i), b = a + 1, c = a + (uint32_t)cells + 1, d = c + 1;
            mesh->indices.insert(mesh->indices.end(), {a, b, c, b, d, c});
        }
}

bool preview_add_part(const std::string &mesh_name, float *lo, float *hi)
{
    Preview &pv = s_preview;
    BrfMesh mesh;
    if (mesh_name == "@plane") {
        if (pv.m.material.empty()) return complain("preview: a plane has no material of its own: name one"), false;
        preview_plane(&mesh);
    } else if (mesh_name == "@sphere") {
        if (pv.m.material.empty()) return complain("preview: a sphere has no material of its own: name one"), false;
        preview_sphere(&mesh, pv.m.windowed ? pv.m.window : nullptr);
    } else if (!pv.library.load_mesh(mesh_name, &mesh) || mesh.vertices.empty() || mesh.indices.empty()) {
        return false;
    }
    PreviewPart part;
    part.skinned = mesh.skinned;
    if (!pv.m.material.empty()) mesh.material = pv.m.material;
    auto material = pv.library.materials.find(lower_case(mesh.material));
    if (material != pv.library.materials.end()) part.material = material->second;
    else complain("preview: mesh %s names a material, %s, that no resource file has", mesh_name.c_str(), mesh.material.c_str());

    // Heads are kept as one half, the engine making the other: every vertex on
    // one side of the middle, and the texture's right half only. The age of a
    // face is put into the alpha of its vertex colours by the game; here, young.
    bool half = !mesh.skinned;
    for (const BrfVertex &v : mesh.vertices) half = half && v.position[0] > -1e-4f && v.uv[0] > 0.49f;
    if (half) {
        size_t n = mesh.vertices.size(), triangles = mesh.indices.size() / 3;
        for (size_t i = 0; i < n; i++) {
            mesh.vertices[i].colour[3] = 0;
            BrfVertex m = mesh.vertices[i];
            m.position[0] = -m.position[0], m.normal[0] = -m.normal[0], m.tangent[0] = -m.tangent[0];
            m.uv[0] = 1.0f - m.uv[0];
            m.handedness = -m.handedness;
            mesh.vertices.push_back(m);
        }
        for (size_t t = 0; t < triangles; t++) {
            uint32_t a = mesh.indices[t * 3], b = mesh.indices[t * 3 + 1], c = mesh.indices[t * 3 + 2];
            mesh.indices.push_back((uint32_t)n + a), mesh.indices.push_back((uint32_t)n + c), mesh.indices.push_back((uint32_t)n + b);
        }
    }
    std::vector<PreviewVertex> vertices(mesh.vertices.size());
    for (size_t i = 0; i < vertices.size(); i++) {
        const BrfVertex &v = mesh.vertices[i];
        PreviewVertex &o = vertices[i];
        memcpy(o.position, v.position, 12), memcpy(o.normal, v.normal, 12), memcpy(o.tangent, v.tangent, 12);
        memcpy(o.colour0, v.colour, 4);
        memset(o.colour1, 255, 4);
        o.uv[0] = v.uv[0], o.uv[1] = v.uv[1]; // as stored: they suit the textures upside down, which is how the game loads them
        const float *n = v.normal, *t = v.tangent;
        o.binormal[0] = (n[1] * t[2] - n[2] * t[1]) * v.handedness;
        o.binormal[1] = (n[2] * t[0] - n[0] * t[2]) * v.handedness;
        o.binormal[2] = (n[0] * t[1] - n[1] * t[0]) * v.handedness;
        memcpy(o.weight, v.weight, 16), memcpy(o.bone, v.bone, 16);
        for (int k = 0; k < 4; k++)
            if (v.weight[k] > 0) pv.m.most_bones = std::max(pv.m.most_bones, (int)v.bone[k] + 1);
        if (!mesh.skinned) o.weight[0] = 1;
        for (int k = 0; k < 3; k++) lo[k] = std::min(lo[k], v.position[k]), hi[k] = std::max(hi[k], v.position[k]);
    }
    part.vertices = pv.next_id++, part.indices = pv.next_id++;
    s_buffers[part.vertices].mtl = [s_device newBufferWithBytes:vertices.data() length:vertices.size() * sizeof(PreviewVertex) options:MTLResourceStorageModeShared];
    s_buffers[part.indices].mtl = [s_device newBufferWithBytes:mesh.indices.data() length:mesh.indices.size() * 4 options:MTLResourceStorageModeShared];
    part.count = (GLsizei)mesh.indices.size();

    // Bits 12 and 13 of a material's flags ask for an alpha test, and say against what.
    bool cut_out = (part.material.flags & 0x3000) != 0;
    part.alpha_test = (part.material.flags & 0x2000) ? 0.53125f : 0.03125f;
    part.diffuse = preview_texture(part.material.diffuse);
    part.diffuse2 = preview_texture(part.material.diffuse2);
    part.normal = preview_texture(part.material.normal);
    part.specular = preview_texture(part.material.specular);
    part.environment = preview_texture(part.material.environment);
    // Water reflects a picture of the scene that the engine draws for it; here its own texture stands in.
    if (!part.environment && part.material.shader.find("water") != std::string::npos) part.environment = part.diffuse;

    // The material's shader names a technique; the game's shader files are named
    // after techniques, with a suffix for the variant in use. Shadowed, and
    // alpha-tested where the base texture has an alpha channel.
    std::string technique;
    auto t = pv.library.techniques.find(lower_case(part.material.shader));
    if (t != pv.library.techniques.end()) technique = t->second;
    // The instanced variants take each copy's placement from vertex attributes; one model needs none.
    const std::string instanced = "_Instanced";
    if (technique.size() > instanced.size() && !technique.compare(technique.size() - instanced.size(), instanced.size(), instanced))
        technique.resize(technique.size() - instanced.size());
    if (cut_out) part.program[0] = preview_program_of({technique + "_ALPHATESTSHDW", technique + "_ALPHATEST"});
    if (!part.program[0]) part.program[0] = preview_program_of({technique + "_SHDW", technique});
    if (!part.program[0]) part.program[0] = preview_program_of({mesh.skinned ? "skin_diffuse_SHDW" : "diffuse_SHDW", "diffuse"});
    if (!part.program[0]) {
        complain("preview: no shader for mesh %s (material %s, shader %s, technique %s)", mesh_name.c_str(), mesh.material.c_str(), part.material.shader.c_str(),
                 technique.c_str());
        return false;
    }
    part.program[1] = preview_program(std::string("earlyz_") + (mesh.skinned ? "skinned_" : "normal_") + (cut_out && !mesh.skinned ? "alphatested" : "solid") + "_shader");
    wb_note("preview: %s, %zu vertices, %zu triangles, material %s, shader %s, technique %s%s", mesh_name.c_str(), mesh.vertices.size(), mesh.indices.size() / 3,
            mesh.material.c_str(), part.material.shader.c_str(), technique.c_str(), mesh.skinned ? ", skinned (shown in its rest pose)" : "");
    pv.m.parts.push_back(part);
    return true;
}

// Loads the named mesh, with the further meshes that belong to it (name.1,
// name.2, ...), or one of its simpler versions (name.lod1, ...). Call between frames.
bool preview_load(const std::string &mesh_name, int lod)
{
    @autoreleasepool {
        Preview &pv = s_preview;
        preview_read_library();
        for (PreviewPart &part : pv.m.parts) s_buffers.erase(part.vertices), s_buffers.erase(part.indices);
        pv.m.parts.clear();
        pv.active = false;
        pv.m.most_bones = 0;
        std::string base = lower_case(mesh_name);
        float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
        // The model and its further parts (name.1 ... and name.lod1.1 ...; some models are parts only).
        bool any = false;
        if (base == "@sphere" || base == "@plane") any = preview_add_part(base, lo, hi);
        else
            for (const std::string &part : pv.library.parts_of(base, lod)) any = preview_add_part(part, lo, hi) || any;
        if (!any) return false;
        float extent = 0;
        for (int k = 0; k < 3; k++) pv.m.centre[k] = (lo[k] + hi[k]) * 0.5f, extent += (hi[k] - lo[k]) * (hi[k] - lo[k]);
        pv.m.radius = std::max(0.05f, sqrtf(extent) * 0.5f);
        pv.m.lowest = lo[2];
        pv.m.name = base;
        pv.active = true;
        return true;
    }
}

bool preview_set_action(const std::string &action);

// Reads a gallery: one model a line, "mesh material right up ahead turn [action [pose]]", a material of "-" for the
// mesh's own, "#" for a remark. Call between frames.
bool preview_load_gallery(const std::string &path, std::string *problem)
{
    Preview &pv = s_preview;
    for (PreviewModel &old : pv.gallery)
        for (PreviewPart &part : old.parts) s_buffers.erase(part.vertices), s_buffers.erase(part.indices);
    pv.gallery.clear();
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return *problem = "cannot read " + path, false;
    char line[1024];
    int number = 0;
    while (fgets(line, sizeof line, f)) {
        number++;
        char mesh[256], material[256], action[256] = "", turning[256] = "0", up[64];
        double at[3], turn = 0, held = -1, tilt[2] = {0, 0};
        char *hash = strchr(line, '#');
        if (hash) *hash = 0;
        // An up written as _h stands the model on the ground at h: its lowest point there, not its middle.
        int got = sscanf(line, "%255s %255s %lf %63s %lf %255s %255s %lf", mesh, material, &at[0], up, &at[2], turning, action, &held);
        bool grounded = got >= 4 && up[0] == '_';
        if (got >= 4) at[1] = atof(up + (grounded ? 1 : 0));
        sscanf(turning, "%lf,%lf,%lf", &turn, &tilt[0], &tilt[1]); // turn about the upright[,tilt about the model's x[,about its y]]
        if (got <= 0) continue;
        if (got < 5) {
            fclose(f);
            return *problem = path + ":" + std::to_string(number) + ": wants a mesh, a material and three distances", false;
        }
        pv.m = PreviewModel();
        // A material may name the part of its texture a ball shows: name[left,top,right,bottom].
        if (char *open = strchr(material, '[')) {
            pv.m.windowed = sscanf(open, "[%f,%f,%f,%f]", &pv.m.window[0], &pv.m.window[1], &pv.m.window[2], &pv.m.window[3]) == 4;
            *open = 0;
        }
        if (strcmp(material, "-")) pv.m.material = material;
        bool loaded = preview_load(mesh, 0);
        if (loaded && got >= 7) {
            pv.m.held = held;
            if (!preview_set_action(action)) wb_note("preview: %s:%d: no action or animation named %s", path.c_str(), number, action);
        }
        if (!loaded) {
            wb_note("preview: %s:%d: %s could not be shown; left out", path.c_str(), number, mesh);
            continue;
        }
        for (int k = 0; k < 3; k++) pv.m.at[k] = at[k];
        pv.m.turn = turn, pv.m.tilt[0] = tilt[0], pv.m.tilt[1] = tilt[1], pv.m.grounded = grounded;
        pv.gallery.push_back(std::move(pv.m));
    }
    fclose(f);
    pv.m = PreviewModel();
    pv.active = !pv.gallery.empty();
    if (!pv.active) *problem = "nothing in " + path + " could be shown";
    return pv.active;
}

// The actions of a horse, which all lie in one animation (Native's actions.txt).
const struct {
    const char *name;
    int from, to;
    double seconds;
} k_horse_actions[] = {{"stand", 600, 644, 1.5}, {"walk", 0, 31, 1.0}, {"trot", 50, 69, 0.8}, {"canter", 100, 116, 0.6}, {"gallop", 150, 165, 0.5},
                       {"rear", 265, 297, 1.7}, {"jump", 205, 222, 1.6}, {"turn", 450, 483, 1.0}};

// Chooses the action by name, or none for an empty name. "name:from:to:seconds" plays any animation of the game's files.
bool preview_set_action(const std::string &action)
{
    Preview &pv = s_preview;
    preview_read_library();
    pv.m.animation = nullptr, pv.m.skeleton = nullptr, pv.m.action = action;
    // How fast a horse doing this would be going, for its mane and tail to stream by.
    s_pretend_speed = action == "walk" ? 1.7f : action == "trot" ? 4.0f : action == "canter" ? 7.0f : action == "gallop" ? 11.0f : action == "jump" ? 6.0f : 0.0f;
    if (action.empty() || action == "none") return true;
    std::string animation = "anim_horse";
    bool found = false;
    for (auto &a : k_horse_actions)
        if (action == a.name) pv.m.from = a.from, pv.m.to = a.to, pv.m.seconds = a.seconds, found = true;
    if (!found) {
        char name[200];
        double from, to, seconds;
        if (sscanf(action.c_str(), "%199[^:]:%lf:%lf:%lf", name, &from, &to, &seconds) != 4 || seconds <= 0) return false;
        animation = lower_case(name), pv.m.from = from, pv.m.to = to, pv.m.seconds = seconds;
    }
    auto it = pv.library.animations.find(animation);
    if (it == pv.library.animations.end()) return false;
    for (auto &sk : pv.library.skeletons) // the skeleton it was made for has as many bones
        if (sk.second.bones.size() == it->second.bones.size()) pv.m.skeleton = &sk.second;
    if (!pv.m.skeleton) return false;
    pv.m.animation = &it->second;
    return true;
}

// Where each bone carries the vertices that hang on it, for the action at this
// moment: the bone's place in the pose, undone by its place at rest, which is
// where the mesh was made. False when nothing is playing or it does not fit the model.
bool preview_pose(double time, std::vector<double> *out)
{
    Preview &pv = s_preview;
    if (!pv.m.animation || !pv.m.skeleton || pv.m.most_bones > (int)pv.m.skeleton->bones.size()) return false;
    const BrfSkeleton &sk = *pv.m.skeleton;
    const BrfAnimation &an = *pv.m.animation;
    double phase = pv.m.held >= 0 ? pv.m.held : fmod(time / pv.m.seconds, 1.0);
    double frame = pv.m.from + (pv.m.to - pv.m.from) * phase;
    size_t n = sk.bones.size();
    std::vector<double> rest(n * 16), posed(n * 16);
    for (size_t b = 0; b < n; b++) {
        const BrfSkeleton::Bone &bone = sk.bones[b];
        double local[16] = {bone.axes[0], bone.axes[1], bone.axes[2], 0, bone.axes[3], bone.axes[4], bone.axes[5], 0,
                            bone.axes[6], bone.axes[7], bone.axes[8], 0, bone.place[0], bone.place[1], bone.place[2], 1};
        double turned[16];
        memcpy(turned, local, sizeof turned);
        const std::vector<BrfAnimation::Turn> &keys = an.bones[b];
        if (!keys.empty()) {
            size_t k = 0;
            while (k + 1 < keys.size() && keys[k + 1].frame <= frame) k++;
            size_t k1 = std::min(k + 1, keys.size() - 1);
            double span = keys[k1].frame - keys[k].frame, t = span > 0 ? std::min(1.0, std::max(0.0, (frame - keys[k].frame) / span)) : 0;
            double q[4], dot = 0, len = 0;
            for (int i = 0; i < 4; i++) dot += keys[k].q[i] * keys[k1].q[i];
            for (int i = 0; i < 4; i++) q[i] = keys[k].q[i] * (1 - t) + keys[k1].q[i] * (dot < 0 ? -t : t), len += q[i] * q[i];
            len = sqrt(std::max(len, 1e-12));
            double x = q[0] / len, y = q[1] / len, z = q[2] / len, w = q[3] / len;
            const double r[9] = {1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w), 2 * (x * y - z * w), 1 - 2 * (x * x + z * z),
                                 2 * (y * z + x * w),     2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)}; // by columns
            for (int c = 0; c < 3; c++)
                for (int row = 0; row < 3; row++) turned[c * 4 + row] = r[c * 3 + row];
        }
        if (bone.parent < 0 && !an.root.empty()) { // the root is also moved
            size_t k = 0;
            while (k + 1 < an.root.size() && an.root[k + 1].frame <= frame) k++;
            size_t k1 = std::min(k + 1, an.root.size() - 1);
            double span = an.root[k1].frame - an.root[k].frame, t = span > 0 ? std::min(1.0, std::max(0.0, (frame - an.root[k].frame) / span)) : 0;
            for (int i = 0; i < 3; i++) turned[12 + i] = bone.place[i] + an.root[k].by[i] * (1 - t) + an.root[k1].by[i] * t; // on top of its place at rest
        }
        if (bone.parent >= 0 && (size_t)bone.parent < b) {
            mat_mul(&rest[(size_t)bone.parent * 16], local, &rest[b * 16]);
            mat_mul(&posed[(size_t)bone.parent * 16], turned, &posed[b * 16]);
        } else {
            memcpy(&rest[b * 16], local, sizeof local);
            memcpy(&posed[b * 16], turned, sizeof turned);
        }
    }
    const double identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    out->assign(32 * 16, 0.0);
    for (size_t b = 0; b < 32; b++) {
        double inverse[16];
        if (b < n && mat_inverse(&rest[b * 16], inverse)) mat_mul(&posed[b * 16], inverse, &(*out)[b * 16]);
        else memcpy(&(*out)[b * 16], identity, sizeof identity);
    }
    return true;
}

void preview_matrix(GLuint program, const char *name, const double *m, int count = 1)
{
    GLint location = mtl_glGetUniformLocation(program, name);
    if (location < 0) return;
    std::vector<float> values((size_t)count * 16);
    for (int n = 0; n < count; n++)
        for (int k = 0; k < 16; k++) values[(size_t)n * 16 + k] = (float)m[k];
    mtl_glUniformMatrix4fv(location, count, GL_FALSE, values.data());
}

void preview_vector(GLuint program, const char *name, float x, float y, float z, float w)
{
    GLint location = mtl_glGetUniformLocation(program, name);
    const float v[4] = {x, y, z, w};
    if (location >= 0) mtl_glUniform4fv(location, 1, v);
}

// How far ahead of the stage's camera the model is put.
double preview_distance() { return std::max(0.5, (double)s_preview.m.radius * 1.7); }

// Draws pv.m into the pass that is current: the scene, or the depth pre-pass.
void preview_draw_one(bool prepass)
{
    Preview &pv = s_preview;
    Replay &rp = s_replay;

    // The model's own space is laid over the world's with its middle put a
    // little way ahead of the camera the frame was captured with.
    double inverse_view[16], seen[16], world[16], projection[16] = {}, world_view[16], world_view_projection[16], view_projection[16], inverse_seen[16];
    if (!mat_inverse(rp.view, inverse_view)) return;
    double ahead = (pv.gallery.empty() ? preview_distance() : 0.0) / rp.C; // w = C * z
    double target[3];
    for (int r = 0; r < 3; r++) target[r] = inverse_view[8 + r] * ahead + inverse_view[12 + r];
    translation(target[0] - pv.m.centre[0], target[1] - pv.m.centre[1], target[2] - pv.m.centre[2], world);
    if (!pv.gallery.empty()) {
        // A gallery's model stands where it was told: right, up and ahead of the camera, ahead and right kept level
        // (the world's up is z), turned about the upright through its middle.
        double forward[3] = {inverse_view[8] / rp.C, inverse_view[9] / rp.C, 0}, flat = hypot(forward[0], forward[1]);
        if (flat > 1e-9) forward[0] /= flat, forward[1] /= flat;
        const double right[3] = {forward[1], -forward[0], 0};
        double place[3];
        for (int k = 0; k < 3; k++) place[k] = target[k] + right[k] * pv.m.at[0] + forward[k] * pv.m.at[2] + (k == 2 ? pv.m.at[1] : 0.0);
        if (pv.m.grounded) place[2] += pv.m.centre[2] - pv.m.lowest;
        double facing = atan2(forward[1], forward[0]) + pv.m.turn * M_PI / 180.0, c = cos(facing), sn = sin(facing);
        const double spin[16] = {c, sn, 0, 0, -sn, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        double to_middle[16], moved[16], spun[16], tilted[16], about_x[16], about_y[16];
        translation(-pv.m.centre[0], -pv.m.centre[1], -pv.m.centre[2], to_middle);
        double ax = pv.m.tilt[0] * M_PI / 180.0, ay = pv.m.tilt[1] * M_PI / 180.0;
        const double rx[16] = {1, 0, 0, 0, 0, cos(ax), sin(ax), 0, 0, -sin(ax), cos(ax), 0, 0, 0, 0, 1};
        const double ry[16] = {cos(ay), 0, -sin(ay), 0, 0, 1, 0, 0, sin(ay), 0, cos(ay), 0, 0, 0, 0, 1};
        mat_mul(rx, to_middle, about_x);
        mat_mul(ry, about_x, about_y);
        memcpy(tilted, about_y, sizeof tilted);
        mat_mul(spin, tilted, spun);
        translation(place[0], place[1], place[2], moved);
        mat_mul(moved, spun, world);
    }
    // Two aids for trying things out. WBM_PREVIEW_MOVE="x y z" puts the model that
    // many metres from where it would be (to sink a sheet of water into the ground
    // of a scene, say). WBM_PREVIEW_SWING=n swings it to and fro about its own
    // origin n times a second, as a weapon is swung: a rigid thing that moves.
    static const char *move = getenv("WBM_PREVIEW_MOVE"), *swing = getenv("WBM_PREVIEW_SWING");
    double by[3];
    if (move && sscanf(move, "%lf %lf %lf", &by[0], &by[1], &by[2]) == 3)
        for (int k = 0; k < 3; k++) world[12 + k] += by[k];
    if (swing && atof(swing) > 0) {
        double angle = 1.2 * sin(6.2831853 * atof(swing) * s_wind_time), c = cos(angle), sn = sin(angle);
        const double turn[16] = {1, 0, 0, 0, 0, c, sn, 0, 0, -sn, c, 0, 0, 0, 0, 1}; // about the model's x axis
        double place[16], turned[16];
        translation(target[0], target[1], target[2], place);
        if (move)
            for (int k = 0; k < 3; k++) place[12 + k] += by[k];
        mat_mul(place, turn, turned);
        memcpy(world, turned, sizeof world);
    }
    if (rp.camera_moved) mat_mul(rp.delta, rp.view, seen);
    else memcpy(seen, rp.view, sizeof seen);
    projection[0] = rp.p00, projection[5] = rp.p11, projection[10] = rp.A, projection[14] = rp.B, projection[11] = rp.C;
    mat_mul(seen, world, world_view);
    mat_mul(projection, world_view, world_view_projection);
    mat_mul(projection, seen, view_projection);
    const double identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float eye[3] = {0, 0, 0};
    if (mat_inverse(seen, inverse_seen))
        for (int k = 0; k < 3; k++) eye[k] = (float)inverse_seen[12 + k];

    // The state of the draw this one stands in for is put back afterwards.
    Attrib attribs[MAX_ATTRIBS];
    memcpy(attribs, s_attribs, sizeof attribs);
    GLuint array_buffer = s_array_buffer, element_buffer = s_element_buffer, program = s_program, unit = s_unit, bound[MAX_UNITS];
    memcpy(bound, s_bound_texture, sizeof bound);
    bool depth_test = s_depth_test, depth_mask = s_depth_mask, cull = s_cull, blend = s_blend;
    GLenum depth_func = s_depth_func;
    Program *stand_in = find(s_programs, program);

    std::vector<double> pose;
    bool posed = preview_pose(s_wind_time, &pose);
    s_depth_test = true, s_depth_mask = true, s_depth_func = GL_LEQUAL, s_cull = false, s_blend = false;
    for (PreviewPart &part : pv.m.parts) {
        if (prepass && part.material.shader.find("water") != std::string::npos) continue; // water is not in the engine's pre-pass: it asks it how deep the bed lies
        // The engine blends water in and does not let it hide what is drawn later; so here, when asked (WBM_PREVIEW_BLEND=1).
        bool blended = getenv("WBM_PREVIEW_BLEND") && part.material.shader.find("water") != std::string::npos;
        s_blend = blended, s_depth_mask = !blended;
        if (blended) s_blend_src = GL_SRC_ALPHA, s_blend_dst = GL_ONE_MINUS_SRC_ALPHA, s_blend_eq = GL_FUNC_ADD;
        GLuint id = part.program[prepass ? 1 : 0];
        Program *p = find(s_programs, id);
        if (!p) continue;
        mtl_glUseProgram(id);
        preview_matrix(id, "matWorld", world);
        preview_matrix(id, "matWorldView", world_view);
        preview_matrix(id, "matWorldViewProj", world_view_projection);
        preview_matrix(id, "matViewProj", view_projection);
        preview_matrix(id, "matWaterWorldViewProj", world_view_projection);
        GLint location = mtl_glGetUniformLocation(id, "use_depth_effects");
        if (location >= 0) mtl_glUniform1i(location, pv.keep_scene ? 1 : 0);
        // The pre-pass divides by the far plane's distance; the water multiplies by one over it.
        auto far_clip = s_stage_uniforms.find("far_clip");
        location = mtl_glGetUniformLocation(id, "far_clip_Inv");
        if (location >= 0 && far_clip != s_stage_uniforms.end()) {
            const std::vector<uint8_t> &raw = far_clip->second.has[0] ? far_clip->second.raw[0] : far_clip->second.raw[1];
            float far_ = 0;
            if (raw.size() >= 4) memcpy(&far_, raw.data(), 4);
            if (far_ > 0) mtl_glUniform1f(location, 1.0f / far_);
        }
        preview_vector(id, "vDepthRT_HalfPixel_ViewportSizeInv", 0, 0, 0, 0);
        if (posed) {
            GLint bones = mtl_glGetUniformLocation(id, "matWorldArray");
            std::vector<float> values(32 * 16);
            for (size_t k = 0; k < values.size(); k++) values[k] = (float)pose[k];
            if (bones >= 0) mtl_glUniformMatrix4fv(bones, 32, GL_FALSE, values.data());
        } else {
            preview_matrix(id, "matWorldArray", identity, 32); // every bone at rest
        }
        preview_vector(id, "vCameraPos", eye[0], eye[1], eye[2], 0);
        preview_vector(id, "vMaterialColor", 1, 1, 1, 1);
        preview_vector(id, "vSpecularColor", part.material.specular_colour[0], part.material.specular_colour[1], part.material.specular_colour[2], 1);
        location = mtl_glGetUniformLocation(id, "fMaterialPower");
        if (location >= 0) mtl_glUniform1f(location, part.material.shininess);
        location = mtl_glGetUniformLocation(id, "alpha_test_val");
        if (location >= 0) mtl_glUniform1f(location, part.alpha_test);
        // Samplers: the model's textures on units of their own; the engine's (the
        // shadow map and the like) wherever the draw being stood in for had them.
        const struct {
            const char *name;
            GLuint texture;
        } own[5] = {{"diffuse_texture", part.diffuse}, {"diffuse_texture_2", part.diffuse2}, {"normal_texture", part.normal},
                    {"specular_texture", part.specular}, {"env_texture", part.environment}};
        for (UniformLocation &l : p->locations) {
            if (l.texture_index[0] < 0 && l.texture_index[1] < 0) continue;
            bool mine = false;
            for (int n = 0; n < 5; n++) {
                if (l.name != own[n].name) continue;
                l.unit = 24 + n;
                s_bound_texture[24 + n] = own[n].texture;
                mine = true;
            }
            // On an empty stage the scenery's shadow map is left out: the trees and
            // walls in it are not there, and at 7 cm a texel their shadows fall on a
            // small model as large blocks.
            if (!pv.keep_scene && l.name == "shadowmap_texture") {
                l.unit = 30;
                s_bound_texture[30] = 0; // nothing bound reads as white: nothing nearer the sun
                continue;
            }
            // Water asks how far off the bed under it is: that is in the picture of the engine's depth pre-pass.
            if (l.name == "depth_texture") {
                if (pv.prepass_texture) { // what the pre-pass of this frame was drawn into (noted in preview_draw)
                    l.unit = 29;
                    s_bound_texture[29] = pv.prepass_texture;
                    mine = true;
                }
            }
            if (mine || !stand_in) continue;
            auto there = stand_in->location_by_name.find(l.name);
            if (there != stand_in->location_by_name.end()) l.unit = stand_in->locations[(size_t)there->second].unit;
        }
        mtl_glBindBuffer(GL_ARRAY_BUFFER, part.vertices);
        mtl_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, part.indices);
        const struct {
            GLint size;
            GLenum type;
            GLboolean normalized;
            size_t offset;
        } layout[9] = {{3, GL_FLOAT, GL_FALSE, offsetof(PreviewVertex, position)},      {3, GL_FLOAT, GL_FALSE, offsetof(PreviewVertex, normal)},
                       {4, GL_UNSIGNED_BYTE, GL_TRUE, offsetof(PreviewVertex, colour0)}, {4, GL_UNSIGNED_BYTE, GL_TRUE, offsetof(PreviewVertex, colour1)},
                       {2, GL_FLOAT, GL_FALSE, offsetof(PreviewVertex, uv)},             {3, GL_FLOAT, GL_FALSE, offsetof(PreviewVertex, tangent)},
                       {3, GL_FLOAT, GL_FALSE, offsetof(PreviewVertex, binormal)},       {4, GL_FLOAT, GL_FALSE, offsetof(PreviewVertex, weight)},
                       {4, GL_FLOAT, GL_FALSE, offsetof(PreviewVertex, bone)}};
        for (GLuint n = 0; n < 9; n++) {
            mtl_glEnableVertexAttribArray(n);
            mtl_glVertexAttribPointer(n, layout[n].size, layout[n].type, layout[n].normalized, (GLsizei)sizeof(PreviewVertex), (const void *)layout[n].offset);
        }
        mtl_glDrawElements(GL_TRIANGLES, part.count, GL_UNSIGNED_INT, nullptr);
    }
    memcpy(s_attribs, attribs, sizeof attribs);
    memcpy(s_bound_texture, bound, sizeof bound);
    s_array_buffer = array_buffer, s_element_buffer = element_buffer, s_program = program, s_unit = unit;
    s_depth_test = depth_test, s_depth_mask = depth_mask, s_cull = cull, s_blend = blend, s_depth_func = depth_func;
    s_last_key_program = 0;
}

// Draws the model, or every model of the gallery, into the pass that is current: the scene, or the depth pre-pass.
void preview_draw(bool prepass)
{
    Preview &pv = s_preview;
    if (!pv.active || !s_replay.camera_known) return;
    if (prepass)
        if (Framebuffer *fb = find(s_framebuffers, s_fbo)) pv.prepass_texture = fb->color_texture;
    if (pv.gallery.empty()) preview_draw_one(prepass);
    for (PreviewModel &shown : pv.gallery) {
        std::swap(pv.m, shown);
        preview_draw_one(prepass);
        std::swap(pv.m, shown);
    }
    if (!prepass) s_preview_in_scene = true;
}

bool preview_wanted(bool *keep_scene)
{
    *keep_scene = s_preview.keep_scene;
    return s_preview.active;
}
