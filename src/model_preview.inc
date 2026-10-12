// Showing one of the game's models in a played-back frame. Included by
// metal_backend.mm, inside its anonymous namespace, after frame_capture.inc.
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

struct Preview {
    bool active = false, keep_scene = false, library_read = false;
    BrfLibrary library;
    std::string root, name;
    std::string material; // when set, used in place of the one the mesh names (heads name none: the game picks a face's by its own rules)
    std::vector<PreviewPart> parts;
    float centre[3] = {0, 0, 0}, radius = 1;
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
    std::string module;
    if (const char *home = getenv("HOME")) {
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
                Texture pbr, hair;
                if (map != s_pack_files.end() && load_replacement(pbr, map->second, true)) t.pbr = pbr.mtl;
                map = s_pack_files.find(name + "_hair");
                if (map != s_pack_files.end() && load_replacement(hair, map->second, true)) t.hair = hair.mtl;
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

bool preview_add_part(const std::string &mesh_name, float *lo, float *hi)
{
    Preview &pv = s_preview;
    BrfMesh mesh;
    if (!pv.library.load_mesh(mesh_name, &mesh) || mesh.vertices.empty() || mesh.indices.empty()) return false;
    PreviewPart part;
    part.skinned = mesh.skinned;
    if (!pv.material.empty()) mesh.material = pv.material;
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
    pv.parts.push_back(part);
    return true;
}

// Loads the named mesh, with the further meshes that belong to it (name.1,
// name.2, ...), or one of its simpler versions (name.lod1, ...). Call between frames.
bool preview_load(const std::string &mesh_name, int lod)
{
    @autoreleasepool {
        Preview &pv = s_preview;
        preview_read_library();
        for (PreviewPart &part : pv.parts) s_buffers.erase(part.vertices), s_buffers.erase(part.indices);
        pv.parts.clear();
        pv.active = false;
        std::string base = lower_case(mesh_name), suffix = lod > 0 ? ".lod" + std::to_string(lod) : std::string();
        float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
        if (!preview_add_part(base + suffix, lo, hi)) return false;
        for (int n = 1; n < 10; n++) preview_add_part(base + "." + std::to_string(n) + suffix, lo, hi);
        float extent = 0;
        for (int k = 0; k < 3; k++) pv.centre[k] = (lo[k] + hi[k]) * 0.5f, extent += (hi[k] - lo[k]) * (hi[k] - lo[k]);
        pv.radius = std::max(0.05f, sqrtf(extent) * 0.5f);
        pv.name = base;
        pv.active = true;
        return true;
    }
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
double preview_distance() { return std::max(0.5, (double)s_preview.radius * 1.7); }

// Draws the model into the pass that is current: the scene, or the depth pre-pass.
void preview_draw(bool prepass)
{
    Preview &pv = s_preview;
    Replay &rp = s_replay;
    if (!pv.active || !rp.camera_known) return;

    // The model's own space is laid over the world's with its middle put a
    // little way ahead of the camera the frame was captured with.
    double inverse_view[16], seen[16], world[16], projection[16] = {}, world_view[16], world_view_projection[16], view_projection[16], inverse_seen[16];
    if (!mat_inverse(rp.view, inverse_view)) return;
    double ahead = preview_distance() / rp.C; // w = C * z
    double target[3];
    for (int r = 0; r < 3; r++) target[r] = inverse_view[8 + r] * ahead + inverse_view[12 + r];
    translation(target[0] - pv.centre[0], target[1] - pv.centre[1], target[2] - pv.centre[2], world);
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

    s_depth_test = true, s_depth_mask = true, s_depth_func = GL_LEQUAL, s_cull = false, s_blend = false;
    for (PreviewPart &part : pv.parts) {
        GLuint id = part.program[prepass ? 1 : 0];
        Program *p = find(s_programs, id);
        if (!p) continue;
        mtl_glUseProgram(id);
        preview_matrix(id, "matWorld", world);
        preview_matrix(id, "matWorldView", world_view);
        preview_matrix(id, "matWorldViewProj", world_view_projection);
        preview_matrix(id, "matViewProj", view_projection);
        preview_matrix(id, "matWorldArray", identity, 32); // every bone at rest
        preview_vector(id, "vCameraPos", eye[0], eye[1], eye[2], 0);
        preview_vector(id, "vMaterialColor", 1, 1, 1, 1);
        preview_vector(id, "vSpecularColor", part.material.specular_colour[0], part.material.specular_colour[1], part.material.specular_colour[2], 1);
        GLint location = mtl_glGetUniformLocation(id, "fMaterialPower");
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
    if (!prepass) s_preview_in_scene = true;
    memcpy(s_attribs, attribs, sizeof attribs);
    memcpy(s_bound_texture, bound, sizeof bound);
    s_array_buffer = array_buffer, s_element_buffer = element_buffer, s_program = program, s_unit = unit;
    s_depth_test = depth_test, s_depth_mask = depth_mask, s_cull = cull, s_blend = blend, s_depth_func = depth_func;
    s_last_key_program = 0;
}

bool preview_wanted(bool *keep_scene)
{
    *keep_scene = s_preview.keep_scene;
    return s_preview.active;
}
