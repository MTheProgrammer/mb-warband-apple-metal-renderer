#include "brf.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <stdexcept>

// The format: a string is a 32-bit length and that many characters; a file is a
// sequence of sections, each a name, a count and that many records, closed by a
// section named "end". "rfver " carries the file's version in place of a count.
// Nothing says how long a record is, so every kind has to be read to be passed.

namespace {

struct Reader {
    const std::vector<uint8_t> &b;
    size_t at = 0;
    void need(size_t n) const
    {
        if (at + n > b.size()) throw std::runtime_error("past the end of the file");
    }
    uint32_t u32()
    {
        need(4);
        uint32_t v;
        memcpy(&v, &b[at], 4);
        at += 4;
        return v;
    }
    float f32()
    {
        need(4);
        float v;
        memcpy(&v, &b[at], 4);
        at += 4;
        return v;
    }
    void skip(size_t n)
    {
        need(n);
        at += n;
    }
    std::string str()
    {
        uint32_t n = u32();
        if (n > 4096) throw std::runtime_error("a name of " + std::to_string(n) + " characters");
        need(n);
        std::string s((const char *)&b[at], n);
        at += n;
        return s;
    }
};

std::string lower(std::string s)
{
    for (char &c : s) c = (char)tolower((unsigned char)c);
    return s;
}

bool read_file(const std::string &path, std::vector<uint8_t> *out)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    out->resize((size_t)std::max(size, 0L));
    bool ok = fread(out->data(), 1, out->size(), f) == out->size();
    fclose(f);
    return ok;
}

// The size of a vertex record is not stored. Version 1 files have a tangent
// and a handedness byte (41 bytes) when bit 16 of the mesh's flags is set, and
// neither (28) otherwise; version 0 has two texture coordinates (36). A handful
// of meshes break the rule, so a size is accepted only if what follows it reads
// as a triangle list and then a name.
bool fits(const std::vector<uint8_t> &b, size_t at, size_t size, uint32_t vertices)
{
    size_t faces_at = at + size * vertices;
    if (faces_at + 4 > b.size()) return false;
    uint32_t faces;
    memcpy(&faces, &b[faces_at], 4);
    size_t after = faces_at + 4 + (size_t)faces * 12;
    if (faces > 5000000 || after + 4 > b.size()) return false;
    for (size_t i = 0; i < std::min<size_t>((size_t)faces * 3, 300); i++) {
        int32_t index;
        memcpy(&index, &b[faces_at + 4 + i * 4], 4);
        if (index < 0 || (uint32_t)index >= vertices) return false;
    }
    uint32_t n;
    memcpy(&n, &b[after], 4);
    if (n == 0 || n > 200 || after + 4 + n > b.size()) return false;
    for (uint32_t i = 0; i < n; i++)
        if (b[after + 4 + i] < 32 || b[after + 4 + i] > 126) return false;
    return true;
}

// Reads one mesh record; fills `mesh` when it is not null.
void read_mesh(Reader &r, int version, BrfMesh *mesh)
{
    std::string name = r.str();
    uint32_t flags = r.u32();
    std::string material = r.str();
    uint32_t positions = r.u32();
    size_t positions_at = r.at;
    r.skip((size_t)positions * 12);
    // Rigging: for each bone, the positions it moves and by how much.
    uint32_t bones = r.u32();
    std::vector<float> weight, bone;
    if (mesh) weight.assign((size_t)positions * 4, 0.0f), bone.assign((size_t)positions * 4, 0.0f);
    for (uint32_t i = 0; i < bones; i++) {
        uint32_t index = r.u32(), pairs = r.u32();
        for (uint32_t k = 0; k < pairs; k++) {
            uint32_t position = r.u32();
            float w = r.f32();
            if (!mesh || position >= positions) continue;
            // Keep the four heaviest.
            float *ws = &weight[(size_t)position * 4], *bs = &bone[(size_t)position * 4];
            int least = 0;
            for (int n = 1; n < 4; n++)
                if (ws[n] < ws[least]) least = n;
            if (w > ws[least]) ws[least] = w, bs[least] = (float)index;
        }
    }
    uint32_t frames = r.u32(); // further poses of a vertex-animated mesh
    if (mesh) mesh->frames.clear();
    for (uint32_t i = 0; i < frames; i++) {
        BrfFrame frame;
        frame.time = (int)r.u32();
        for (std::vector<float> *list : {&frame.positions, &frame.normals}) {
            uint32_t n = r.u32();
            r.need((size_t)n * 12);
            if (mesh) {
                list->resize((size_t)n * 3);
                if (n) memcpy(list->data(), &r.b[r.at], (size_t)n * 12);
            }
            r.skip((size_t)n * 12);
        }
        if (mesh) mesh->frames.push_back(std::move(frame));
    }
    uint32_t vertices = r.u32();
    size_t size = 0;
    const size_t first = version >= 1 ? ((flags & 0x10000) ? 41 : 28) : 36;
    for (size_t candidate : {first, (size_t)41, (size_t)36, (size_t)28})
        if (!size && fits(r.b, r.at, candidate, vertices)) size = candidate;
    if (!size) throw std::runtime_error("mesh " + name + ": no vertex layout fits");
    if (mesh) {
        mesh->name = name, mesh->material = material, mesh->flags = flags;
        mesh->skinned = bones > 0;
        mesh->has_tangents = size == 41;
        mesh->positions = positions, mesh->vertex_size = (int)size;
        mesh->position_of.resize(vertices);
        mesh->vertices.resize(vertices);
        for (uint32_t i = 0; i < vertices; i++) {
            const uint8_t *p = &r.b[r.at + (size_t)i * size];
            BrfVertex &v = mesh->vertices[i];
            memset(&v, 0, sizeof v);
            uint32_t position;
            memcpy(&position, p, 4);
            if (position >= positions) throw std::runtime_error("mesh " + name + ": a vertex outside its positions");
            memcpy(v.position, &r.b[positions_at + (size_t)position * 12], 12);
            mesh->position_of[i] = position;
            memcpy(v.colour, p + 4, 4);
            memcpy(v.normal, p + 8, 12);
            v.handedness = 1;
            if (size == 41) {
                memcpy(v.tangent, p + 20, 12);
                v.handedness = p[32] ? -1.0f : 1.0f;
                memcpy(v.uv, p + 33, 8);
            } else {
                memcpy(v.uv, p + 20, 8);
            }
            memcpy(v.weight, &weight[(size_t)position * 4], 16);
            memcpy(v.bone, &bone[(size_t)position * 4], 16);
            float total = v.weight[0] + v.weight[1] + v.weight[2] + v.weight[3];
            if (total > 1e-6f)
                for (float &w : v.weight) w /= total;
        }
    }
    r.skip(size * vertices);
    uint32_t faces = r.u32();
    if (mesh) {
        mesh->indices.resize((size_t)faces * 3);
        r.need((size_t)faces * 12);
        memcpy(mesh->indices.data(), &r.b[r.at], (size_t)faces * 12);
    }
    r.skip((size_t)faces * 12);
}

void read_body_part(Reader &r, const std::string &type)
{
    if (type == "manifold") {
        r.skip((size_t)r.u32() * 12);
        for (uint32_t faces = r.u32(); faces; faces--) {
            r.skip(8);
            r.skip((size_t)r.u32() * 4);
        }
    } else if (type == "capsule") {
        r.skip(32);
    } else if (type == "sphere") {
        r.skip(20);
    } else if (type == "face") {
        r.skip((size_t)r.u32() * 12);
        r.skip(4);
    } else {
        throw std::runtime_error("a collision body of kind " + type);
    }
}

} // namespace

void BrfLibrary::scan(const std::string &folder)
{
    DIR *d = opendir(folder.c_str());
    if (!d) return;
    std::vector<std::string> names;
    while (struct dirent *e = readdir(d)) {
        std::string name = e->d_name;
        if (name.size() > 4 && !strcasecmp(name.c_str() + name.size() - 4, ".brf")) names.push_back(name);
    }
    closedir(d);
    std::sort(names.begin(), names.end());
    for (const std::string &name : names) {
        std::string path = folder + "/" + name;
        std::vector<uint8_t> data;
        if (!read_file(path, &data)) continue;
        int file = (int)files.size();
        files.push_back(path);
        Reader r{data};
        int version = 0;
        try {
            for (;;) {
                std::string section = r.str();
                if (section == "end") break;
                if (section == "rfver ") {
                    version = (int)r.u32();
                    continue;
                }
                uint32_t count = r.u32();
                for (uint32_t i = 0; i < count; i++) {
                    if (section == "mesh") {
                        size_t at = r.at;
                        uint32_t n;
                        memcpy(&n, &data[at], 4);
                        std::string mesh_name = n <= 4096 && at + 4 + n <= data.size() ? std::string((const char *)&data[at + 4], n) : std::string();
                        read_mesh(r, version, nullptr);
                        meshes[lower(mesh_name)] = {file, at, version}; // a module's file read later replaces the game's own
                    } else if (section == "material") {
                        BrfMaterial m;
                        m.name = r.str();
                        m.flags = r.u32();
                        m.shader = r.str();
                        m.diffuse = r.str(), m.diffuse2 = r.str(), m.normal = r.str(), m.environment = r.str(), m.specular = r.str();
                        m.shininess = r.f32();
                        for (float &c : m.specular_colour) c = r.f32();
                        materials[lower(m.name)] = m;
                    } else if (section == "texture") {
                        r.str();
                        r.skip(4);
                    } else if (section == "shader") {
                        std::string shader = r.str();
                        r.skip(8);
                        std::string technique = r.str();
                        for (uint32_t n = r.u32(); n; n--) r.str();
                        r.skip((size_t)r.u32() * 16);
                        techniques[lower(shader)] = technique;
                    } else if (section == "skeleton") {
                        BrfSkeleton skeleton;
                        skeleton.name = r.str();
                        for (uint32_t n = r.u32(); n; n--) {
                            BrfSkeleton::Bone bone;
                            bone.parent = (int)r.u32();
                            bone.name = r.str();
                            bone.flags = r.u32();
                            for (float &v : bone.axes) v = r.f32();
                            for (float &v : bone.place) v = r.f32();
                            skeleton.bones.push_back(bone);
                        }
                        skeletons[lower(skeleton.name)] = skeleton;
                    } else if (section == "skeleton_anim") {
                        BrfAnimation animation;
                        animation.name = r.str();
                        for (uint32_t bones = r.u32(); bones; bones--) {
                            std::vector<BrfAnimation::Turn> turns(r.u32());
                            for (BrfAnimation::Turn &t : turns) {
                                t.frame = (int)r.u32();
                                for (float &v : t.q) v = r.f32();
                            }
                            animation.bones.push_back(std::move(turns));
                        }
                        animation.root.resize(r.u32());
                        for (BrfAnimation::Move &m : animation.root) {
                            m.frame = (int)r.u32();
                            for (float &v : m.by) v = r.f32();
                        }
                        animations[lower(animation.name)] = std::move(animation);
                    } else if (section == "body") {
                        r.str();
                        std::string type = r.str();
                        if (type == "composite") {
                            for (uint32_t n = r.u32(); n; n--) read_body_part(r, r.str());
                        } else {
                            read_body_part(r, type);
                        }
                    } else {
                        throw std::runtime_error("a section named " + section);
                    }
                }
            }
        } catch (const std::exception &e) {
            problems.push_back(name + ": " + e.what());
        }
    }
}

std::vector<std::string> BrfLibrary::parts_of(const std::string &base_name, int lod) const
{
    const std::string stem = lower(base_name) + (lod > 0 ? ".lod" + std::to_string(lod) : std::string());
    std::vector<std::pair<long, std::string>> found;
    if (meshes.count(stem)) found.push_back({0, stem});
    // Everything that begins "stem." lies together in the map.
    for (auto it = meshes.lower_bound(stem + "."); it != meshes.end() && it->first.compare(0, stem.size() + 1, stem + ".") == 0; ++it) {
        const std::string rest = it->first.substr(stem.size() + 1);
        if (rest.empty() || rest.find_first_not_of("0123456789") != std::string::npos) continue;
        found.push_back({atol(rest.c_str()), it->first});
    }
    std::sort(found.begin(), found.end());
    std::vector<std::string> names;
    for (auto &f : found) names.push_back(f.second);
    return names;
}

int BrfLibrary::lods_of(const std::string &base_name) const
{
    int most = 0;
    for (int k = 1; k <= 9; k++)
        if (!parts_of(base_name, k).empty()) most = k;
    return most;
}

bool BrfLibrary::mesh_extent(const std::string &name, std::string *path, size_t *begin, size_t *end) const
{
    auto it = meshes.find(lower(name));
    if (it == meshes.end()) return false;
    std::vector<uint8_t> data;
    if (!read_file(files[(size_t)it->second.file], &data)) return false;
    Reader r{data};
    r.at = it->second.offset;
    try {
        read_mesh(r, it->second.version, nullptr);
    } catch (const std::exception &) {
        return false;
    }
    *path = files[(size_t)it->second.file], *begin = it->second.offset, *end = r.at;
    return true;
}

std::vector<uint8_t> brf_mesh_record(const BrfMesh &mesh)
{
    std::vector<uint8_t> out;
    auto bytes = [&](const void *p, size_t n) { out.insert(out.end(), (const uint8_t *)p, (const uint8_t *)p + n); };
    auto u32 = [&](uint32_t v) { bytes(&v, 4); };
    auto str = [&](const std::string &s) { u32((uint32_t)s.size()), bytes(s.data(), s.size()); };
    str(mesh.name);
    u32(mesh.flags);
    str(mesh.material);
    // Positions, and for each the first vertex that stands there.
    std::vector<int> first(mesh.positions, -1);
    for (size_t i = 0; i < mesh.vertices.size(); i++)
        if (first[mesh.position_of[i]] < 0) first[mesh.position_of[i]] = (int)i;
    u32(mesh.positions);
    for (uint32_t p = 0; p < mesh.positions; p++) {
        const float none[3] = {0, 0, 0};
        bytes(first[p] >= 0 ? mesh.vertices[(size_t)first[p]].position : none, 12);
    }
    // Rigging, bone by bone: the positions it moves and by how much.
    std::map<uint32_t, std::vector<std::pair<uint32_t, float>>> rig;
    if (mesh.skinned)
        for (uint32_t p = 0; p < mesh.positions; p++) {
            if (first[p] < 0) continue;
            const BrfVertex &v = mesh.vertices[(size_t)first[p]];
            for (int k = 0; k < 4; k++)
                if (v.weight[k] > 0) rig[(uint32_t)lroundf(v.bone[k])].push_back({p, v.weight[k]});
        }
    u32((uint32_t)rig.size());
    for (auto &bone : rig) {
        u32(bone.first), u32((uint32_t)bone.second.size());
        for (auto &pair : bone.second) u32(pair.first), bytes(&pair.second, 4);
    }
    u32((uint32_t)mesh.frames.size());
    for (const BrfFrame &frame : mesh.frames) {
        u32((uint32_t)frame.time);
        for (const std::vector<float> *list : {&frame.positions, &frame.normals}) u32((uint32_t)(list->size() / 3)), bytes(list->data(), list->size() * 4);
    }
    u32((uint32_t)mesh.vertices.size());
    for (size_t i = 0; i < mesh.vertices.size(); i++) {
        const BrfVertex &v = mesh.vertices[i];
        u32(mesh.position_of[i]);
        bytes(v.colour, 4);
        bytes(v.normal, 12);
        if (mesh.vertex_size == 41) {
            bytes(v.tangent, 12);
            out.push_back(v.handedness < 0 ? 1 : 0);
        }
        bytes(v.uv, 8);
        if (mesh.vertex_size == 36) bytes(v.uv, 8); // the older layout's second set
    }
    u32((uint32_t)(mesh.indices.size() / 3));
    bytes(mesh.indices.data(), mesh.indices.size() * 4);
    return out;
}

bool BrfLibrary::load_mesh(const std::string &name, BrfMesh *mesh) const
{
    auto it = meshes.find(lower(name));
    if (it == meshes.end()) return false;
    std::vector<uint8_t> data;
    if (!read_file(files[(size_t)it->second.file], &data)) return false;
    Reader r{data};
    r.at = it->second.offset;
    try {
        read_mesh(r, it->second.version, mesh);
    } catch (const std::exception &) {
        return false;
    }
    return true;
}
