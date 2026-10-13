// Takes models that have been to Blender and back (glTF 2.0 binary files, as
// gltf_export writes them and Blender's own exporter does) into the game's
// resource files:
//
//   gltf_import a.glb [b.glb ...] -o folder [--module name]
//
// Every mesh in the files that goes by the name of one of the game's (and its
// parts, name.1 ...) takes that mesh's place in a copy of the resource file it
// lies in, written to the folder; the rest of the file is copied as it is.
// What a glTF file does not carry, or Blender does not hand back, stays as the
// game has it: the mesh's flags and material, its vertex colours (--colours:
// the file's are meant), and its further poses and its rigging where the file
// has none of its own. Tangents are worked out
// anew from the texture coordinates, since those are what will have changed.
//
//   gltf_import ... --check
//
// says, for each mesh taken, how it compares with the game's: a file that has
// only been out and back should give the game's mesh again.
//
//   gltf_import ... --install module
//
// also puts the files where the game reads them: into the module's Resource
// folder, with the line of its module.ini that loads the game's own file of
// that name (load_resource) turned into one that loads the module's
// (load_mod_resource). What is replaced there is kept beside it as *.before.
//
// The spaces are gltf_export's, run backwards: the game's (x, y, z) is glTF's
// (-x, z, y), its v is 1 - v, and its handedness the other sign.
//
//   gltf_import ... --lods uv
//
// A model's simpler versions (name.lod1 ...), which the game shows from afar,
// are not in a file from Blender, and are left as they are unless this is
// given: the changed texture mapping is then carried over to them, each
// keeping its shape. Where the shape of the model itself was changed that
// cannot be done, and the full mesh takes their place, as it does throughout
// with --lods copy. A simpler version that is in the files is taken as it comes.
//
// Left as it is: a model's collision body.

#include "../src/brf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

void warn(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fputs("gltf_import: ", stderr);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

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

bool write_file(const std::string &path, const std::vector<uint8_t> &data)
{
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = fwrite(data.data(), 1, data.size(), f) == data.size();
    return fclose(f) == 0 && ok;
}

// ---- JSON, as much of it as a glTF file uses -----------------------------------------------

struct Json {
    enum Kind { Null, Number, Text, List, Object } kind = Null;
    double number = 0;
    std::string text;
    std::vector<Json> list;
    std::vector<std::pair<std::string, Json>> members;

    const Json &operator[](const char *key) const
    {
        static const Json none;
        for (auto &m : members)
            if (m.first == key) return m.second;
        return none;
    }
    const Json &operator[](size_t i) const
    {
        static const Json none;
        return i < list.size() ? list[i] : none;
    }
    bool has(const char *key) const { return (*this)[key].kind != Null; }
    int integer(int otherwise = -1) const { return kind == Number ? (int)number : otherwise; }
};

struct JsonReader {
    const char *p, *end;
    bool ok = true;
    void space()
    {
        while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
    }
    std::string string()
    {
        std::string s;
        p++;
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) {
                p++;
                if (*p == 'u' && p + 4 < end) { // names are plain; anything else becomes a question mark
                    unsigned code = (unsigned)strtoul(std::string(p + 1, 4).c_str(), nullptr, 16);
                    s += code < 128 ? (char)code : '?';
                    p += 4;
                } else {
                    s += *p == 'n' ? '\n' : *p == 't' ? '\t' : *p;
                }
            } else {
                s += *p;
            }
            p++;
        }
        p++;
        return s;
    }
    Json value()
    {
        Json j;
        space();
        if (p >= end) {
            ok = false;
        } else if (*p == '{') {
            j.kind = Json::Object;
            p++;
            for (space(); ok && p < end && *p != '}'; space()) {
                if (*p != '"') {
                    ok = false;
                    break;
                }
                std::string key = string();
                space();
                if (p >= end || *p != ':') ok = false;
                p++;
                j.members.push_back({key, value()});
                space();
                if (p < end && *p == ',') p++;
            }
            p++;
        } else if (*p == '[') {
            j.kind = Json::List;
            p++;
            for (space(); ok && p < end && *p != ']'; space()) {
                j.list.push_back(value());
                space();
                if (p < end && *p == ',') p++;
            }
            p++;
        } else if (*p == '"') {
            j.kind = Json::Text;
            j.text = string();
        } else if (!strncmp(p, "true", 4) || !strncmp(p, "null", 4)) {
            j.kind = *p == 't' ? Json::Number : Json::Null, j.number = 1, p += 4;
        } else if (!strncmp(p, "false", 5)) {
            j.kind = Json::Number, p += 5;
        } else {
            char *after;
            j.kind = Json::Number;
            j.number = strtod(p, &after);
            if (after == p) ok = false;
            p = after;
        }
        return j;
    }
};

// ---- a glTF file -----------------------------------------------------------------------------

// Column-major, as glTF has them.
struct Matrix {
    double m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};

Matrix multiply(const Matrix &a, const Matrix &b)
{
    Matrix r;
    for (int c = 0; c < 4; c++)
        for (int row = 0; row < 4; row++) {
            double v = 0;
            for (int k = 0; k < 4; k++) v += a.m[k * 4 + row] * b.m[c * 4 + k];
            r.m[c * 4 + row] = v;
        }
    return r;
}

struct Gltf {
    Json json;
    std::vector<uint8_t> bin;
    std::vector<int> parent;

    bool load(const std::string &path)
    {
        std::vector<uint8_t> data;
        if (!read_file(path, &data) || data.size() < 20 || memcmp(data.data(), "glTF", 4)) return false;
        size_t at = 12;
        while (at + 8 <= data.size()) {
            uint32_t length, type;
            memcpy(&length, &data[at], 4), memcpy(&type, &data[at + 4], 4);
            if (at + 8 + length > data.size()) return false;
            if (type == 0x4E4F534A) { // JSON
                JsonReader r{(const char *)&data[at + 8], (const char *)&data[at + 8] + length};
                json = r.value();
                if (!r.ok) return false;
            } else if (type == 0x004E4942) { // BIN
                bin.assign(data.begin() + (long)at + 8, data.begin() + (long)(at + 8 + length));
            }
            at += 8 + length;
        }
        parent.assign(json["nodes"].list.size(), -1);
        for (size_t n = 0; n < json["nodes"].list.size(); n++)
            for (const Json &child : json["nodes"][n]["children"].list)
                if (child.integer() >= 0 && (size_t)child.integer() < parent.size()) parent[(size_t)child.integer()] = (int)n;
        return json.kind == Json::Object;
    }

    Matrix local(size_t node) const
    {
        const Json &n = json["nodes"][node];
        Matrix r;
        if (n["matrix"].list.size() == 16) {
            for (int i = 0; i < 16; i++) r.m[i] = n["matrix"][(size_t)i].number;
            return r;
        }
        double t[3] = {0, 0, 0}, q[4] = {0, 0, 0, 1}, s[3] = {1, 1, 1};
        for (size_t i = 0; i < 3 && n["translation"].list.size() == 3; i++) t[i] = n["translation"][i].number;
        for (size_t i = 0; i < 4 && n["rotation"].list.size() == 4; i++) q[i] = n["rotation"][i].number;
        for (size_t i = 0; i < 3 && n["scale"].list.size() == 3; i++) s[i] = n["scale"][i].number;
        double x = q[0], y = q[1], z = q[2], w = q[3];
        const double rot[9] = {1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w), 2 * (x * y - z * w), 1 - 2 * (x * x + z * z),
                               2 * (y * z + x * w),     2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)};
        for (int c = 0; c < 3; c++)
            for (int row = 0; row < 3; row++) r.m[c * 4 + row] = rot[c * 3 + row] * s[c];
        for (int row = 0; row < 3; row++) r.m[12 + row] = t[row];
        return r;
    }

    Matrix world(size_t node) const
    {
        Matrix m = local(node);
        for (int p = parent[node], guard = 0; p >= 0 && guard < 64; p = parent[(size_t)p], guard++) m = multiply(local((size_t)p), m);
        return m;
    }

    // An accessor's values as numbers, `width` to an element; whole numbers that stand for 0..1 are made so.
    bool read(int accessor, int width, std::vector<double> *out) const
    {
        const Json &a = json["accessors"][(size_t)accessor];
        const Json &view = json["bufferViews"][(size_t)a["bufferView"].integer(0)];
        if (accessor < 0 || a.kind != Json::Object || !a.has("bufferView") || a.has("sparse")) return false;
        static const std::map<std::string, int> widths = {{"SCALAR", 1}, {"VEC2", 2}, {"VEC3", 3}, {"VEC4", 4}};
        auto w = widths.find(a["type"].text);
        if (w == widths.end() || w->second < width) return false;
        int type = a["componentType"].integer(), count = a["count"].integer(0);
        size_t each = type == 5126 || type == 5125 ? 4 : type == 5123 || type == 5122 ? 2 : 1;
        size_t stride = view.has("byteStride") ? (size_t)view["byteStride"].integer() : each * (size_t)w->second;
        size_t start = (size_t)view["byteOffset"].integer(0) + (size_t)a["byteOffset"].integer(0);
        if (count && start + stride * ((size_t)count - 1) + each * (size_t)w->second > bin.size()) return false;
        bool unit = a["normalized"].integer(0) != 0;
        out->resize((size_t)count * (size_t)width);
        for (int i = 0; i < count; i++)
            for (int k = 0; k < width; k++) {
                const uint8_t *p = &bin[start + stride * (size_t)i + each * (size_t)k];
                double v = 0;
                if (type == 5126) {
                    float f;
                    memcpy(&f, p, 4), v = f;
                } else if (type == 5125) {
                    uint32_t u;
                    memcpy(&u, p, 4), v = u;
                } else if (type == 5123) {
                    uint16_t u;
                    memcpy(&u, p, 2), v = unit ? u / 65535.0 : u;
                } else if (type == 5121) {
                    v = unit ? *p / 255.0 : *p;
                } else {
                    return false;
                }
                (*out)[(size_t)i * (size_t)width + (size_t)k] = v;
            }
        return true;
    }
};

// ---- from a glTF mesh to the game's ----------------------------------------------------------

struct Importer {
    BrfLibrary library;
    std::map<std::string, std::vector<std::pair<std::string, BrfMesh>>> changed; // by resource file: mesh name, new mesh
    std::map<std::string, std::pair<BrfMesh, BrfMesh>> pairs; // by mesh name: the game's mesh, and the one taken in its place
    int taken = 0, passed = 0, carried = 0, copied = 0;
    bool check = false; // say how each mesh taken compares with the game's
    bool file_colours = false;

    static void to_game(const double *g, float *out) { out[0] = (float)-g[0], out[1] = (float)g[2], out[2] = (float)g[1]; }

    // The bones a skin's joints stand for, by the numbers the game's skeleton gives them.
    std::vector<int> bones_of(const Gltf &g, int skin) const
    {
        const Json &joints = g.json["skins"][(size_t)skin]["joints"];
        std::vector<std::string> names;
        for (const Json &j : joints.list) names.push_back(lower(g.json["nodes"][(size_t)j.integer(0)]["name"].text));
        std::vector<int> best;
        size_t most = 0;
        for (auto &s : library.skeletons) {
            std::vector<int> map(names.size(), -1);
            size_t found = 0;
            for (size_t i = 0; i < names.size(); i++)
                for (size_t b = 0; b < s.second.bones.size(); b++)
                    if (lower(s.second.bones[b].name) == names[i]) map[i] = (int)b, found++;
            if (found > most) most = found, best = map;
        }
        if (most < names.size()) { // unknown names: the order they were written in will have to do
            best.resize(names.size());
            for (size_t i = 0; i < names.size(); i++)
                if (most == 0 || best[i] < 0) best[i] = (int)i;
        }
        return best;
    }

    void take(const Gltf &g, size_t node, const std::string &path)
    {
        const Json &n = g.json["nodes"][node];
        int mesh_index = n["mesh"].integer();
        if (mesh_index < 0) return;
        const Json &m = g.json["meshes"][(size_t)mesh_index];
        std::string name = lower(n["name"].text);
        if (!library.meshes.count(name)) name = lower(m["name"].text);
        BrfMesh was;
        std::string file;
        size_t begin, end;
        if (!library.meshes.count(name) || !library.load_mesh(name, &was) || !library.mesh_extent(name, &file, &begin, &end)) {
            warn("%s: the game has no mesh called %s; passed over", path.c_str(), n["name"].text.empty() ? m["name"].text.c_str() : n["name"].text.c_str());
            passed++;
            return;
        }
        int skin = n["skin"].integer();
        std::vector<int> bones;
        if (skin >= 0) bones = bones_of(g, skin);
        const Matrix place = skin >= 0 ? Matrix() : g.world(node); // a skinned mesh stands where its bones put it

        BrfMesh mesh;
        mesh.name = was.name, mesh.material = was.material, mesh.flags = was.flags, mesh.vertex_size = was.vertex_size;
        mesh.has_tangents = was.vertex_size == 41;
        mesh.skinned = was.skinned;
        bool coloured = false, rigged_all = was.skinned;
        std::vector<std::vector<double>> moved; // per further pose, a shift for every vertex
        std::vector<std::vector<double>> turned;
        for (const Json &prim : m["primitives"].list) {
            if (prim.has("mode") && prim["mode"].integer() != 4) continue;
            const Json &attr = prim["attributes"];
            std::vector<double> position, normal, uv, colour, joint, weight, index;
            if (!g.read(attr["POSITION"].integer(), 3, &position)) {
                warn("%s: mesh %s has a part whose positions cannot be read; passed over", path.c_str(), name.c_str());
                passed++;
                return;
            }
            size_t count = position.size() / 3, base = mesh.vertices.size();
            bool has_normal = g.read(attr["NORMAL"].integer(), 3, &normal) && normal.size() == count * 3;
            bool has_uv = g.read(attr["TEXCOORD_0"].integer(), 2, &uv) && uv.size() == count * 2;
            int colour_width = g.json["accessors"][(size_t)attr["COLOR_0"].integer(0)]["type"].text == "VEC3" ? 3 : 4;
            bool has_colour = attr.has("COLOR_0") && g.read(attr["COLOR_0"].integer(), colour_width, &colour) && colour.size() == count * (size_t)colour_width;
            bool rigged = skin >= 0 && mesh.skinned && g.read(attr["JOINTS_0"].integer(), 4, &joint) && g.read(attr["WEIGHTS_0"].integer(), 4, &weight) && joint.size() == count * 4;
            coloured = coloured || has_colour;
            rigged_all = rigged_all && rigged;
            for (size_t i = 0; i < count; i++) {
                BrfVertex v;
                memset(&v, 0, sizeof v);
                double p[3], nrm[3] = {0, 1, 0};
                for (int r = 0; r < 3; r++) p[r] = place.m[r] * position[i * 3] + place.m[4 + r] * position[i * 3 + 1] + place.m[8 + r] * position[i * 3 + 2] + place.m[12 + r];
                if (has_normal)
                    for (int r = 0; r < 3; r++) nrm[r] = place.m[r] * normal[i * 3] + place.m[4 + r] * normal[i * 3 + 1] + place.m[8 + r] * normal[i * 3 + 2];
                double length = sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]);
                for (double &c : nrm) c = length > 1e-12 ? c / length : 0;
                to_game(p, v.position);
                to_game(nrm, v.normal);
                v.handedness = 1;
                v.uv[0] = has_uv ? (float)uv[i * 2] : 0, v.uv[1] = has_uv ? 1.0f - (float)uv[i * 2 + 1] : 0;
                memset(v.colour, 255, 4);
                if (has_colour) { // stored blue, green, red, alpha
                    auto byte = [&](size_t k) { return (uint8_t)lround(std::min(1.0, std::max(0.0, colour[i * (size_t)colour_width + k])) * 255.0); };
                    v.colour[0] = byte(2), v.colour[1] = byte(1), v.colour[2] = byte(0), v.colour[3] = colour_width == 4 ? byte(3) : 255;
                }
                if (rigged) {
                    double total = 0;
                    for (size_t k = 0; k < 4; k++) total += weight[i * 4 + k];
                    for (size_t k = 0; k < 4; k++) {
                        size_t j = (size_t)joint[i * 4 + k];
                        v.weight[k] = total > 1e-9 ? (float)(weight[i * 4 + k] / total) : 0;
                        v.bone[k] = (float)(j < bones.size() ? bones[j] : 0);
                    }
                }
                mesh.vertices.push_back(v);
            }
            if (g.read(prim["indices"].integer(), 1, &index)) {
                for (double i : index) mesh.indices.push_back((uint32_t)(base + (size_t)i));
            } else {
                for (size_t i = 0; i < count; i++) mesh.indices.push_back((uint32_t)(base + i));
            }
            // Further poses, as shifts of every vertex from the first.
            const Json &targets = prim["targets"];
            if (moved.size() < targets.list.size()) moved.resize(targets.list.size()), turned.resize(targets.list.size());
            for (size_t t = 0; t < targets.list.size(); t++) {
                std::vector<double> dp, dn;
                bool has_p = g.read(targets[t]["POSITION"].integer(), 3, &dp) && dp.size() == count * 3;
                bool has_n = g.read(targets[t]["NORMAL"].integer(), 3, &dn) && dn.size() == count * 3;
                moved[t].resize(base * 3, 0), turned[t].resize(base * 3, 0);
                for (size_t i = 0; i < count * 3; i++) moved[t].push_back(has_p ? dp[i] : 0), turned[t].push_back(has_n ? dn[i] : 0);
            }
        }
        size_t count = mesh.vertices.size();
        if (!count || mesh.indices.empty()) {
            warn("%s: mesh %s is empty; passed over", path.c_str(), name.c_str());
            passed++;
            return;
        }
        while (mesh.indices.size() % 3) mesh.indices.pop_back();

        // Vertices at one place share a position, as the game has it: its rigging and its further poses go by
        // position. At one place means there in every pose, and hung on the same bones: a flag's corners lie
        // together when it is furled and apart when it flies.
        std::map<std::vector<uint32_t>, uint32_t> places;
        mesh.position_of.resize(count);
        for (size_t i = 0; i < count; i++) {
            std::vector<float> values(mesh.vertices[i].position, mesh.vertices[i].position + 3);
            for (auto &pose : moved)
                for (size_t k = 0; k < 3; k++) values.push_back((float)pose[i * 3 + k]);
            values.insert(values.end(), mesh.vertices[i].weight, mesh.vertices[i].weight + 4);
            values.insert(values.end(), mesh.vertices[i].bone, mesh.vertices[i].bone + 4);
            std::vector<uint32_t> key(values.size());
            memcpy(key.data(), values.data(), values.size() * 4);
            for (uint32_t &k : key)
                if (k == 0x80000000u) k = 0; // minus nought is nought
            auto it = places.emplace(key, (uint32_t)places.size()).first;
            mesh.position_of[i] = it->second;
        }
        mesh.positions = (uint32_t)places.size();

        // Vertex colours tint a thing, and their alpha says how far foliage sways. Blender hands them back only
        // when asked to, and then not as it got them (it takes them for light and re-encodes them). So each
        // vertex takes the colour of the game's vertex at, or nearest to, its place, unless told that the
        // file's colours are meant (--colours).
        // The same goes for the bones a vertex hangs on, when the file comes without its skeleton.
        bool was_coloured = false;
        for (const BrfVertex &v : was.vertices) was_coloured = was_coloured || v.colour[0] != 255 || v.colour[1] != 255 || v.colour[2] != 255 || v.colour[3] != 255;
        bool keep_colours = (!coloured || !file_colours) && was_coloured, keep_rig = was.skinned && !rigged_all;
        if (!file_colours && !was_coloured)
            for (BrfVertex &v : mesh.vertices) memset(v.colour, 255, 4);
        if (keep_colours || keep_rig) {
            std::map<std::array<uint32_t, 3>, size_t> at;
            for (size_t i = 0; i < was.vertices.size(); i++) {
                std::array<uint32_t, 3> key;
                memcpy(key.data(), was.vertices[i].position, 12);
                at.emplace(key, i);
            }
            size_t searched = 0;
            for (BrfVertex &v : mesh.vertices) {
                std::array<uint32_t, 3> key;
                memcpy(key.data(), v.position, 12);
                auto it = at.find(key);
                size_t from = it != at.end() ? it->second : 0;
                if (it == at.end()) {
                    float nearest = 1e30f;
                    for (size_t i = 0; i < was.vertices.size(); i++) {
                        float d = 0;
                        for (int k = 0; k < 3; k++) d += (was.vertices[i].position[k] - v.position[k]) * (was.vertices[i].position[k] - v.position[k]);
                        if (d < nearest) nearest = d, from = i;
                    }
                    searched++;
                }
                if (keep_colours) memcpy(v.colour, was.vertices[from].colour, 4);
                if (keep_rig) memcpy(v.weight, was.vertices[from].weight, 16), memcpy(v.bone, was.vertices[from].bone, 16);
            }
            if (keep_colours && searched) warn("mesh %s: the game's vertex colours are kept, by nearest place for %zu vertices that have moved", name.c_str(), searched);
            if (keep_rig) warn("mesh %s: the file has no skin; the game's rigging is kept%s", name.c_str(), searched ? ", by nearest place where vertices have moved" : "");
        }

        // Further poses: the file's own when it has as many as the game, at the game's times; the game's when the
        // file has none and the mesh still has the game's positions; none otherwise.
        if (!was.frames.empty()) {
            if (moved.size() == was.frames.size()) {
                for (size_t t = 0; t < moved.size(); t++) {
                    BrfFrame frame;
                    frame.time = was.frames[t].time;
                    frame.positions.assign((size_t)mesh.positions * 3, 0), frame.normals.resize(count * 3);
                    for (size_t i = 0; i < count; i++) {
                        double p[3], nrm[3];
                        for (int k = 0; k < 3; k++) p[k] = moved[t][i * 3 + (size_t)k], nrm[k] = turned[t][i * 3 + (size_t)k];
                        float shift[3], turn[3];
                        to_game(p, shift), to_game(nrm, turn);
                        for (size_t k = 0; k < 3; k++) {
                            frame.positions[(size_t)mesh.position_of[i] * 3 + k] = mesh.vertices[i].position[k] + shift[k];
                            frame.normals[i * 3 + k] = mesh.vertices[i].normal[k] + turn[k];
                        }
                    }
                    mesh.frames.push_back(std::move(frame));
                }
            } else if (moved.empty() && keep_poses(was, &mesh)) {
                // the game's, each vertex taking those of the game's vertex that stands where it does
            } else {
                warn("mesh %s: the game's has %zu further poses and the file's %zu; it is written with none", name.c_str(), was.frames.size(), moved.size());
            }
        }

        // Thin things seen from both sides (leaves, hair, flags) have every triangle twice, once each way round.
        // Blender keeps one of a pair; where the game's mesh was so and the file's is not, the other is put back.
        auto twinned = [](const std::vector<uint32_t> &idx, const std::vector<uint32_t> &place) {
            std::set<std::array<uint32_t, 3>> seen;
            size_t twins = 0;
            auto turned_to = [](std::array<uint32_t, 3> t) {
                while (t[0] > t[1] || t[0] > t[2]) t = {t[1], t[2], t[0]};
                return t;
            };
            for (size_t i = 0; i + 2 < idx.size(); i += 3) seen.insert(turned_to({place[idx[i]], place[idx[i + 1]], place[idx[i + 2]]}));
            for (size_t i = 0; i + 2 < idx.size(); i += 3) twins += seen.count(turned_to({place[idx[i]], place[idx[i + 2]], place[idx[i + 1]]}));
            return idx.empty() ? 0.0 : (double)twins / (double)(idx.size() / 3);
        };
        if (twinned(was.indices, was.position_of) > 0.9 && twinned(mesh.indices, mesh.position_of) < 0.1) {
            size_t faces = mesh.indices.size();
            for (size_t i = 0; i < faces; i += 3) {
                const uint32_t back[3] = {mesh.indices[i], mesh.indices[i + 2], mesh.indices[i + 1]};
                mesh.indices.insert(mesh.indices.end(), back, back + 3);
            }
        }

        if (mesh.has_tangents) tangents(mesh);
        if (check) compare(name, was, mesh);
        pairs[name] = {was, mesh};
        changed[file].push_back({name, std::move(mesh)});
        taken++;
    }

    // Tangents: along growing u, and a handedness that makes normal x tangent point against growing v, as the
    // game's meshes have it.
    static void tangents(BrfMesh &mesh)
    {
        const size_t count = mesh.vertices.size();
        std::vector<double> along(count * 3, 0), down(count * 3, 0);
        for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
            const BrfVertex &a = mesh.vertices[mesh.indices[i]], &b = mesh.vertices[mesh.indices[i + 1]], &c = mesh.vertices[mesh.indices[i + 2]];
            double du1 = b.uv[0] - a.uv[0], du2 = c.uv[0] - a.uv[0], dv1 = b.uv[1] - a.uv[1], dv2 = c.uv[1] - a.uv[1], area = du1 * dv2 - du2 * dv1;
            if (fabs(area) < 1e-14) continue;
            for (size_t k = 0; k < 3; k++) {
                double e1 = b.position[k] - a.position[k], e2 = c.position[k] - a.position[k];
                double t = (e1 * dv2 - e2 * dv1) / area, s = (e2 * du1 - e1 * du2) / area;
                for (size_t corner = 0; corner < 3; corner++) along[(size_t)mesh.indices[i + corner] * 3 + k] += t, down[(size_t)mesh.indices[i + corner] * 3 + k] -= s;
            }
        }
        for (size_t i = 0; i < count; i++) {
            BrfVertex &v = mesh.vertices[i];
            double t[3], d = 0, length = 0;
            for (size_t k = 0; k < 3; k++) d += along[i * 3 + k] * v.normal[k];
            for (size_t k = 0; k < 3; k++) t[k] = along[i * 3 + k] - d * v.normal[k], length += t[k] * t[k];
            if (length < 1e-20) { // no texture to run along: any direction square to the normal
                const double other[3] = {fabs(v.normal[0]) < 0.9 ? 1.0 : 0.0, fabs(v.normal[0]) < 0.9 ? 0.0 : 1.0, 0};
                t[0] = v.normal[1] * other[2] - v.normal[2] * other[1], t[1] = v.normal[2] * other[0] - v.normal[0] * other[2], t[2] = v.normal[0] * other[1] - v.normal[1] * other[0];
                length = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
            }
            for (size_t k = 0; k < 3; k++) v.tangent[k] = (float)(t[k] / sqrt(length));
            const double cross[3] = {v.normal[1] * v.tangent[2] - v.normal[2] * v.tangent[1], v.normal[2] * v.tangent[0] - v.normal[0] * v.tangent[2],
                                     v.normal[0] * v.tangent[1] - v.normal[1] * v.tangent[0]};
            v.handedness = cross[0] * down[i * 3] + cross[1] * down[i * 3 + 1] + cross[2] * down[i * 3 + 2] < 0 ? -1.0f : 1.0f;
        }
    }

    // Gives a mesh the game's further poses, when every vertex of it stands where one of the game's does: it
    // then moves as that one does (of several there, the one whose normal is nearest its own).
    static bool keep_poses(const BrfMesh &was, BrfMesh *mesh)
    {
        std::multimap<std::array<uint32_t, 3>, size_t> at;
        auto key = [](const BrfVertex &v) {
            std::array<uint32_t, 3> k;
            memcpy(k.data(), v.position, 12);
            return k;
        };
        for (size_t i = 0; i < was.vertices.size(); i++) at.emplace(key(was.vertices[i]), i);
        std::vector<size_t> from(mesh->vertices.size());
        for (size_t i = 0; i < mesh->vertices.size(); i++) {
            auto range = at.equal_range(key(mesh->vertices[i]));
            if (range.first == range.second) return false;
            float best = -2;
            for (auto it = range.first; it != range.second; ++it) {
                const float *a = was.vertices[it->second].normal, *b = mesh->vertices[i].normal;
                float d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
                if (d > best) best = d, from[i] = it->second;
            }
        }
        for (const BrfFrame &old : was.frames) {
            if (old.positions.size() != (size_t)was.positions * 3 || old.normals.size() != was.vertices.size() * 3) return mesh->frames.clear(), false;
            BrfFrame frame;
            frame.time = old.time;
            frame.positions.assign((size_t)mesh->positions * 3, 0), frame.normals.resize(mesh->vertices.size() * 3);
            for (size_t i = 0; i < mesh->vertices.size(); i++)
                for (size_t k = 0; k < 3; k++) {
                    frame.positions[(size_t)mesh->position_of[i] * 3 + k] = old.positions[(size_t)was.position_of[from[i]] * 3 + k];
                    frame.normals[i * 3 + k] = old.normals[from[i] * 3 + k];
                }
            mesh->frames.push_back(std::move(frame));
        }
        return true;
    }

    // ---- a model's simpler versions ---------------------------------------------------------
    //
    // The game shows a model from afar by simpler meshes of its own (name.lod1 ...), which a file from Blender
    // does not have. Where only the texture mapping was changed, the new mapping is carried over to them:
    // each corner of a simpler mesh finds the spot of the full mesh it lies on, by where it is and by the
    // mapping it had, and takes what that spot's mapping has become.

    // One triangle of the game's full mesh: where it is, how it was mapped and how it is mapped now.
    struct Source {
        double p[3][3], normal[3], was[3][2], now[3][2], lo[3], hi[3];
    };

    // The fixed mesh's triangles, each with the game's triangle at the same three places. False when too few
    // have one: the shape was changed, and there is nothing to go by.
    static bool sources(const BrfMesh &was, const BrfMesh &now, std::vector<Source> *out)
    {
        typedef std::array<long, 9> Key;
        auto key_of = [](const BrfMesh &m, size_t tri, size_t *first) {
            std::array<std::array<long, 3>, 3> c;
            for (size_t k = 0; k < 3; k++)
                for (size_t a = 0; a < 3; a++) c[k][a] = lround((double)m.vertices[m.indices[tri + k]].position[a] * 1e5);
            *first = c[1] < c[0] ? (c[2] < c[1] ? 2 : 1) : (c[2] < c[0] ? 2 : 0);
            Key key;
            for (size_t k = 0; k < 3; k++) std::copy(c[(*first + k) % 3].begin(), c[(*first + k) % 3].end(), key.begin() + (long)k * 3);
            return key;
        };
        std::multimap<Key, std::pair<size_t, size_t>> at; // the game's triangles: where they begin, and their first corner
        for (size_t t = 0; t + 2 < was.indices.size(); t += 3) {
            size_t first;
            Key key = key_of(was, t, &first);
            at.emplace(key, std::make_pair(t, first));
        }
        size_t triangles = now.indices.size() / 3, matched = 0;
        std::vector<bool> used(was.indices.size() / 3, false);
        std::vector<size_t> unmatched;
        // A pair of triangles at the same places: the game's from corner `turn`, the fixed one from corner `first`.
        auto pair = [&](size_t old_at, size_t turn, size_t t, size_t first) {
            Source s;
            for (size_t k = 0; k < 3; k++) {
                const BrfVertex &old = was.vertices[was.indices[old_at + (turn + k) % 3]];
                const BrfVertex &fixed = now.vertices[now.indices[t + (first + k) % 3]];
                for (size_t a = 0; a < 3; a++) s.p[k][a] = old.position[a];
                for (size_t a = 0; a < 2; a++) s.was[k][a] = old.uv[a], s.now[k][a] = fixed.uv[a];
            }
            double e1[3], e2[3];
            for (size_t a = 0; a < 3; a++) {
                e1[a] = s.p[1][a] - s.p[0][a], e2[a] = s.p[2][a] - s.p[0][a];
                s.lo[a] = std::min({s.p[0][a], s.p[1][a], s.p[2][a]}), s.hi[a] = std::max({s.p[0][a], s.p[1][a], s.p[2][a]});
            }
            s.normal[0] = e1[1] * e2[2] - e1[2] * e2[1], s.normal[1] = e1[2] * e2[0] - e1[0] * e2[2], s.normal[2] = e1[0] * e2[1] - e1[1] * e2[0];
            out->push_back(s);
            used[old_at / 3] = true; // each of the game's once: of two that lie on each other, the next gets the other
            matched++;
        };
        for (size_t t = 0; t + 2 < now.indices.size(); t += 3) {
            size_t first;
            auto it = at.find(key_of(now, t, &first));
            if (it == at.end()) {
                unmatched.push_back(t);
                continue;
            }
            pair(it->second.first, it->second.second, t, first);
            at.erase(it);
        }
        // Blender moves a rigged mesh's vertices by a hair: what found no triangle to the hundredth of a
        // millimetre takes one, still free, whose corners all lie within half a millimetre of its own.
        for (size_t t : unmatched) {
            bool found = false;
            for (size_t old_at = 0; !found && old_at + 2 < was.indices.size(); old_at += 3) {
                if (used[old_at / 3]) continue;
                for (size_t turn = 0; !found && turn < 3; turn++) {
                    bool near = true;
                    for (size_t k = 0; near && k < 3; k++)
                        for (size_t a = 0; near && a < 3; a++)
                            near = fabsf(was.vertices[was.indices[old_at + (turn + k) % 3]].position[a] - now.vertices[now.indices[t + k]].position[a]) < 5e-4f;
                    if (near) pair(old_at, turn, t, 0), found = true;
                }
            }
        }
        return triangles && (double)matched >= 0.97 * (double)triangles && (double)triangles >= 0.97 * (double)(was.indices.size() / 3);
    }

    // The point of a triangle nearest to p, as shares of its three corners.
    static void nearest_on(const double *p, const Source &s, double *share)
    {
        const double *a = s.p[0], *b = s.p[1], *c = s.p[2];
        double ab[3], ac[3], ap[3], bp[3], cp[3];
        for (size_t k = 0; k < 3; k++) ab[k] = b[k] - a[k], ac[k] = c[k] - a[k], ap[k] = p[k] - a[k], bp[k] = p[k] - b[k], cp[k] = p[k] - c[k];
        auto dot = [](const double *x, const double *y) { return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]; };
        auto set = [&](double u, double v, double w) { share[0] = u, share[1] = v, share[2] = w; };
        double d1 = dot(ab, ap), d2 = dot(ac, ap);
        if (d1 <= 0 && d2 <= 0) return set(1, 0, 0);
        double d3 = dot(ab, bp), d4 = dot(ac, bp);
        if (d3 >= 0 && d4 <= d3) return set(0, 1, 0);
        double vc = d1 * d4 - d3 * d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0) return set(1 - d1 / (d1 - d3), d1 / (d1 - d3), 0);
        double d5 = dot(ab, cp), d6 = dot(ac, cp);
        if (d6 >= 0 && d5 <= d6) return set(0, 0, 1);
        double vb = d5 * d2 - d1 * d6;
        if (vb <= 0 && d2 >= 0 && d6 <= 0) return set(1 - d2 / (d2 - d6), 0, d2 / (d2 - d6));
        double va = d3 * d6 - d5 * d4;
        if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
            double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return set(0, 1 - w, w);
        }
        double sum = va + vb + vc;
        if (fabs(sum) < 1e-30) return set(1, 0, 0);
        set(va / sum, vb / sum, vc / sum);
    }

    struct Carried {
        size_t corners = 0, split = 0;
        double farthest = 0, moved = 0; // metres off the full mesh at worst; and the largest change of a coordinate
    };

    // A simpler mesh with the mapping its corners find on the full mesh's triangles.
    static BrfMesh carry(const BrfMesh &lod, const std::vector<Source> &from, Carried *stats)
    {
        double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
        for (const Source &s : from)
            for (size_t a = 0; a < 3; a++) lo[a] = std::min(lo[a], s.lo[a]), hi[a] = std::max(hi[a], s.hi[a]);
        const double size = std::max(1e-6, sqrt((hi[0] - lo[0]) * (hi[0] - lo[0]) + (hi[1] - lo[1]) * (hi[1] - lo[1]) + (hi[2] - lo[2]) * (hi[2] - lo[2])));
        const double near = 0.02 * size, alike = 0.05; // what counts as one step off, in space and in the mapping

        BrfMesh out = lod;
        out.vertices.clear(), out.indices.clear(), out.position_of.clear();
        std::vector<size_t> origin;                                    // for each new vertex, the simpler mesh's own
        std::vector<std::vector<std::pair<std::array<float, 2>, uint32_t>>> made(lod.vertices.size()); // per vertex: mappings given, and to which new vertex
        for (size_t t = 0; t + 2 < lod.indices.size(); t += 3) {
            double centre[5] = {0, 0, 0, 0, 0};
            for (size_t k = 0; k < 3; k++) {
                const BrfVertex &v = lod.vertices[lod.indices[t + k]];
                for (size_t a = 0; a < 3; a++) centre[a] += v.position[a] / 3.0;
                for (size_t a = 0; a < 2; a++) centre[3 + a] += v.uv[a] / 3.0;
            }
            for (size_t k = 0; k < 3; k++) {
                const uint32_t index = lod.indices[t + k];
                const BrfVertex &v = lod.vertices[index];
                // Asked for a little way into its own triangle: at a seam, that is the side to take.
                double q[3], quv[2], p[3] = {v.position[0], v.position[1], v.position[2]};
                for (size_t a = 0; a < 3; a++) q[a] = p[a] + 0.02 * (centre[a] - p[a]);
                for (size_t a = 0; a < 2; a++) quv[a] = v.uv[a] + 0.02 * (centre[3 + a] - v.uv[a]);
                const Source *best = nullptr;
                double least = 1e30, off = 0;
                for (const Source &s : from) {
                    double box = 0; // no nearer than its box is
                    for (size_t a = 0; a < 3; a++) {
                        double d = std::max({s.lo[a] - q[a], q[a] - s.hi[a], 0.0});
                        box += d * d;
                    }
                    if (sqrt(box) / near >= least) continue;
                    double share[3], d3 = 0, duv = 0;
                    nearest_on(q, s, share);
                    for (size_t a = 0; a < 3; a++) {
                        double at = share[0] * s.p[0][a] + share[1] * s.p[1][a] + share[2] * s.p[2][a] - q[a];
                        d3 += at * at;
                    }
                    for (size_t a = 0; a < 2; a++) {
                        double at = share[0] * s.was[0][a] + share[1] * s.was[1][a] + share[2] * s.was[2][a] - quv[a];
                        duv += at * at;
                    }
                    bool facing = s.normal[0] * v.normal[0] + s.normal[1] * v.normal[1] + s.normal[2] * v.normal[2] >= 0;
                    double score = sqrt(d3) / near + sqrt(duv) / alike + (facing ? 0 : 2);
                    if (score < least) least = score, best = &s, off = sqrt(d3);
                }
                float uv[2] = {v.uv[0], v.uv[1]};
                if (best) {
                    const Source &s = *best;
                    double share[3], old[2], fresh[2];
                    nearest_on(p, s, share);
                    for (size_t a = 0; a < 2; a++) {
                        old[a] = share[0] * s.was[0][a] + share[1] * s.was[1][a] + share[2] * s.was[2][a];
                        fresh[a] = share[0] * s.now[0][a] + share[1] * s.now[1][a] + share[2] * s.now[2][a];
                    }
                    // What the corner's old mapping differed by from the spot's goes along, turned as the triangle's mapping was.
                    const double a11 = s.was[1][0] - s.was[0][0], a12 = s.was[2][0] - s.was[0][0], a21 = s.was[1][1] - s.was[0][1], a22 = s.was[2][1] - s.was[0][1];
                    const double det = a11 * a22 - a12 * a21, du = v.uv[0] - old[0], dv = v.uv[1] - old[1];
                    if (fabs(det) > 1e-12) {
                        const double x = (du * a22 - a12 * dv) / det, y = (a11 * dv - du * a21) / det; // along the triangle's two sides
                        for (size_t a = 0; a < 2; a++) fresh[a] += x * (s.now[1][a] - s.now[0][a]) + y * (s.now[2][a] - s.now[0][a]);
                    } else { // a triangle mapped to a line or a point turns nothing: the difference goes along as it is
                        fresh[0] += du, fresh[1] += dv;
                    }
                    uv[0] = (float)fresh[0], uv[1] = (float)fresh[1];
                    stats->farthest = std::max(stats->farthest, off);
                }
                stats->corners++;
                stats->moved = std::max({stats->moved, (double)fabsf(uv[0] - v.uv[0]), (double)fabsf(uv[1] - v.uv[1])});
                // One vertex may now need two mappings, where a seam has been cut through it.
                uint32_t to = ~0u;
                for (auto &m : made[index])
                    if (fabsf(m.first[0] - uv[0]) < 2e-3f && fabsf(m.first[1] - uv[1]) < 2e-3f) to = m.second;
                if (to == ~0u) {
                    to = (uint32_t)out.vertices.size();
                    if (!made[index].empty()) stats->split++;
                    made[index].push_back({{uv[0], uv[1]}, to});
                    BrfVertex n = v;
                    n.uv[0] = uv[0], n.uv[1] = uv[1];
                    out.vertices.push_back(n);
                    out.position_of.push_back(lod.position_of[index]);
                    origin.push_back(index);
                }
                out.indices.push_back(to);
            }
        }
        for (size_t f = 0; f < out.frames.size(); f++) { // a pose has a normal for every vertex
            const std::vector<float> &old = lod.frames[f].normals;
            if (old.size() != lod.vertices.size() * 3) continue;
            out.frames[f].normals.resize(out.vertices.size() * 3);
            for (size_t i = 0; i < out.vertices.size(); i++) memcpy(&out.frames[f].normals[i * 3], &old[origin[i] * 3], 12);
        }
        if (out.has_tangents) tangents(out);
        return out;
    }

    static std::string base_of(const std::string &name)
    {
        size_t dot = name.rfind('.');
        if (dot == std::string::npos || dot + 1 >= name.size() || name.find_first_not_of("0123456789", dot + 1) != std::string::npos) return name;
        return name.substr(0, dot);
    }

    // Gives the simpler versions of every model that had a mesh taken the change too: "uv" carries the new
    // mapping over, and where the shape was changed, or for "copy", they become the full mesh itself.
    void lods(const std::string &mode)
    {
        std::set<std::string> bases;
        for (auto &p : pairs)
            if (p.first.find(".lod") == std::string::npos) bases.insert(base_of(p.first));
        for (const std::string &base : bases) {
            std::vector<std::string> parts;
            for (const std::string &part : library.parts_of(base, 0))
                if (pairs.count(part)) parts.push_back(part);
            // Every material the model wears, to tell a part left as it was from one that is not there.
            std::set<std::string> worn;
            for (const std::string &part : library.parts_of(base, 0)) {
                BrfMesh m;
                if (library.load_mesh(part, &m)) worn.insert(lower(m.material));
            }
            for (int level = 1; level <= library.lods_of(base); level++) {
                for (const std::string &name : library.parts_of(base, level)) {
                    BrfMesh lod;
                    std::string file;
                    size_t begin, end;
                    if (pairs.count(name) || !library.load_mesh(name, &lod) || !library.mesh_extent(name, &file, &begin, &end)) continue; // one from the file stands
                    // A simpler mesh goes by the parts that wear its material: they share its texture.
                    std::vector<std::string> like;
                    for (const std::string &part : parts)
                        if (lower(pairs[part].first.material) == lower(lod.material)) like.push_back(part);
                    if (like.empty() && worn.count(lower(lod.material))) continue; // its part was not in the file: nothing has changed for it
                    if (like.empty()) like = parts;
                    std::vector<Source> from;
                    bool same_shape = mode == "uv";
                    for (const std::string &part : like) same_shape = same_shape && sources(pairs[part].first, pairs[part].second, &from);
                    BrfMesh made;
                    if (same_shape) {
                        Carried stats;
                        made = carry(lod, from, &stats);
                        carried++;
                        if (check)
                            printf("check %s: mapping carried to %zu corners from %zu triangles; at worst %.3f m off the full mesh; changed by up to %.4f; %zu vertices split at new seams\n",
                                   name.c_str(), stats.corners, from.size(), stats.farthest, stats.moved, stats.split);
                    } else {
                        if (mode == "uv") warn("%s: the shape of %s was changed, so its mapping cannot be carried over; the full mesh takes this one's place", name.c_str(), base.c_str());
                        made = pairs[like[0]].second;
                        for (size_t n = 1; n < like.size(); n++) { // several parts in one
                            const BrfMesh &more = pairs[like[n]].second;
                            if (more.vertex_size != made.vertex_size || !more.frames.empty() || !made.frames.empty()) continue;
                            uint32_t vertices = (uint32_t)made.vertices.size(), positions = made.positions;
                            made.vertices.insert(made.vertices.end(), more.vertices.begin(), more.vertices.end());
                            for (uint32_t at : more.position_of) made.position_of.push_back(at + positions);
                            for (uint32_t index : more.indices) made.indices.push_back(index + vertices);
                            made.positions += more.positions;
                        }
                        made.name = lod.name, made.material = lod.material;
                        copied++;
                    }
                    changed[file].push_back({name, std::move(made)});
                }
            }
        }
    }

    // How a mesh taken from a file compares with the game's: of use on one that has only been out and back.
    static void compare(const std::string &name, const BrfMesh &was, const BrfMesh &now)
    {
        auto corners = [](const BrfMesh &m) {
            std::multiset<std::array<long, 15>> set;
            for (size_t i = 0; i + 2 < m.indices.size(); i += 3) {
                std::array<std::array<long, 5>, 3> c;
                for (size_t k = 0; k < 3; k++) {
                    const BrfVertex &v = m.vertices[m.indices[i + k]];
                    c[k] = {lround(v.position[0] * 1e4), lround(v.position[1] * 1e4), lround(v.position[2] * 1e4), lround(v.uv[0] * 1e3), lround(v.uv[1] * 1e3)};
                }
                size_t first = c[1] < c[0] ? (c[2] < c[1] ? 2 : 1) : (c[2] < c[0] ? 2 : 0);
                std::array<long, 15> key;
                for (size_t k = 0; k < 3; k++) std::copy(c[(first + k) % 3].begin(), c[(first + k) % 3].end(), key.begin() + (long)k * 5);
                set.insert(key);
            }
            return set;
        };
        auto a = corners(was), b = corners(now);
        size_t same = 0;
        for (auto &t : b) same += a.count(t) != 0;
        // Per place: tangents, handedness, colours and bones of the game's vertex with the same place, normal and texture coordinate.
        std::map<std::array<long, 8>, const BrfVertex *> by;
        auto key = [](const BrfVertex &v) {
            return std::array<long, 8>{lround(v.position[0] * 1e4), lround(v.position[1] * 1e4), lround(v.position[2] * 1e4), lround(v.uv[0] * 1e4),
                                       lround(v.uv[1] * 1e4),       lround(v.normal[0] * 1e2),   lround(v.normal[1] * 1e2),   lround(v.normal[2] * 1e2)};
        };
        for (const BrfVertex &v : was.vertices) by[key(v)] = &v;
        size_t found = 0, tangent = 0, hand = 0, colour = 0, rig = 0;
        for (const BrfVertex &v : now.vertices) {
            auto it = by.find(key(v));
            if (it == by.end()) continue;
            const BrfVertex &o = *it->second;
            found++;
            tangent += o.tangent[0] * v.tangent[0] + o.tangent[1] * v.tangent[1] + o.tangent[2] * v.tangent[2] > 0.8f;
            hand += o.handedness == v.handedness;
            colour += !memcmp(o.colour, v.colour, 4);
            std::map<int, float> wa, wb;
            for (int k = 0; k < 4; k++) {
                if (o.weight[k] > 0) wa[(int)o.bone[k]] += o.weight[k];
                if (v.weight[k] > 0) wb[(int)v.bone[k]] += v.weight[k];
            }
            bool agree = wa.size() == wb.size();
            for (auto &w : wa) agree = agree && fabsf(wb[w.first] - w.second) < 0.01f;
            rig += agree;
        }
        printf("check %s: %zu of %zu triangles are the game's (which has %zu); %u positions (%u), %zu vertices (%zu), %zu poses (%zu); of %zu vertices found again:",
               name.c_str(), same, b.size(), a.size(), now.positions, was.positions, now.vertices.size(), was.vertices.size(), now.frames.size(), was.frames.size(), found);
        if (now.has_tangents) printf(" tangent %zu, handedness %zu,", tangent, hand);
        printf(" colour %zu, bones %zu\n", colour, rig);
    }

    // The resource files, each with its changed meshes in place of the game's.
    bool write(const std::string &out, std::vector<std::string> *written) const
    {
        for (auto &f : changed) {
            std::vector<uint8_t> data;
            if (!read_file(f.first, &data)) return false;
            struct Splice {
                size_t begin, end;
                std::vector<uint8_t> record;
            };
            std::vector<Splice> splices;
            for (auto &m : f.second) {
                std::string path;
                Splice s;
                if (!library.mesh_extent(m.first, &path, &s.begin, &s.end)) return false;
                s.record = brf_mesh_record(m.second);
                splices.push_back(std::move(s));
            }
            std::sort(splices.begin(), splices.end(), [](const Splice &a, const Splice &b) { return a.begin > b.begin; });
            for (const Splice &s : splices) { // from the last backwards, so that the places of the earlier ones hold
                data.erase(data.begin() + (long)s.begin, data.begin() + (long)s.end);
                data.insert(data.begin() + (long)s.begin, s.record.begin(), s.record.end());
            }
            std::string name = f.first.substr(f.first.rfind('/') + 1), path = out + "/" + name;
            if (!write_file(path, data)) {
                warn("cannot write %s", path.c_str());
                return false;
            }
            written->push_back(path);
        }
        return true;
    }
};

std::string game_folder()
{
    if (const char *g = getenv("WBM_GAME")) return g;
    const char *home = getenv("HOME");
    return std::string(home ? home : "") + "/Library/Application Support/Steam/steamapps/common/MountBlade Warband";
}

// Puts a written file into the module's Resource folder and has the module load it in place of the game's own.
bool install(const std::string &root, const std::string &module, const std::string &path)
{
    std::string file = path.substr(path.rfind('/') + 1), stem = file.substr(0, file.size() - 4);
    std::string folder = root + "/Modules/" + module, target = folder + "/Resource/" + file, ini = folder + "/module.ini";
    std::vector<uint8_t> data, text;
    if (!read_file(path, &data) || !read_file(ini, &text)) {
        warn("cannot read %s or %s", path.c_str(), ini.c_str());
        return false;
    }
    if (access(target.c_str(), F_OK) == 0 && access((target + ".before").c_str(), F_OK) != 0) rename(target.c_str(), (target + ".before").c_str());
    if (!write_file(target, data)) {
        warn("cannot write %s", target.c_str());
        return false;
    }
    // load_resource = name  ->  load_mod_resource = name
    std::string lines((const char *)text.data(), text.size()), result;
    bool switched = false, loaded = false;
    size_t at = 0;
    while (at < lines.size()) {
        size_t end = lines.find('\n', at);
        std::string line = lines.substr(at, end == std::string::npos ? std::string::npos : end - at + 1);
        at += line.size();
        std::string key, value;
        size_t eq = line.find('=');
        if (eq != std::string::npos && line.find('#') > eq) {
            for (char c : line.substr(0, eq))
                if (!isspace((unsigned char)c)) key += c;
            for (char c : line.substr(eq + 1))
                if (!isspace((unsigned char)c)) value += c;
        }
        if (lower(value) == lower(stem) && key == "load_resource") {
            line = "load_mod_resource = " + stem + (line.back() == '\n' ? (line.size() > 1 && line[line.size() - 2] == '\r' ? "\r\n" : "\n") : "");
            switched = true;
        } else if (lower(value) == lower(stem) && key == "load_mod_resource") {
            loaded = true;
        }
        result += line;
    }
    if (switched) {
        if (access((ini + ".before").c_str(), F_OK) != 0) write_file(ini + ".before", text);
        if (!write_file(ini, std::vector<uint8_t>(result.begin(), result.end()))) return false;
    } else if (!loaded) {
        warn("%s loads no resource file called %s: the file is in place, but the module will not read it", ini.c_str(), stem.c_str());
    }
    printf("%s -> %s%s\n", file.c_str(), target.c_str(), switched ? ", and module.ini now loads it from the module" : "");
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    std::vector<std::string> files;
    std::string out, module, into, lods = "keep";
    bool usage = false, check = false, colours = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--module") && i + 1 < argc) module = argv[++i];
        else if (!strcmp(argv[i], "--install") && i + 1 < argc) into = argv[++i];
        else if (!strcmp(argv[i], "--check")) check = true;
        else if (!strcmp(argv[i], "--colours")) colours = true;
        else if (!strcmp(argv[i], "--lods") && i + 1 < argc) lods = argv[++i];
        else if (argv[i][0] != '-') files.push_back(argv[i]);
        else usage = true;
    }
    if (usage || files.empty() || out.empty() || (lods != "keep" && lods != "uv" && lods != "copy")) {
        fprintf(stderr, "usage: gltf_import a.glb [b.glb ...] -o folder [--module name] [--check] [--colours] [--lods keep|uv|copy] [--install module]\n");
        return 2;
    }
    if (module.empty()) module = into.empty() ? "Native" : into;
    Importer im;
    im.check = check;
    im.file_colours = colours;
    const std::string root = game_folder();
    im.library.scan(root + "/CommonRes");
    im.library.scan(root + "/Modules/" + module + "/Resource");
    // A file of the module's that goes by the name of one of the game's own is, as a rule, one this tool put
    // there before. What a model is compared with, and its simpler versions carried over from, has to be the
    // game's own again, or a second run would find nothing changed.
    {
        BrfLibrary own;
        own.scan(root + "/CommonRes");
        std::set<std::string> names;
        for (const std::string &f : own.files) names.insert(lower(f.substr(f.rfind('/') + 1)));
        for (auto &m : im.library.meshes) {
            const std::string &file = im.library.files[(size_t)m.second.file];
            auto original = own.meshes.find(m.first);
            if (file.compare(0, root.size() + 9, root + "/Modules/") != 0 || !names.count(lower(file.substr(file.rfind('/') + 1))) || original == own.meshes.end()) continue;
            im.library.files.push_back(own.files[(size_t)original->second.file]);
            m.second = original->second;
            m.second.file = (int)im.library.files.size() - 1;
        }
    }
    if (im.library.meshes.empty()) {
        fprintf(stderr, "no resource files found; set WBM_GAME to the game's folder\n");
        return 1;
    }
    for (const std::string &path : files) {
        Gltf g;
        if (!g.load(path)) {
            warn("%s cannot be read as a glTF binary file; passed over", path.c_str());
            continue;
        }
        for (size_t n = 0; n < g.json["nodes"].list.size(); n++) im.take(g, n, path);
    }
    if (lods != "keep") im.lods(lods);
    mkdir(out.c_str(), 0755);
    std::vector<std::string> written;
    if (!im.write(out, &written)) return 1;
    printf("%d meshes taken into %zu resource files in %s; %d passed over", im.taken, written.size(), out.c_str(), im.passed);
    if (im.carried || im.copied) printf("; of their simpler versions, %d given the new mapping and %d replaced by the full mesh", im.carried, im.copied);
    printf("\n");
    if (!into.empty())
        for (const std::string &path : written)
            if (!install(root, into, path)) return 1;
    return 0;
}
