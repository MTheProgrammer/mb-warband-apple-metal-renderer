// Writes one of the game's models as a glTF 2.0 binary file (.glb), to be
// opened in Blender or any other program that reads the format:
//
//   gltf_export <mesh> [-o out.glb]      the mesh, with the further meshes that belong to it (name.1, name.2, ...)
//       --lod n                          one of its simpler versions (name.lodN, with its parts name.lodN.1 ...) instead
//       --skeleton name                  the skeleton a rigged mesh hangs on, when the one picked is wrong
//       --anim action                    an action of the module's actions.txt (horse_pace_1, or stand:2 for its
//                                        second sequence), or animation:from:to:seconds, as an animation of the bones
//       --module name                    the module whose resources and textures are read as well as the game's own
//       --no-textures                    materials without their pictures (a much smaller file)
//       --textures folder                pictures written there once, as PNG, and named by the file instead of
//                                        being put inside it: for many models that share them (the folder is
//                                        given as the file is to find it: textures, for one beside the file)
//       --check                          also says how the mesh's data lies: winding, tangents, extent
//   gltf_export --list word              the names of meshes, materials, skeletons and animations containing a word
//
// What becomes what:
//   a mesh of the resource files   a node and a mesh of the same name; with a skin when it is rigged
//   its material                   a material of the same name; the diffuse texture as base colour, the normal map
//   a texture                      an image of the texture's name, PNG, inside the file
//   a skeleton                     a skin of the skeleton's name; a node for each bone, named after it, in the file's order
//   a frame of a vertex animation  a morph target named after its time
// What glTF has no place for is kept in `extras` under names beginning brf_, so
// that a model can be taken back to the game: flags are written as text
// ("0x00010000") because Blender keeps whole numbers in 32 bits with a sign.
//
// The game's space has x to the model's right, y to its front and z up, in
// metres; glTF has x to the model's left, y up and z to its front. So
// (x, y, z) -> (-x, z, y): a turn, not a mirroring, so triangles stay wound as
// they are stored, anticlockwise seen from outside. Pictures are another
// matter. A mesh's v runs up a picture, 0 at its bottom row (the eye a horse
// has painted near the top of its texture is at v = 0.85 on its head), where
// glTF's runs down: v -> 1 - v. And the stored binormal (normal x tangent,
// times the stored handedness) points down the picture, where glTF's points
// up: the handedness is turned about, and so is the green of a normal map,
// which says how far the surface leans that way. --check measures the winding
// and the binormal's direction on the mesh being written.
//
// The game's folder is taken from WBM_GAME, or from where Steam puts it.

#include "../src/brf.h"
#include "dds.h"

#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <dirent.h>
#include <map>
#include <set>
#include <zlib.h>

namespace {

// glTF's axis i is the game's axis k_axis[i], times k_sign[i].
const int k_axis[3] = {0, 2, 1};
const double k_sign[3] = {-1, 1, 1};

void warn(const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    fputs("gltf_export: ", stderr);
    vfprintf(stderr, format, ap);
    fputc('\n', stderr);
    va_end(ap);
}

std::string lower(std::string s)
{
    for (char &c : s) c = (char)tolower((unsigned char)c);
    return s;
}

std::string game_folder()
{
    if (const char *game = getenv("WBM_GAME")) return game;
    const char *home = getenv("HOME");
    return std::string(home ? home : "") + "/Library/Application Support/Steam/steamapps/common/MountBlade Warband";
}

template <class T> void to_gltf(const T *in, T *out)
{
    for (int i = 0; i < 3; i++) out[i] = (T)(k_sign[i] * in[k_axis[i]]);
}

// A rotation of the game's space (by rows) as one of glTF's.
void turn_to_gltf(const double in[9], double out[9])
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) out[i * 3 + j] = k_sign[i] * k_sign[j] * in[k_axis[i] * 3 + k_axis[j]];
}

void quaternion_of(const double m[9], double q[4]) // m by rows; q is x, y, z, w
{
    double trace = m[0] + m[4] + m[8];
    if (trace > 0) {
        double s = sqrt(trace + 1) * 2;
        q[3] = s / 4, q[0] = (m[7] - m[5]) / s, q[1] = (m[2] - m[6]) / s, q[2] = (m[3] - m[1]) / s;
    } else if (m[0] > m[4] && m[0] > m[8]) {
        double s = sqrt(1 + m[0] - m[4] - m[8]) * 2;
        q[3] = (m[7] - m[5]) / s, q[0] = s / 4, q[1] = (m[1] + m[3]) / s, q[2] = (m[2] + m[6]) / s;
    } else if (m[4] > m[8]) {
        double s = sqrt(1 + m[4] - m[0] - m[8]) * 2;
        q[3] = (m[2] - m[6]) / s, q[0] = (m[1] + m[3]) / s, q[1] = s / 4, q[2] = (m[5] + m[7]) / s;
    } else {
        double s = sqrt(1 + m[8] - m[0] - m[4]) * 2;
        q[3] = (m[3] - m[1]) / s, q[0] = (m[2] + m[6]) / s, q[1] = (m[5] + m[7]) / s, q[2] = s / 4;
    }
    double len = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (int i = 0; i < 4; i++) q[i] /= len;
}

void rotation_of(const double q[4], double m[9]) // by rows
{
    double x = q[0], y = q[1], z = q[2], w = q[3];
    const double r[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                         2 * (y * z - x * w),     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)};
    memcpy(m, r, sizeof r);
}

// A placement: a rotation by rows and where the origin goes.
struct Place {
    double r[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, t[3] = {0, 0, 0};
};

Place operator*(const Place &a, const Place &b)
{
    Place o;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) o.r[i * 3 + j] = a.r[i * 3] * b.r[j] + a.r[i * 3 + 1] * b.r[3 + j] + a.r[i * 3 + 2] * b.r[6 + j];
        o.t[i] = a.r[i * 3] * b.t[0] + a.r[i * 3 + 1] * b.t[1] + a.r[i * 3 + 2] * b.t[2] + a.t[i];
    }
    return o;
}

Place inverse(const Place &a)
{
    Place o;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) o.r[i * 3 + j] = a.r[j * 3 + i];
    for (int i = 0; i < 3; i++) o.t[i] = -(o.r[i * 3] * a.t[0] + o.r[i * 3 + 1] * a.t[1] + o.r[i * 3 + 2] * a.t[2]);
    return o;
}

// A skeleton at rest in glTF's space: each bone in its parent's, as the turn
// and move a node carries, and in the model's.
struct Rig {
    const BrfSkeleton *skeleton = nullptr;
    std::vector<double> turn, move; // four and three numbers a bone
    std::vector<Place> world;
};

Rig rig_of(const BrfSkeleton &sk)
{
    Rig rig;
    rig.skeleton = &sk;
    size_t n = sk.bones.size();
    rig.turn.resize(n * 4), rig.move.resize(n * 3), rig.world.resize(n);
    for (size_t b = 0; b < n; b++) {
        const BrfSkeleton::Bone &bone = sk.bones[b];
        double game[9], turned[9], place[3] = {bone.place[0], bone.place[1], bone.place[2]};
        for (int row = 0; row < 3; row++)
            for (int c = 0; c < 3; c++) game[row * 3 + c] = bone.axes[c * 3 + row];
        double det = game[0] * (game[4] * game[8] - game[5] * game[7]) - game[1] * (game[3] * game[8] - game[5] * game[6]) + game[2] * (game[3] * game[7] - game[4] * game[6]);
        if (fabs(det - 1) > 1e-3) warn("skeleton %s: the axes of bone %s are not a plain turn (determinant %.4f)", sk.name.c_str(), bone.name.c_str(), det);
        turn_to_gltf(game, turned);
        quaternion_of(turned, &rig.turn[b * 4]);
        to_gltf(place, &rig.move[b * 3]);
        // From the numbers that are written, so that the file agrees with itself to the last digit.
        Place local;
        float q[4];
        for (int i = 0; i < 4; i++) q[i] = (float)rig.turn[b * 4 + i];
        const double written[4] = {q[0], q[1], q[2], q[3]};
        rotation_of(written, local.r);
        for (int i = 0; i < 3; i++) local.t[i] = (float)rig.move[b * 3 + i];
        rig.world[b] = bone.parent >= 0 && (size_t)bone.parent < b ? rig.world[(size_t)bone.parent] * local : local;
    }
    return rig;
}

// PNG, eight bits of red, green, blue and alpha.
std::vector<uint8_t> png_of(const Image &img)
{
    size_t w = (size_t)img.w, h = (size_t)img.h, row = w * 4;
    std::vector<uint8_t> raw((row + 1) * h), tries[5];
    for (auto &t : tries) t.resize(row);
    const std::vector<uint8_t> none(row, 0);
    for (size_t y = 0; y < h; y++) {
        // Each row is stored as its difference from its neighbours in whichever of the five ways leaves least.
        const uint8_t *cur = &img.px[y * row], *up = y ? cur - row : none.data();
        long sums[5] = {};
        for (size_t x = 0; x < row; x++) {
            int a = x >= 4 ? cur[x - 4] : 0, b = up[x], c = x >= 4 ? up[x - 4] : 0;
            int p = a + b - c, pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
            int paeth = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
            const int v[5] = {cur[x], cur[x] - a, cur[x] - b, cur[x] - (a + b) / 2, cur[x] - paeth};
            for (int f = 0; f < 5; f++) tries[f][x] = (uint8_t)v[f], sums[f] += abs((int8_t)tries[f][x]);
        }
        int best = 0;
        for (int f = 1; f < 5; f++)
            if (sums[f] < sums[best]) best = f;
        raw[y * (row + 1)] = (uint8_t)best;
        memcpy(&raw[y * (row + 1) + 1], tries[best].data(), row);
    }
    uLongf size = compressBound((uLong)raw.size());
    std::vector<uint8_t> packed(size);
    compress2(packed.data(), &size, raw.data(), (uLong)raw.size(), 6);
    packed.resize(size);

    std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    auto chunk = [&out](const char *type, const std::vector<uint8_t> &data) {
        size_t at = out.size();
        out.resize(at + 12 + data.size());
        uint32_t n = (uint32_t)data.size();
        for (int i = 0; i < 4; i++) out[at + (size_t)i] = (uint8_t)(n >> (24 - 8 * i));
        memcpy(&out[at + 4], type, 4);
        if (n) memcpy(&out[at + 8], data.data(), n);
        uint32_t crc = (uint32_t)crc32(0, &out[at + 4], n + 4);
        for (int i = 0; i < 4; i++) out[at + 8 + n + (size_t)i] = (uint8_t)(crc >> (24 - 8 * i));
    };
    std::vector<uint8_t> header(13, 0);
    for (int i = 0; i < 4; i++) header[(size_t)i] = (uint8_t)((uint32_t)img.w >> (24 - 8 * i)), header[4 + (size_t)i] = (uint8_t)((uint32_t)img.h >> (24 - 8 * i));
    header[8] = 8, header[9] = 6;
    chunk("IHDR", header);
    chunk("IDAT", packed);
    chunk("IEND", {});
    return out;
}

std::string quoted(const std::string &s)
{
    std::string o = "\"";
    for (unsigned char c : s) {
        char code[8];
        if (c == '"' || c == '\\') o += '\\', o += (char)c;
        else if (c < 32 || c > 126) snprintf(code, sizeof code, "\\u%04x", c), o += code; // the game's names are Latin-1 at most
        else o += (char)c;
    }
    return o + "\"";
}

std::string number(double v)
{
    char text[40];
    snprintf(text, sizeof text, "%.9g", std::isfinite(v) ? v : 0.0);
    return text;
}

std::string hex(uint32_t v)
{
    char text[16];
    snprintf(text, sizeof text, "\"0x%08x\"", v);
    return text;
}

template <class T> std::string list_of(const T *values, size_t n)
{
    std::string o = "[";
    for (size_t i = 0; i < n; i++) o += (i ? "," : "") + number((double)values[i]);
    return o + "]";
}

std::string joined(const std::vector<std::string> &items)
{
    std::string o = "[";
    for (size_t i = 0; i < items.size(); i++) o += (i ? "," : "") + items[i];
    return o + "]";
}

// The file being put together: the lists of its description and its one buffer.
struct Gltf {
    std::vector<std::string> nodes, scene, meshes, materials, textures, images, accessors, views, skins, animations;
    std::vector<uint8_t> bin;
    std::map<std::string, int> material_of, texture_of; // by lower-case name; -1 for a texture that could not be had

    // target: 34962 for vertex data, 34963 for indices, 0 for anything else.
    int view(const void *data, size_t bytes, int target = 0)
    {
        bin.resize((bin.size() + 3) & ~(size_t)3, 0);
        std::string v = "{\"buffer\":0,\"byteOffset\":" + std::to_string(bin.size()) + ",\"byteLength\":" + std::to_string(bytes);
        if (target) v += ",\"target\":" + std::to_string(target);
        views.push_back(v + "}");
        bin.insert(bin.end(), (const uint8_t *)data, (const uint8_t *)data + bytes);
        return (int)views.size() - 1;
    }

    // type: 5121 bytes, 5123 and 5125 unsigned 16 and 32 bits, 5126 floats.
    int accessor(int view_index, int type, size_t count, const char *shape, const std::string &more = "")
    {
        accessors.push_back("{\"bufferView\":" + std::to_string(view_index) + ",\"componentType\":" + std::to_string(type) + ",\"count\":" + std::to_string(count) +
                            ",\"type\":\"" + shape + "\"" + more + "}");
        return (int)accessors.size() - 1;
    }

    int floats(const std::vector<float> &values, int width, const char *shape, int target, bool bounds = false)
    {
        std::string more;
        if (bounds) {
            std::vector<float> lo((size_t)width, 1e30f), hi((size_t)width, -1e30f);
            for (size_t i = 0; i < values.size(); i++) lo[i % (size_t)width] = std::min(lo[i % (size_t)width], values[i]), hi[i % (size_t)width] = std::max(hi[i % (size_t)width], values[i]);
            more = ",\"min\":" + list_of(lo.data(), lo.size()) + ",\"max\":" + list_of(hi.data(), hi.size());
        }
        return accessor(view(values.data(), values.size() * 4, target), 5126, values.size() / (size_t)width, shape, more);
    }
};

struct Exporter {
    std::string root, module;
    BrfLibrary library;
    std::map<std::string, std::string> texture_files; // by lower-case name, without ".dds"
    bool textures = true, check = false;
    std::string texture_folder, texture_place; // where pictures are written, and what the file calls that place
    Gltf g;

    void read_library()
    {
        library.scan(root + "/CommonRes");
        library.scan(root + "/Modules/" + module + "/Resource");
        for (const std::string &problem : library.problems) warn("%s", problem.c_str());
        // A module's own picture comes before the game's, and the game's larger version before its smaller.
        for (const std::string &folder : {root + "/Modules/" + module + "/Textures", root + "/TexturesHD", root + "/Textures"}) {
            DIR *d = opendir(folder.c_str());
            if (!d) continue;
            while (struct dirent *e = readdir(d)) {
                std::string name = e->d_name;
                if (name.size() < 5 || strcasecmp(name.c_str() + name.size() - 4, ".dds")) continue;
                texture_files.emplace(lower(name.substr(0, name.size() - 4)), folder + "/" + name);
            }
            closedir(d);
        }
    }

    // The texture of that name as a texture of the file, or -1.
    int texture(const std::string &texture_name, bool normal_map = false)
    {
        std::string name = lower(texture_name);
        if (name.size() > 4 && !name.compare(name.size() - 4, 4, ".dds")) name.resize(name.size() - 4);
        if (!textures || name.empty() || name == "none") return -1;
        const std::string key = name + (normal_map ? "#normal" : "");
        auto known = g.texture_of.find(key);
        if (known != g.texture_of.end()) return known->second;
        int index = -1;
        auto file = texture_files.find(name);
        if (file == texture_files.end()) file = texture_files.find(name + "_0"); // an animated texture is a run of files: the first will do
        Image img;
        if (file == texture_files.end()) {
            warn("no texture file named %s", texture_name.c_str());
        } else if (!read_dds(file->second, &img)) {
            warn("%s is of a kind that is not read", file->second.c_str());
        } else {
            if (normal_map) {
                // Some normal maps (those of rooms, and of water) keep x in alpha and nothing in red and blue:
                // they are written as the others are, x in red and the length made up in blue.
                bool red = false, alpha = false;
                for (size_t i = 0; i < img.px.size(); i += 4) red = red || img.px[i] > 8, alpha = alpha || img.px[i + 3] < 247;
                if (!red && alpha)
                    for (size_t i = 0; i < img.px.size(); i += 4) {
                        float x = (float)img.px[i + 3] / 127.5f - 1.0f, y = (float)img.px[i + 1] / 127.5f - 1.0f;
                        img.px[i] = img.px[i + 3];
                        img.px[i + 2] = (uint8_t)lroundf((sqrtf(fmaxf(0.0f, 1.0f - x * x - y * y)) * 0.5f + 0.5f) * 255.0f);
                        img.px[i + 3] = 255;
                    }
                // Green the other way up, with the binormal: see the top of the file.
                for (size_t i = 1; i < img.px.size(); i += 4) img.px[i] = (uint8_t)(255 - img.px[i]);
            }
            std::vector<uint8_t> png = png_of(img);
            if (!texture_folder.empty()) {
                // Normal maps are not as the game has them (see the top of the file): kept apart from the pictures that are.
                std::string sub = normal_map ? "normals/" : "", path = texture_folder + "/" + sub + name + ".png";
                if (access(path.c_str(), R_OK) != 0) { // another export may be writing the same one: whole, or not at all
                    mkdir(texture_folder.c_str(), 0755);
                    if (normal_map) mkdir((texture_folder + "/normals").c_str(), 0755);
                    std::string part = path + "." + std::to_string(getpid());
                    FILE *f = fopen(part.c_str(), "wb");
                    if (f) fwrite(png.data(), 1, png.size(), f), fclose(f), rename(part.c_str(), path.c_str());
                    else warn("cannot write %s", path.c_str());
                }
                g.images.push_back("{\"name\":" + quoted(texture_name) + ",\"uri\":" + quoted(texture_place + "/" + sub + name + ".png") + "}");
            } else {
                int view = g.view(png.data(), png.size());
                g.images.push_back("{\"name\":" + quoted(texture_name) + ",\"mimeType\":\"image/png\",\"bufferView\":" + std::to_string(view) + "}");
            }
            g.textures.push_back("{\"name\":" + quoted(texture_name) + ",\"source\":" + std::to_string(g.images.size() - 1) + "}");
            index = (int)g.textures.size() - 1;
        }
        g.texture_of[key] = index;
        return index;
    }

    int material(const std::string &material_name)
    {
        std::string key = lower(material_name);
        auto known = g.material_of.find(key);
        if (known != g.material_of.end()) return known->second;
        std::string m = "{\"name\":" + quoted(material_name);
        auto it = library.materials.find(key);
        if (it == library.materials.end()) {
            // Heads name none: the game picks a face's material by its own rules.
            warn("no resource file has a material named %s", material_name.c_str());
            m += ",\"pbrMetallicRoughness\":{\"metallicFactor\":0,\"roughnessFactor\":0.7}";
        } else {
            const BrfMaterial &bm = it->second;
            m = "{\"name\":" + quoted(bm.name) + ",\"pbrMetallicRoughness\":{";
            int diffuse = texture(bm.diffuse), normal = texture(bm.normal, true);
            if (diffuse >= 0) m += "\"baseColorTexture\":{\"index\":" + std::to_string(diffuse) + "},";
            m += "\"metallicFactor\":0,\"roughnessFactor\":0.7}";
            if (normal >= 0) m += ",\"normalTexture\":{\"index\":" + std::to_string(normal) + "}";
            // Bits 12 and 13 of the flags ask for an alpha test, and say against what.
            if (bm.flags & 0x3000) m += std::string(",\"alphaMode\":\"MASK\",\"alphaCutoff\":") + ((bm.flags & 0x2000) ? "0.53125" : "0.03125");
            m += ",\"extras\":{\"brf_flags\":" + hex(bm.flags) + ",\"brf_shader\":" + quoted(bm.shader) + ",\"brf_diffuse\":" + quoted(bm.diffuse) +
                 ",\"brf_diffuse2\":" + quoted(bm.diffuse2) + ",\"brf_normal\":" + quoted(bm.normal) + ",\"brf_environment\":" + quoted(bm.environment) +
                 ",\"brf_specular\":" + quoted(bm.specular) + ",\"brf_shininess\":" + number(bm.shininess) + ",\"brf_specular_colour\":" +
                 list_of(bm.specular_colour, 3) + "}";
        }
        g.materials.push_back(m + "}");
        return g.material_of[key] = (int)g.materials.size() - 1;
    }

    // How the numbers of a mesh lie, measured rather than taken on trust.
    void report(const BrfMesh &mesh)
    {
        float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
        for (const BrfVertex &v : mesh.vertices)
            for (int k = 0; k < 3; k++) lo[k] = std::min(lo[k], v.position[k]), hi[k] = std::max(hi[k], v.position[k]);
        size_t faces = 0, outward = 0, framed = 0, along_u = 0, along_v = 0;
        for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
            const BrfVertex &a = mesh.vertices[mesh.indices[t]], &b = mesh.vertices[mesh.indices[t + 1]], &c = mesh.vertices[mesh.indices[t + 2]];
            double e1[3], e2[3], n[3], du1 = b.uv[0] - a.uv[0], du2 = c.uv[0] - a.uv[0], dv1 = b.uv[1] - a.uv[1], dv2 = c.uv[1] - a.uv[1];
            for (int k = 0; k < 3; k++) e1[k] = b.position[k] - a.position[k], e2[k] = c.position[k] - a.position[k];
            n[0] = e1[1] * e2[2] - e1[2] * e2[1], n[1] = e1[2] * e2[0] - e1[0] * e2[2], n[2] = e1[0] * e2[1] - e1[1] * e2[0];
            double facing = 0, area = du1 * dv2 - du2 * dv1;
            for (int k = 0; k < 3; k++) facing += n[k] * (a.normal[k] + b.normal[k] + c.normal[k]);
            if (facing == 0) continue;
            faces++, outward += facing > 0;
            if (!mesh.has_tangents || fabs(area) < 1e-12) continue;
            // Which way the surface runs as u and as v grow, against the stored tangent and the binormal made from it.
            double with_u = 0, with_v = 0;
            const float *nn = a.normal, *tt = a.tangent;
            const double binormal[3] = {(nn[1] * tt[2] - nn[2] * tt[1]) * a.handedness, (nn[2] * tt[0] - nn[0] * tt[2]) * a.handedness, (nn[0] * tt[1] - nn[1] * tt[0]) * a.handedness};
            for (int k = 0; k < 3; k++) with_u += (e1[k] * dv2 - e2[k] * dv1) / area * tt[k], with_v += (e2[k] * du1 - e1[k] * du2) / area * binormal[k];
            framed++, along_u += with_u > 0, along_v += with_v < 0;
        }
        fprintf(stderr, "check %s: x %.3f..%.3f, y %.3f..%.3f, z %.3f..%.3f in the game's space\n", mesh.name.c_str(), lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
        fprintf(stderr, "check %s: %zu of %zu triangles are wound anticlockwise seen from where their normals point\n", mesh.name.c_str(), outward, faces);
        if (framed)
            fprintf(stderr, "check %s: of %zu triangles the tangent runs with growing u in %zu, the binormal against growing v in %zu\n", mesh.name.c_str(), framed, along_u, along_v);
    }

    // How well a skeleton suits rigged meshes: how far, on average, a vertex
    // lies from the bones it hangs on. Negative when a mesh names a bone it lacks.
    double misfit(const Rig &rig, const std::vector<BrfMesh> &meshes)
    {
        double sum = 0, weight = 0;
        for (const BrfMesh &mesh : meshes) {
            if (!mesh.skinned) continue;
            for (const BrfVertex &v : mesh.vertices) {
                float p[3];
                to_gltf(v.position, p);
                for (int k = 0; k < 4; k++) {
                    if (v.weight[k] <= 0) continue;
                    if ((size_t)v.bone[k] >= rig.world.size()) return -1;
                    const double *at = rig.world[(size_t)v.bone[k]].t;
                    sum += v.weight[k] * sqrt((p[0] - at[0]) * (p[0] - at[0]) + (p[1] - at[1]) * (p[1] - at[1]) + (p[2] - at[2]) * (p[2] - at[2]));
                    weight += v.weight[k];
                }
            }
        }
        return weight > 0 ? sum / weight : 0;
    }

    // The bones as nodes, and the skin that names them. Returns the first bone's node.
    int add_skin(const Rig &rig)
    {
        const BrfSkeleton &sk = *rig.skeleton;
        int first = (int)g.nodes.size();
        size_t n = sk.bones.size();
        std::vector<float> inverses(n * 16, 0.0f);
        std::vector<std::string> joints;
        for (size_t b = 0; b < n; b++) {
            const BrfSkeleton::Bone &bone = sk.bones[b];
            std::vector<std::string> children;
            for (size_t c = 0; c < n; c++)
                if (sk.bones[c].parent == (int)b) children.push_back(std::to_string(first + (int)c));
            std::string node = "{\"name\":" + quoted(bone.name) + ",\"translation\":" + list_of(&rig.move[b * 3], 3) + ",\"rotation\":" + list_of(&rig.turn[b * 4], 4);
            if (!children.empty()) node += ",\"children\":" + joined(children);
            node += ",\"extras\":{\"brf_bone\":" + std::to_string(b) + ",\"brf_flags\":" + hex(bone.flags) + "}}";
            g.nodes.push_back(node);
            if (bone.parent < 0 || (size_t)bone.parent >= n) g.scene.push_back(std::to_string(first + (int)b));
            joints.push_back(std::to_string(first + (int)b));
            Place back = inverse(rig.world[b]);
            float *m = &inverses[b * 16]; // by columns
            for (int c = 0; c < 3; c++)
                for (int row = 0; row < 3; row++) m[c * 4 + row] = (float)back.r[row * 3 + c];
            m[12] = (float)back.t[0], m[13] = (float)back.t[1], m[14] = (float)back.t[2], m[15] = 1;
        }
        int matrices = g.accessor(g.view(inverses.data(), inverses.size() * 4), 5126, n, "MAT4");
        g.skins.push_back("{\"name\":" + quoted(sk.name) + ",\"joints\":" + joined(joints) + ",\"inverseBindMatrices\":" + std::to_string(matrices) + "}");
        return first;
    }

    void add_mesh(const BrfMesh &mesh, const BrfLibrary::MeshRef &from, const Rig *rig)
    {
        size_t n = mesh.vertices.size();
        std::vector<float> positions(n * 3), normals(n * 3), tangents, uvs(n * 2), weights;
        std::vector<uint8_t> colours, joints;
        bool coloured = false, skinned = mesh.skinned && rig;
        if (mesh.has_tangents) tangents.resize(n * 4);
        if (skinned) weights.resize(n * 4), joints.resize(n * 4);
        size_t loose = 0;
        auto unit = [](float *v, const float *instead) {
            float len = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            for (int k = 0; k < 3; k++) v[k] = len > 1e-12f ? v[k] / len : instead[k];
        };
        for (size_t i = 0; i < n; i++) {
            const BrfVertex &v = mesh.vertices[i];
            const float up[3] = {0, 1, 0};
            to_gltf(v.position, &positions[i * 3]);
            to_gltf(v.normal, &normals[i * 3]);
            unit(&normals[i * 3], up);
            if (mesh.has_tangents) {
                // A tangent of no length is replaced by any direction square to the normal.
                const float *nn = &normals[i * 3];
                const float other[3] = {fabsf(nn[0]) < 0.9f ? 1.0f : 0.0f, fabsf(nn[0]) < 0.9f ? 0.0f : 1.0f, 0};
                const float across[3] = {nn[1] * other[2] - nn[2] * other[1], nn[2] * other[0] - nn[0] * other[2], nn[0] * other[1] - nn[1] * other[0]};
                to_gltf(v.tangent, &tangents[i * 4]);
                unit(&tangents[i * 4], across);
                // The turn between the spaces keeps a cross product as it is; the binormal is wanted the other way.
                tangents[i * 4 + 3] = -v.handedness;
            }
            uvs[i * 2] = v.uv[0], uvs[i * 2 + 1] = 1.0f - v.uv[1];
            const uint8_t rgba[4] = {v.colour[2], v.colour[1], v.colour[0], v.colour[3]};
            colours.insert(colours.end(), rgba, rgba + 4);
            coloured = coloured || rgba[0] != 255 || rgba[1] != 255 || rgba[2] != 255 || rgba[3] != 255;
            if (skinned) {
                // A few meshes list a position under one bone twice: once, with the sum.
                float share[4], total = 0;
                for (int k = 0; k < 4; k++) {
                    share[k] = v.weight[k] > 0 ? v.weight[k] : 0;
                    for (int earlier = 0; earlier < k; earlier++)
                        if (share[k] > 0 && share[earlier] > 0 && v.bone[earlier] == v.bone[k]) share[earlier] += share[k], share[k] = 0;
                }
                for (float w : share) total += w;
                if (total <= 0) { // hangs on nothing: on the first bone, then
                    weights[i * 4] = 1;
                    loose++;
                    continue;
                }
                // Heaviest first, and the bone of an unused place is the first.
                int order[4] = {0, 1, 2, 3};
                std::stable_sort(order, order + 4, [&share](int a, int b) { return share[a] > share[b]; });
                for (int k = 0; k < 4; k++) {
                    float w = share[order[k]] / total;
                    weights[i * 4 + (size_t)k] = w;
                    joints[i * 4 + (size_t)k] = w > 0 ? (uint8_t)v.bone[order[k]] : 0;
                }
            }
        }
        if (loose) warn("mesh %s: %zu vertices hang on no bone; they are put on the first", mesh.name.c_str(), loose);

        std::string attributes = "\"POSITION\":" + std::to_string(g.floats(positions, 3, "VEC3", 34962, true)) + ",\"NORMAL\":" + std::to_string(g.floats(normals, 3, "VEC3", 34962));
        if (mesh.has_tangents) attributes += ",\"TANGENT\":" + std::to_string(g.floats(tangents, 4, "VEC4", 34962));
        attributes += ",\"TEXCOORD_0\":" + std::to_string(g.floats(uvs, 2, "VEC2", 34962));
        // Every vertex has a colour in the game's files; all white is left out, and means the same.
        if (coloured) attributes += ",\"COLOR_0\":" + std::to_string(g.accessor(g.view(colours.data(), colours.size(), 34962), 5121, n, "VEC4", ",\"normalized\":true"));
        if (skinned) {
            attributes += ",\"JOINTS_0\":" + std::to_string(g.accessor(g.view(joints.data(), joints.size(), 34962), 5121, n, "VEC4"));
            attributes += ",\"WEIGHTS_0\":" + std::to_string(g.floats(weights, 4, "VEC4", 34962));
        }
        const std::vector<uint32_t> &indices = mesh.indices;
        int index_accessor;
        if (n <= 65535) {
            std::vector<uint16_t> narrow(indices.begin(), indices.end());
            index_accessor = g.accessor(g.view(narrow.data(), narrow.size() * 2, 34963), 5123, narrow.size(), "SCALAR");
        } else {
            index_accessor = g.accessor(g.view(indices.data(), indices.size() * 4, 34963), 5125, indices.size(), "SCALAR");
        }

        // Leaves, hair and cloth are often there twice over the same vertices, the
        // second time wound the other way to be seen from behind. Blender keeps
        // one face of such a pair, so how many there were is noted.
        std::set<std::array<uint32_t, 3>> corners;
        for (size_t t = 0; t + 2 < indices.size(); t += 3) {
            std::array<uint32_t, 3> triangle = {indices[t], indices[t + 1], indices[t + 2]};
            std::sort(triangle.begin(), triangle.end());
            corners.insert(triangle);
        }
        size_t repeated = indices.size() / 3 - corners.size();

        // Further frames, as what each adds to the first.
        std::vector<std::string> targets, target_names;
        std::vector<int> times;
        for (const BrfFrame &frame : mesh.frames) {
            times.push_back(frame.time);
            if (frame.positions.size() != (size_t)mesh.positions * 3 || frame.normals.size() != n * 3) {
                warn("mesh %s: the frame at %d has %zu positions and %zu normals where %u and %zu are due; left out", mesh.name.c_str(), frame.time,
                     frame.positions.size() / 3, frame.normals.size() / 3, mesh.positions, n);
                continue;
            }
            std::vector<float> moved(n * 3), turned(n * 3);
            for (size_t i = 0; i < n; i++) {
                float normal[3];
                to_gltf(&frame.positions[(size_t)mesh.position_of[i] * 3], &moved[i * 3]);
                to_gltf(&frame.normals[i * 3], normal);
                unit(normal, &normals[i * 3]);
                for (int k = 0; k < 3; k++) moved[i * 3 + (size_t)k] -= positions[i * 3 + (size_t)k], turned[i * 3 + (size_t)k] = normal[k] - normals[i * 3 + (size_t)k];
            }
            targets.push_back("{\"POSITION\":" + std::to_string(g.floats(moved, 3, "VEC3", 34962, true)) + ",\"NORMAL\":" + std::to_string(g.floats(turned, 3, "VEC3", 34962)) + "}");
            target_names.push_back("\"frame_" + std::to_string(frame.time) + "\"");
        }

        const std::string &file = library.files[(size_t)from.file];
        std::string primitive = "{\"attributes\":{" + attributes + "},\"indices\":" + std::to_string(index_accessor) + ",\"mode\":4";
        if (!mesh.material.empty()) primitive += ",\"material\":" + std::to_string(material(mesh.material));
        if (!targets.empty()) primitive += ",\"targets\":" + joined(targets);
        std::string extras = "{\"brf_flags\":" + hex(mesh.flags) + ",\"brf_material\":" + quoted(mesh.material) + ",\"brf_file\":" + quoted(file.substr(file.rfind('/') + 1)) +
                             ",\"brf_version\":" + std::to_string(from.version) + ",\"brf_vertex_size\":" + std::to_string(mesh.vertex_size) +
                             ",\"brf_positions\":" + std::to_string(mesh.positions) + ",\"brf_repeated_faces\":" + std::to_string(repeated) + ",\"brf_frame_times\":" + list_of(times.data(), times.size());
        if (!targets.empty()) extras += ",\"targetNames\":" + joined(target_names);
        g.meshes.push_back("{\"name\":" + quoted(mesh.name) + ",\"primitives\":[" + primitive + "}],\"extras\":" + extras + "}}");
        std::string node = "{\"name\":" + quoted(mesh.name) + ",\"mesh\":" + std::to_string(g.meshes.size() - 1);
        if (skinned) node += ",\"skin\":0";
        g.scene.push_back(std::to_string(g.nodes.size()));
        g.nodes.push_back(node + "}");
    }

    // The frames and length of an action: from the module's actions.txt, or spelt out.
    bool find_action(const std::string &action, std::string *animation, double *from, double *to, double *seconds)
    {
        char name[200];
        if (sscanf(action.c_str(), "%199[^:]:%lf:%lf:%lf", name, from, to, seconds) == 4 && *seconds > 0) {
            *animation = lower(name);
            return true;
        }
        int sequence = 1;
        std::string wanted = action;
        size_t colon = action.find(':');
        if (colon != std::string::npos) wanted = action.substr(0, colon), sequence = atoi(action.c_str() + colon + 1);
        FILE *f = fopen((root + "/Modules/" + module + "/actions.txt").c_str(), "r");
        if (!f) return false;
        // After the count on the first line, each action is a name, two numbers of
        // flags and how many sequences; each sequence a length in seconds, an
        // animation, its first and last frame, and eight numbers more.
        char word[256];
        int count = 0, sequences = 0;
        bool found = false;
        if (fscanf(f, "%d", &count) == 1) {
            while (!found && fscanf(f, "%255s %*s %*s %d", word, &sequences) == 2) {
                for (int s = 1; s <= sequences; s++) {
                    char anim[256];
                    double length, first, last;
                    if (fscanf(f, "%lf %255s %lf %lf %*s %*s %*s %*s %*s %*s", &length, anim, &first, &last) != 4) break;
                    if (wanted == word && s == sequence) *animation = lower(anim), *from = first, *to = last, *seconds = length, found = true;
                }
            }
        }
        fclose(f);
        return found && *seconds > 0;
    }

    bool add_animation(const std::string &action, const Rig &rig, int first_node)
    {
        std::string name;
        double from, to, seconds;
        if (!find_action(action, &name, &from, &to, &seconds)) {
            warn("no action named %s in %s's actions.txt (and it is not animation:from:to:seconds)", action.c_str(), module.c_str());
            return false;
        }
        auto it = library.animations.find(name);
        if (it == library.animations.end()) {
            warn("no resource file has an animation named %s", name.c_str());
            return false;
        }
        const BrfAnimation &an = it->second;
        const BrfSkeleton &sk = *rig.skeleton;
        if (an.bones.size() != sk.bones.size()) {
            warn("animation %s moves %zu bones, skeleton %s has %zu", name.c_str(), an.bones.size(), sk.name.c_str(), sk.bones.size());
            return false;
        }
        // The frames an action shows are those it has keys at between its ends, and the ends themselves.
        auto moments = [from, to](const std::vector<int> &keyed) {
            std::vector<double> frames = {from};
            for (int f : keyed)
                if (f > std::min(from, to) && f < std::max(from, to)) frames.push_back(f);
            if (to != from) frames.push_back(to);
            std::sort(frames.begin(), frames.end(), [from, to](double a, double b) { return to >= from ? a < b : a > b; });
            frames.erase(std::unique(frames.begin(), frames.end()), frames.end());
            return frames;
        };
        auto times_of = [&](const std::vector<double> &frames) {
            std::vector<float> times;
            for (double f : frames) times.push_back(to != from ? (float)((f - from) / (to - from) * seconds) : 0.0f);
            float last = times.back();
            return g.accessor(g.view(times.data(), times.size() * 4), 5126, times.size(), "SCALAR", ",\"min\":[0],\"max\":[" + number(last) + "]");
        };
        std::vector<std::string> samplers, channels;
        auto channel = [&](int input, int output, size_t bone, const char *path) {
            channels.push_back("{\"sampler\":" + std::to_string(samplers.size()) + ",\"target\":{\"node\":" + std::to_string(first_node + (int)bone) + ",\"path\":\"" + path + "\"}}");
            samplers.push_back("{\"input\":" + std::to_string(input) + ",\"output\":" + std::to_string(output) + ",\"interpolation\":\"LINEAR\"}");
        };
        for (size_t b = 0; b < sk.bones.size(); b++) {
            // A bone the animation has keys for is turned as they say, in place of its turn at rest.
            const std::vector<BrfAnimation::Turn> &keys = an.bones[b];
            if (keys.empty()) continue;
            std::vector<int> keyed;
            for (const BrfAnimation::Turn &k : keys) keyed.push_back(k.frame);
            std::vector<double> frames = moments(keyed);
            std::vector<float> turns;
            for (double frame : frames) {
                size_t k = 0;
                while (k + 1 < keys.size() && keys[k + 1].frame <= frame) k++;
                size_t k1 = std::min(k + 1, keys.size() - 1);
                double span = keys[k1].frame - keys[k].frame, t = span > 0 ? std::min(1.0, std::max(0.0, (frame - keys[k].frame) / span)) : 0;
                double q[4], dot = 0;
                for (int i = 0; i < 4; i++) dot += keys[k].q[i] * keys[k1].q[i];
                for (int i = 0; i < 4; i++) q[i] = keys[k].q[i] * (1 - t) + keys[k1].q[i] * (dot < 0 ? -t : t);
                double len = sqrt(std::max(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3], 1e-12));
                for (int i = 0; i < 4; i++) q[i] /= len;
                double game[9], turned[9], out[4];
                rotation_of(q, game);
                turn_to_gltf(game, turned);
                quaternion_of(turned, out);
                // Of the two ways of writing a turn, the one nearer the last, so that what lies between is the short way round.
                if (!turns.empty()) {
                    const float *last = &turns[turns.size() - 4];
                    if (last[0] * out[0] + last[1] * out[1] + last[2] * out[2] + last[3] * out[3] < 0)
                        for (double &v : out) v = -v;
                }
                for (double v : out) turns.push_back((float)v);
            }
            channel(times_of(frames), g.accessor(g.view(turns.data(), turns.size() * 4), 5126, frames.size(), "VEC4"), b, "rotation");
        }
        for (size_t b = 0; b < sk.bones.size() && !an.root.empty(); b++) {
            // The root is also moved, on top of its place at rest.
            if (sk.bones[b].parent >= 0) continue;
            std::vector<int> keyed;
            for (const BrfAnimation::Move &m : an.root) keyed.push_back(m.frame);
            std::vector<double> frames = moments(keyed);
            std::vector<float> moves;
            for (double frame : frames) {
                size_t k = 0;
                while (k + 1 < an.root.size() && an.root[k + 1].frame <= frame) k++;
                size_t k1 = std::min(k + 1, an.root.size() - 1);
                double span = an.root[k1].frame - an.root[k].frame, t = span > 0 ? std::min(1.0, std::max(0.0, (frame - an.root[k].frame) / span)) : 0;
                double game[3], out[3];
                for (int i = 0; i < 3; i++) game[i] = sk.bones[b].place[i] + an.root[k].by[i] * (1 - t) + an.root[k1].by[i] * t;
                to_gltf(game, out);
                for (double v : out) moves.push_back((float)v);
            }
            channel(times_of(frames), g.accessor(g.view(moves.data(), moves.size() * 4), 5126, frames.size(), "VEC3"), b, "translation");
        }
        if (channels.empty()) {
            warn("animation %s has no keys", name.c_str());
            return false;
        }
        char extras[300];
        snprintf(extras, sizeof extras, "{\"brf_animation\":%s,\"brf_first_frame\":%s,\"brf_last_frame\":%s,\"brf_seconds\":%s}", quoted(an.name).c_str(), number(from).c_str(),
                 number(to).c_str(), number(seconds).c_str());
        size_t colon = action.find(':');
        g.animations.push_back("{\"name\":" + quoted(colon == std::string::npos ? action : action.substr(0, colon)) + ",\"samplers\":" + joined(samplers) +
                               ",\"channels\":" + joined(channels) + ",\"extras\":" + extras + "}");
        return true;
    }

    bool write(const std::string &path)
    {
        std::string json = "{\"asset\":{\"version\":\"2.0\",\"generator\":\"warband-metal gltf_export\",\"extras\":{\"brf_space\":\"gltf (x, y, z) = game (-x, z, y); uv (u, v) = game (u, 1 - v); tangent w = -game handedness; normal map green inverted\"}}"
                           ",\"scene\":0,\"scenes\":[{\"nodes\":" + joined(g.scene) + "}],\"nodes\":" + joined(g.nodes) + ",\"meshes\":" + joined(g.meshes);
        const std::pair<const char *, const std::vector<std::string> *> lists[] = {{"materials", &g.materials}, {"textures", &g.textures},     {"images", &g.images},
                                                                                   {"skins", &g.skins},         {"animations", &g.animations}, {"accessors", &g.accessors},
                                                                                   {"bufferViews", &g.views}};
        for (auto &list : lists) // an empty list is not allowed
            if (!list.second->empty()) json += std::string(",\"") + list.first + "\":" + joined(*list.second);
        g.bin.resize((g.bin.size() + 3) & ~(size_t)3, 0);
        json += ",\"buffers\":[{\"byteLength\":" + std::to_string(g.bin.size()) + "}]}";
        json.resize((json.size() + 3) & ~(size_t)3, ' ');
        // A header of three numbers, then two chunks, each a length, a kind and its bytes, padded to a multiple of four.
        const uint32_t header[3] = {0x46546c67, 2, (uint32_t)(12 + 8 + json.size() + 8 + g.bin.size())};
        const uint32_t json_chunk[2] = {(uint32_t)json.size(), 0x4e4f534a}, bin_chunk[2] = {(uint32_t)g.bin.size(), 0x004e4942};
        FILE *f = fopen(path.c_str(), "wb");
        if (!f) return false;
        bool ok = fwrite(header, 4, 3, f) == 3 && fwrite(json_chunk, 4, 2, f) == 2 && fwrite(json.data(), 1, json.size(), f) == json.size() &&
                  fwrite(bin_chunk, 4, 2, f) == 2 && fwrite(g.bin.data(), 1, g.bin.size(), f) == g.bin.size();
        return fclose(f) == 0 && ok;
    }
};

} // namespace

int main(int argc, char **argv)
{
    Exporter ex;
    std::string name, out, skeleton, action;
    const char *list = nullptr;
    int lod = 0;
    bool usage = argc < 2;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--list") && i + 1 < argc) list = argv[++i];
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--lod") && i + 1 < argc) lod = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--skeleton") && i + 1 < argc) skeleton = argv[++i];
        else if (!strcmp(argv[i], "--anim") && i + 1 < argc) action = argv[++i];
        else if (!strcmp(argv[i], "--module") && i + 1 < argc) ex.module = argv[++i];
        else if (!strcmp(argv[i], "--no-textures")) ex.textures = false;
        else if (!strcmp(argv[i], "--textures") && i + 1 < argc) ex.texture_place = argv[++i];
        else if (!strcmp(argv[i], "--check")) ex.check = true;
        else if (argv[i][0] != '-' && name.empty()) name = argv[i];
        else usage = true;
    }
    if (!ex.texture_place.empty()) { // given as the file is to find it: from where the file lies
        size_t slash = std::string(out).rfind('/');
        ex.texture_folder = ex.texture_place[0] == '/' ? ex.texture_place : (slash == std::string::npos ? std::string(".") : std::string(out).substr(0, slash)) + "/" + ex.texture_place;
    }
    if (usage || (name.empty() && !list)) {
        fprintf(stderr, "usage: gltf_export <mesh> [-o out.glb] [--lod n] [--skeleton name] [--anim action | --anim animation:from:to:seconds]\n"
                        "                   [--module name] [--no-textures | --textures folder] [--check]\n       gltf_export --list word\n");
        return 2;
    }
    ex.root = game_folder();
    // The module last played, as the game's launcher noted it; Native otherwise.
    if (ex.module.empty()) {
        const char *home = getenv("HOME");
        if (FILE *f = fopen((std::string(home ? home : "") + "/Library/Application Support/MBWarband/last_module_warband").c_str(), "r")) {
            char text[512];
            size_t n = fread(text, 1, sizeof text - 1, f);
            fclose(f);
            ex.module.assign(text, n);
            while (!ex.module.empty() && isspace((unsigned char)ex.module.back())) ex.module.pop_back();
        }
        if (ex.module.empty()) ex.module = "Native";
    }
    ex.read_library();
    if (ex.library.files.empty()) {
        fprintf(stderr, "no resource files under %s (WBM_GAME names the game's folder)\n", ex.root.c_str());
        return 1;
    }

    if (list) {
        std::string want = lower(list);
        int found = 0;
        for (auto &m : ex.library.meshes) {
            if (m.first.find(".lod") != std::string::npos || m.first.find(want) == std::string::npos) continue;
            const std::string &file = ex.library.files[(size_t)m.second.file];
            printf("%-44s %s\n", m.first.c_str(), file.substr(file.rfind('/') + 1).c_str());
            found++;
        }
        for (auto &m : ex.library.materials)
            if (m.first.find(want) != std::string::npos)
                printf("%-44s material: shader %s, textures %s %s %s %s\n", m.first.c_str(), m.second.shader.c_str(), m.second.diffuse.c_str(), m.second.diffuse2.c_str(),
                       m.second.normal.c_str(), m.second.specular.c_str()), found++;
        for (auto &s : ex.library.skeletons)
            if (s.first.find(want) != std::string::npos) printf("%-44s skeleton: %zu bones\n", s.first.c_str(), s.second.bones.size()), found++;
        for (auto &a : ex.library.animations)
            if (a.first.find(want) != std::string::npos) printf("%-44s animation: %zu bones\n", a.first.c_str(), a.second.bones.size()), found++;
        return found ? 0 : 1;
    }

    // The mesh, with the further meshes that belong to it (name.1, name.2, ...), or one of its simpler versions.
    std::string base = lower(name), suffix = lod > 0 ? ".lod" + std::to_string(lod) : std::string();
    std::vector<BrfMesh> meshes;
    std::vector<BrfLibrary::MeshRef> refs;
    for (const std::string &part : ex.library.parts_of(base, lod)) {
        BrfMesh mesh;
        if (ex.library.load_mesh(part, &mesh) && !mesh.vertices.empty() && !mesh.indices.empty()) {
            meshes.push_back(std::move(mesh));
            refs.push_back(ex.library.meshes.find(part)->second);
        }
    }
    if (meshes.empty()) {
        fprintf(stderr, "no mesh named %s%s (gltf_export --list <word> finds names)\n", name.c_str(), suffix.c_str());
        return 1;
    }

    // A rigged mesh does not say which skeleton it was made for: the one whose
    // bones lie nearest the vertices that hang on them, of those with bones enough.
    bool skinned = false;
    for (const BrfMesh &mesh : meshes) skinned = skinned || mesh.skinned;
    Rig rig;
    if (!skinned && !action.empty() && skeleton.empty()) {
        fprintf(stderr, "%s hangs on no bones, so nothing says whose action %s is; --skeleton names a skeleton to write beside it\n", name.c_str(), action.c_str());
        return 1;
    }
    if (skinned || !action.empty()) {
        double least = 1e30;
        for (auto &sk : ex.library.skeletons) {
            if (!skeleton.empty() && sk.first != lower(skeleton)) continue;
            Rig candidate = rig_of(sk.second);
            double misfit = ex.misfit(candidate, meshes);
            if (ex.check) fprintf(stderr, "check skeleton %s: %zu bones, vertices %.3f m from their bones\n", sk.second.name.c_str(), sk.second.bones.size(), misfit);
            if (misfit >= 0 && misfit < least) least = misfit, rig = candidate;
        }
        if (!rig.skeleton) {
            if (skeleton.empty()) fprintf(stderr, "no skeleton has the bones %s hangs on\n", name.c_str());
            else fprintf(stderr, "skeleton %s is not there, or has not the bones %s hangs on\n", skeleton.c_str(), name.c_str());
            return 1;
        }
    }
    int first_bone = rig.skeleton ? ex.add_skin(rig) : 0;
    for (size_t i = 0; i < meshes.size(); i++) {
        if (ex.check) ex.report(meshes[i]);
        ex.add_mesh(meshes[i], refs[i], rig.skeleton ? &rig : nullptr);
    }
    if (!action.empty() && !ex.add_animation(action, rig, first_bone)) return 1;

    if (out.empty()) out = base + suffix + ".glb";
    if (!ex.write(out)) {
        fprintf(stderr, "could not write %s\n", out.c_str());
        return 1;
    }
    size_t vertices = 0, triangles = 0;
    for (const BrfMesh &mesh : meshes) vertices += mesh.vertices.size(), triangles += mesh.indices.size() / 3;
    printf("%s: %zu mesh%s, %zu vertices, %zu triangles, %zu material%s, %zu texture%s", out.c_str(), meshes.size(), meshes.size() == 1 ? "" : "es", vertices, triangles,
           ex.g.materials.size(), ex.g.materials.size() == 1 ? "" : "s", ex.g.images.size(), ex.g.images.size() == 1 ? "" : "s");
    if (rig.skeleton) printf(", skeleton %s (%zu bones)", rig.skeleton->name.c_str(), rig.skeleton->bones.size());
    if (!ex.g.animations.empty()) printf(", animation %s", action.c_str());
    printf("\n");
    return 0;
}
