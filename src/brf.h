#pragma once

// Reads the game's resource files (CommonRes/*.brf and a module's
// Resource/*.brf): meshes, materials and the shader table. Nothing here
// depends on the renderer.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct BrfVertex {
    float position[3], normal[3], tangent[3];
    float handedness; // of the tangent frame: +1 or -1
    uint8_t colour[4]; // as stored: blue, green, red, alpha
    float uv[2];
    float weight[4]; // bone weights, all zero for a rigid mesh
    float bone[4];
};

// A further pose of a vertex-animated mesh (a face's features, a bow drawn):
// every position anew and a normal for every vertex, at a time the game picks it by.
struct BrfFrame {
    int time = 0;
    std::vector<float> positions, normals; // three numbers each
};

struct BrfMesh {
    std::string name, material;
    uint32_t flags = 0;
    bool skinned = false, has_tangents = false;
    std::vector<BrfVertex> vertices;
    std::vector<uint32_t> indices;
    // Vertices that differ only in normal, colour or texture coordinate share
    // a position; rigging and frames go by position.
    std::vector<uint32_t> position_of; // for each vertex
    uint32_t positions = 0;
    int vertex_size = 0; // bytes of a vertex record in the file: 28, 36 or 41
    std::vector<BrfFrame> frames;
};

struct BrfMaterial {
    std::string name, shader;
    uint32_t flags = 0;
    std::string diffuse, diffuse2, normal, environment, specular; // texture names; "none" when absent
    float shininess = 0, specular_colour[3] = {0, 0, 0};
};

struct BrfSkeleton {
    struct Bone {
        int parent; // -1 for the root
        std::string name;
        float axes[9];  // the bone's x, y and z axes in its parent's space, at rest
        float place[3]; // and its place there
        uint32_t flags = 0;
    };
    std::string name;
    std::vector<Bone> bones;
};

// Rotations of each bone in its parent's space at numbered frames, and where
// the root is moved to. One animation usually holds many actions end to end;
// which frames make up an action is the module's business (actions.txt).
struct BrfAnimation {
    struct Turn {
        int frame;
        float q[4]; // x, y, z, w
    };
    struct Move {
        int frame;
        float by[3];
    };
    std::string name;
    std::vector<std::vector<Turn>> bones;
    std::vector<Move> root;
};

struct BrfLibrary {
    // Names are lower case. A mesh name maps to the file and place it was found at.
    struct MeshRef {
        int file;
        size_t offset;
        int version;
    };
    std::vector<std::string> files;
    std::map<std::string, MeshRef> meshes;
    std::map<std::string, BrfMaterial> materials;
    std::map<std::string, std::string> techniques; // shader name -> technique name
    std::map<std::string, BrfSkeleton> skeletons;
    std::map<std::string, BrfAnimation> animations;
    std::vector<std::string> problems;            // files that could not be read to their end

    // Reads every .brf file in the folder; meshes are indexed, not kept.
    void scan(const std::string &folder);
    bool load_mesh(const std::string &name, BrfMesh *mesh) const;
    // A model is a mesh and, where it wears more than one material, further meshes
    // called name.1, name.2 and so on (any numbers, with gaps); its simpler
    // versions for seeing from afar are name.lod1 ... and name.lod1.2 ...
    // The names of all the meshes of a model at one level (0: the model itself),
    // in the order of their numbers; and how many simpler levels it has.
    std::vector<std::string> parts_of(const std::string &base, int lod) const;
    int lods_of(const std::string &base) const;
    // Where a mesh's record lies: its file, and the bytes of the file it takes up.
    bool mesh_extent(const std::string &name, std::string *path, size_t *begin, size_t *end) const;
};

// A mesh as a record of a resource file, to go where one read by load_mesh
// lay. A vertex's place is that of its position_of entry, and its weights
// those of the first vertex there; `vertex_size` picks the layout.
std::vector<uint8_t> brf_mesh_record(const BrfMesh &mesh);
