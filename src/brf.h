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

struct BrfMesh {
    std::string name, material;
    uint32_t flags = 0;
    bool skinned = false, has_tangents = false;
    std::vector<BrfVertex> vertices;
    std::vector<uint32_t> indices;
};

struct BrfMaterial {
    std::string name, shader;
    uint32_t flags = 0;
    std::string diffuse, diffuse2, normal, environment, specular; // texture names; "none" when absent
    float shininess = 0, specular_colour[3] = {0, 0, 0};
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
    std::vector<std::string> problems;            // files that could not be read to their end

    // Reads every .brf file in the folder; meshes are indexed, not kept.
    void scan(const std::string &folder);
    bool load_mesh(const std::string &name, BrfMesh *mesh) const;
};
