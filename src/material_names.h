#pragma once

// What a texture of the game shows, judged from the name of its file. Shared
// by the renderer and tools/texture_tool.mm.

#include <cctype>
#include <string>

enum MaterialClass { MAT_UNKNOWN, MAT_METAL, MAT_CLOTH, MAT_LEATHER, MAT_SKIN, MAT_WOOD, MAT_STONE, MAT_HORSE, MAT_COUNT };
static const char *const k_material_names[MAT_COUNT] = {"unknown", "metal", "cloth", "leather", "skin and hair", "wood", "stone and ground", "horse"};

inline int classify_texture_name(std::string name)
{
    for (char &c : name) c = (char)tolower((unsigned char)c);
    // Words that mark each class. A name with words of more than one class
    // (a surcoat over mail) is a mixture, and is left to the specular mask.
    static const struct {
        int cls;
        const char *words[32];
    } table[] = {
        {MAT_METAL, {"plate", "mail", "helmet", "helm", "lamellar", "scale", "gauntlet", "greave", "sword", "blade", "axe", "mace", "lance", "spear", "metal", "iron", "steel", "brass", "chain", "coif", "bascinet", "cuirass", "weapon"}},
        {MAT_CLOTH, {"dress", "tunic", "shirt", "cloth", "coat", "surcoat", "vest", "costume", "outfit", "robe", "padded", "ragged", "banner", "heraldic", "gambeson", "hood", "cape", "cloak", "linen", "turban", "veil", "wimple", "blouse", "peasant", "lady", "tabard"}},
        {MAT_LEATHER, {"leather", "boot", "saddle", "hide", "fur", "nomad", "belt", "glove", "shoe", "pelt"}},
        {MAT_SKIN, {"face", "body", "hand", "skin", "hair", "beard"}},
        {MAT_WOOD, {"plank", "wood", "bark", "tree", "trunk", "bow", "barrel", "cart", "fence", "palisade", "timber"}},
        {MAT_STONE, {"stone", "wall", "rock", "stucco", "ground", "terrain", "roof", "building", "village", "brick", "earth", "road", "sand", "snow", "grass", "castle", "town", "floor", "mud", "steppe", "desert", "plaster"}},
        {MAT_HORSE, {"horse", "courser", "hunter", "steed"}},
    };
    if (name.find("warhorse") != std::string::npos) return MAT_UNKNOWN; // a horse in armour
    int found = MAT_UNKNOWN, classes = 0;
    for (auto &row : table) {
        for (const char *w : row.words) {
            if (w && name.find(w) != std::string::npos) {
                found = row.cls;
                classes++;
                break;
            }
        }
    }
    return classes == 1 ? found : MAT_UNKNOWN;
}

