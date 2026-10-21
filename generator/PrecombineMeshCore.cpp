// Faster Objects (SkyrimPrecombines): precombined meshes and Hi-Z occlusion culling for Skyrim.
// Copyright (C) 2026 Asciim0v
//
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with this program. If not, see
// <https://www.gnu.org/licenses/>.

// Precombine geometry baker core (library). nifly is GPL-3.0: this library is linked only into the separate
// PrecombineMesh executable and its tests, never into the SKSE plugin (the runtime bake launches the executable).
#include "PrecombineMeshCore.h"
#include <NifFile.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

using namespace nifly;
using json = nlohmann::json;
namespace fs = std::filesystem;

static void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
static Vector3 vec(const json& a) { return {a.at(0), a.at(1), a.at(2)}; }
static json vec(const Vector3& a) { return {a.x, a.y, a.z}; }
static std::string hex(uint64_t v, int width = 16) {
    std::ostringstream out; out << std::hex << std::setw(width) << std::setfill('0') << v; return out.str();
}
// Scene paths are UTF-8 both ways. path::string() is the ANSI code page: its bytes fail JSON's UTF-8 check, and it
// throws for characters that page lacks, so one non-ASCII folder name failed every runtime bake.
static std::string utf8(const fs::path& path) { const auto text = path.u8string(); return std::string(text.begin(), text.end()); }
static fs::path pathOf(const std::string& text) { return fs::path(std::u8string(text.begin(), text.end())); }
static std::string hashFile(const fs::path& file) {
    std::ifstream in(file, std::ios::binary); require(bool(in), "Cannot read " + utf8(file));
    uint64_t h = 14695981039346656037ull;
    char buf[65536];
    while (in) { in.read(buf, sizeof buf); for (std::streamsize i = 0; i < in.gcount(); ++i) h = (h ^ static_cast<unsigned char>(buf[i])) * 1099511628211ull; }
    require(in.eof(), "Read failed: " + utf8(file)); return hex(h);
}

static MatTransform referenceTransform(const json& ref) {
    const auto a = vec(ref.at("rotation"));
    const float sx = std::sin(a.x), sy = std::sin(a.y), sz = std::sin(a.z);
    const float cx = std::cos(a.x), cy = std::cos(a.y), cz = std::cos(a.z);
    // Same convention as CommonLibSSE NiMatrix3::SetEulerAnglesXYZ.
    MatTransform t;
    t.rotation = Matrix3(cy*cz, cy*sz, -sy,
        sx*sy*cz-cx*sz, sx*sy*sz+cx*cz, sx*cy,
        cx*sy*cz+sx*sz, cx*sy*sz-sx*cz, cx*cy);
    t.translation = vec(ref.at("position")); t.scale = ref.at("scale");
    return t;
}
static json transformJson(const MatTransform& t) {
    return {{"position", vec(t.translation)}, {"scale", t.scale},
        {"matrix", {vec(t.rotation[0]), vec(t.rotation[1]), vec(t.rotation[2])}}};
}
static bool sane(const MatTransform& t) {
    if (!std::isfinite(t.scale) || t.scale <= 0) return false;
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(t.translation[i])) return false;
        for (int j = 0; j < 3; ++j) {
            if (!std::isfinite(t.rotation[i][j])) return false;
            float dot = 0;
            for (int k = 0; k < 3; ++k) dot += t.rotation[i][k] * t.rotation[j][k];
            if (std::abs(dot - (i == j ? 1.f : 0.f)) > 0.001f) return false;
        }
    }
    return std::abs(t.rotation.Determinant() - 1.f) < 0.001f;
}
static std::string bytes(NiObject& block, NiHeader& header) {
    std::ostringstream out(std::ios::binary);
    NiOStream stream(&out, &header); block.Put(stream); return out.str();
}
// nifly stores each component as uint8(round((x+1)/2*255)) without clamping.
// Vanilla vectors decode up to ~1% long, so a rotated component can exceed
// 1+1/255 and wrap 255->0, flipping the vector. Store unit, in-range vectors.
static Vector3 unitVector(Vector3 v) {
    v.Normalize();
    v.x = std::clamp(v.x, -1.f, 1.f); v.y = std::clamp(v.y, -1.f, 1.f); v.z = std::clamp(v.z, -1.f, 1.f);
    return v;
}

// Fallout 4 Creation Kit default small-object radius (RVA 0x3BD5A90 = 100.0). An exterior source
// fades as a small object when its whole model's bound radius x reference scale is below it.
constexpr double kSmallObjectRadius = 100.0;

// Smallest sphere holding both spheres (NiBound::Merge).
static BoundingSphere mergeBounds(const BoundingSphere& a, const BoundingSphere& b) {
    const Vector3 d = b.center - a.center;
    const float distance = d.length();
    if (distance + b.radius <= a.radius) return a;
    if (distance + a.radius <= b.radius) return b;
    const float radius = (distance + a.radius + b.radius) * 0.5f;
    return {a.center + d * ((radius - a.radius) / distance), radius};
}
// Bound of the whole model in root space: every shape reachable from the root (including
// shapes that cannot merge), merged in block order. All shapes of one reference share it.
static float modelBoundRadius(NifFile& nif) {
    std::optional<BoundingSphere> total;
    for (auto* shape : nif.GetShapes()) {
        MatTransform toRoot = shape->transform;
        NiAVObject* child = shape;
        std::set<NiAVObject*> visited;
        while (child && child != nif.GetRootNode() && visited.insert(child).second) {
            auto* parent = nif.GetParentNode(child);
            if (parent) toRoot = parent->transform.ComposeTransforms(toRoot);
            child = parent;
        }
        if (!child || child != nif.GetRootNode()) continue;
        const auto bounds = shape->GetBounds();
        const BoundingSphere sphere(toRoot.ApplyTransform(bounds.center), bounds.radius * toRoot.scale);
        total = total ? mergeBounds(*total, sphere) : sphere;
    }
    return total ? total->radius : 0.f;
}
static std::string lower(std::string text) {
    for (auto& c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return text;
}
// Plugin names end up in chunk and manifest file names: plain names with a plugin extension only.
static bool pluginName(const std::string& name) {
    if (name.size() < 5 || name.size() > 255) return false;
    const auto ext = lower(name.substr(name.size() - 4));
    if (ext != ".esm" && ext != ".esp" && ext != ".esl") return false;
    if (!std::isalnum(static_cast<unsigned char>(name.front()))) return false;
    for (const char c : name)
        if (!std::isalnum(static_cast<unsigned char>(c)) && (c == '\0' || !std::strchr(" _'()+,&!.-", c))) return false;
    return true;
}
// A form in a scene or manifest: a number (a Skyrim.esm form ID, as in old manifests) or a plugin-relative
// {"plugin": file, "id": local form ID}, also accepted as ["file", local form ID] (the runtime's resolveForm).
struct FormRef { std::string plugin; uint32_t id = 0; bool relative = false; };
static FormRef formRef(const json& value) {
    if (value.is_number_integer()) {
        const auto id = value.get<int64_t>();
        require(id > 0 && id < 0x1000000, "Invalid form reference");
        return {"Skyrim.esm", static_cast<uint32_t>(id), false};
    }
    FormRef form{{}, 0, true};
    int64_t id = 0;
    if (value.is_object() && value.size() == 2 && value.contains("plugin") && value.contains("id")
            && value.at("plugin").is_string() && value.at("id").is_number_integer()) {
        form.plugin = value.at("plugin").get<std::string>(); id = value.at("id").get<int64_t>();
    } else if (value.is_array() && value.size() == 2 && value[0].is_string() && value[1].is_number_integer()) {
        form.plugin = value[0].get<std::string>(); id = value[1].get<int64_t>();
    } else throw std::runtime_error("Invalid form reference");
    require(pluginName(form.plugin) && id > 0 && id < 0x1000000, "Invalid form reference");
    form.id = static_cast<uint32_t>(id);
    return form;
}
static std::string formKey(const json& value) {
    const auto form = formRef(value);
    return lower(form.plugin) + ':' + hex(form.id, 6);
}
// Manifest and chunk file stem. A numeric cell keeps the original 8-digit form ID name (unchanged for every
// existing scene); a plugin-relative cell of Skyrim.esm uses the same name, and a cell of another plugin is
// "<plugin in lower case>_<6-digit local form ID>".
static std::string cellStem(const json& cell) {
    if (cell.is_number_integer()) return hex(cell.get<uint32_t>(), 8);
    const auto form = formRef(cell);
    return lower(form.plugin) == "skyrim.esm" ? hex(form.id, 8) : lower(form.plugin) + '_' + hex(form.id, 6);
}
// Interior scenes are schema 1 as before; exterior scenes are schema 2 with worldspace and grid. A numeric cell
// is a Skyrim.esm form ("plugin": "Skyrim.esm"); a plugin-relative cell names its own plugin.
static bool exteriorScene(const json& scene) {
    const auto& schema = scene.at("schema");
    const auto& cell = scene.at("cell");
    const bool source = cell.is_number_integer() ? scene.at("plugin") == "Skyrim.esm"
        : scene.at("plugin").is_string() && lower(scene.at("plugin").get<std::string>()) == lower(formRef(cell).plugin);
    if (source && schema == 1 && scene.at("interior") == true && !scene.contains("worldspace") && !scene.contains("grid"))
        return false;
    require(source && schema == 2 && scene.at("interior") == false
        && scene.contains("worldspace") && scene.contains("grid"), "Unsupported scene schema or source");
    const auto& world = scene.at("worldspace");
    const auto& grid = scene.at("grid");
    const auto validWorld = [&] {
        if (!world.is_object() || world.size() != 2 || !world.contains("form_id") || !world.contains("editor_id")
                || !world.at("editor_id").is_string()) return false;
        const auto& form = world.at("form_id");
        if (form.is_number_integer()) return form.get<int64_t>() > 0 && form.get<int64_t>() < 0x1000000;
        try { formRef(form); return true; } catch (const std::runtime_error&) { return false; }
    };
    require(validWorld() && grid.is_array() && grid.size() == 2 && grid[0].is_number_integer() && grid[1].is_number_integer(),
        "Invalid exterior worldspace or grid");
    return true;
}

// Use every serialized shader/alpha value, not just textures and flag bits.
// Normalize only non-rendering names and block indices before comparing.
// Lighting shader types that may merge (scene "shader_types"; default {0}, the plain lighting shader).
// 1 = environment map (Dwemer metal), 2 = glow map, 3 = parallax: identical serialized materials share
// one shader permutation, and the baker rotates normals/tangents/bitangents like any other vertex.
// Per thread: one bake reconfigures it, so in-process callers may bake on several threads at once.
static thread_local std::set<uint32_t> mergeableShaderTypes{0};

static std::string materialKey(NifFile& nif, BSTriShape& shape) {
    auto* shader = dynamic_cast<BSLightingShaderProperty*>(nif.GetShader(&shape));
    require(shader && mergeableShaderTypes.count(shader->GetShaderType()), "unsupported_shader");
    require(shader->controllerRef.IsEmpty() && shader->extraDataRefs.GetSize() == 0, "animated_material");
    require(!(shader->shaderFlags1 & ((1u<<1)|(1u<<2)|(1u<<12)|(1u<<15)|(1u<<26)|(1u<<27))), "special_shader_flags");
    // LOD landscape/objects, glow map, tree animation, HD LOD. Double-sided is
    // allowed: it is part of the serialized key, so merged sources all share it.
    require(!(shader->shaderFlags2 & ((1u<<1)|(1u<<2)|(1u<<6)|(1u<<29)|(1u<<31))), "animated_or_lod_shader");
    require(shader->alpha == 1.f && shader->refractionStrength == 0.f, "translucent_material");
    require(shader->name.get().empty() && shader->rootMaterialName.get().empty(), "external_material");
    auto* textures = nif.GetHeader().GetBlock(shader->textureSetRef);
    require(textures, "missing_textures");
    auto copy = *shader;
    copy.name.clear(); copy.rootMaterialName.clear(); copy.textureSetRef.Clear();
    std::string result = bytes(copy, nif.GetHeader()) + bytes(*textures, nif.GetHeader());
    if (shape.HasAlphaProperty()) {
        auto* alpha = nif.GetHeader().GetBlock<NiAlphaProperty>(shape.AlphaPropertyRef());
        require(alpha && !(alpha->flags & 1) && alpha->controllerRef.IsEmpty() && alpha->extraDataRefs.GetSize() == 0, "alpha_blend_or_controller");
        auto alphaCopy = *alpha; alphaCopy.name.clear();
        result += bytes(alphaCopy, nif.GetHeader());
    }
    result += ':' + std::to_string(static_cast<unsigned>(shape.vertexDesc.GetFlags()) & ~VF_FULLPREC);
    result += ':' + std::to_string(shape.flags);
    return result;
}

// Base alternate textures (TES5 STAT MODS) as the game swaps them at 3D load. A scene reference may carry
// "alternate_textures": [{"name3d", "index3d", "texture_set", "textures": [TX00..TX07]}]; the shapes they swap
// merge with the swapped textures, and their manifest source records the swap ("texture_swap": index).
// The game (SkyrimSE 1.5.97: TESModelTextureSwap apply 0x19BAA0, walker 0x19C5C0) keys the swaps by 3D index,
// skips index -1, walks the loaded 3D depth-first counting every geometry (particle systems too), and gives a
// lighting or grass shader a fresh BSShaderTextureSet built from the record (0x2D19A0: all nine slots,
// "Data\Textures\" + path, empty record slots stay empty). A swap to NullTextureSet (form 0x28) hides the shape.
// - Which shape: the game uses index3d only, but it is checked against name3d. In Skyrim.esm both pick the same
//   unique shape for 1,291 of 1,307 swaps. A swap is applied only when its name and its index (depth-first and block
//   order) pick one shape with that name. Otherwise every shape that either could pick stays original
//   ("alternate_texture_ambiguous"); two swaps on one shape conflict.
// - Which slots: texture set records list diffuse, normal, environment mask, glow, height, environment, multilayer,
//   backlight (TX00..TX07); BGSTextureSet's path getter (0x2D1B00) returns them in SSE NIF slot order: diffuse,
//   normal, glow, height, environment, environment mask, multilayer, backlight (vanilla NIFs: _n in 1, _g/_sk in 2,
//   _e in 4, _em/_m in 5). The game replaces the whole set; a swap is applied only when every texture the shape had
//   is replaced by a non-empty one, so the chunk renders exactly what the game shows ("alternate_texture_partial").
//   NIF paths carry "textures\" (the runtime compares paths without a leading data\ or textures\).
// - The swapped texture set must belong to that shape alone ("alternate_texture_shared_material").
constexpr size_t kTextureSetToNifSlot[8]{0, 1, 5, 2, 3, 4, 6, 7};
static bool particleSystem(NiObject* block) {
    if (!block) return false;
    const std::string type = block->GetBlockName();
    return type == "NiParticleSystem" || type == "NiMeshParticleSystem" || type == "BSStripParticleSystem";
}
struct TextureSwaps {
    std::unique_ptr<NifFile> nif;                     // a private copy of the model with the swaps applied
    std::map<const NiShape*, int> applied;            // shape -> index of its swap
    std::map<const NiShape*, std::string> rejected;   // shape -> skip reason
};
static std::string nifTexturePath(const std::string& path) {
    const auto low = lower(path);
    return low.rfind("textures\\", 0) == 0 || low.rfind("textures/", 0) == 0 ? path : "textures\\" + path;
}
// Geometry in the order the game counts it for index3d: shapes and particle systems, depth-first from the root.
static void depthFirstShapes(NifFile& nif, NiNode* node, std::vector<NiObject*>& out, std::set<NiObject*>& seen, int depth) {
    if (!node || depth > 64 || !seen.insert(node).second) return;
    for (auto& ref : node->childRefs) {
        auto* child = nif.GetHeader().GetBlock(ref);
        if (dynamic_cast<NiShape*>(child) || particleSystem(child)) { if (seen.insert(child).second) out.push_back(child); }
        else if (auto* childNode = dynamic_cast<NiNode*>(child)) depthFirstShapes(nif, childNode, out, seen, depth + 1);
    }
}
static TextureSwaps applyTextureSwaps(const fs::path& file, const json& swaps) {
    require(swaps.is_array() && !swaps.empty() && swaps.size() <= 64, "Invalid alternate textures");
    TextureSwaps result;
    result.nif = std::make_unique<NifFile>();
    require(result.nif->Load(file) == 0 && result.nif->IsValid(), "Cannot reload a model for its texture swaps");
    auto& nif = *result.nif;
    std::vector<NiObject*> dfs, blocks;
    std::set<NiObject*> seen;
    depthFirstShapes(nif, nif.GetRootNode(), dfs, seen, 0);
    for (uint32_t id = 0; id < nif.GetHeader().GetNumBlocks(); ++id) {
        auto* block = nif.GetHeader().GetBlock<NiObject>(id);
        if (dynamic_cast<NiShape*>(block) || particleSystem(block)) blocks.push_back(block);
    }
    const auto named = [](NiObject* block) -> std::string {
        auto* object = dynamic_cast<NiAVObject*>(block);
        return object ? object->name.get() : std::string();
    };
    std::map<const NiShape*, int> target;
    std::set<const NiShape*> conflicts;
    for (int k = 0; k < int(swaps.size()); ++k) {
        const auto& swap = swaps.at(size_t(k));
        require(swap.at("name3d").is_string() && swap.at("index3d").is_number_integer() && swap.at("textures").is_array()
            && swap.at("textures").size() == 8, "Invalid alternate textures");
        formRef(swap.at("texture_set"));
        for (const auto& path : swap.at("textures")) require(path.is_string() && path.get<std::string>().size() < 260, "Invalid alternate textures");
        const auto name = swap.at("name3d").get<std::string>();
        const auto index = swap.at("index3d").get<int64_t>();
        if (index < 0) continue;  // the game never applies a swap without a 3D index
        std::set<NiObject*> candidates;
        for (auto* s : dfs) if (named(s) == name) candidates.insert(s);
        for (auto* s : blocks) if (named(s) == name) candidates.insert(s);
        if (index < int64_t(dfs.size())) candidates.insert(dfs[size_t(index)]);
        if (index < int64_t(blocks.size())) candidates.insert(blocks[size_t(index)]);
        if (candidates.empty()) continue;  // neither rule picks a shape: the game swaps nothing
        auto* only = dynamic_cast<NiShape*>(*candidates.begin());
        if (candidates.size() != 1 || !only || named(only) != name) {
            for (auto* s : candidates) if (auto* shape = dynamic_cast<NiShape*>(s)) result.rejected[shape] = "alternate_texture_ambiguous";
            continue;
        }
        if (target.count(only)) conflicts.insert(only);
        target[only] = k;
    }
    for (auto* s : conflicts) result.rejected[s] = "alternate_texture_conflict";
    for (const auto& [constShape, k] : target) {
        if (result.rejected.count(constShape)) continue;
        auto* shape = const_cast<NiShape*>(constShape);
        auto* shader = dynamic_cast<BSLightingShaderProperty*>(nif.GetShader(shape));
        auto* textures = shader ? nif.GetHeader().GetBlock(shader->textureSetRef) : nullptr;
        if (!textures) { result.rejected[shape] = "alternate_texture_unsupported_shader"; continue; }
        size_t users = 0;
        for (auto* other : nif.GetShapes()) {
            auto* otherShader = dynamic_cast<BSLightingShaderProperty*>(nif.GetShader(other));
            users += otherShader == shader || (otherShader && nif.GetHeader().GetBlock(otherShader->textureSetRef) == textures);
        }
        if (users != 1) { result.rejected[shape] = "alternate_texture_shared_material"; continue; }
        std::vector<std::string> next;
        for (auto& texture : textures->textures) next.push_back(texture.get());
        std::set<size_t> written;
        bool partial = false;
        for (size_t t = 0; t < 8; ++t) {
            const auto path = swaps.at(size_t(k)).at("textures").at(t).get<std::string>();
            const size_t slot = kTextureSetToNifSlot[t];
            if (slot >= next.size()) { partial = partial || !path.empty(); continue; }
            written.insert(slot);
            if (path.empty()) partial = partial || !next[slot].empty();
            else next[slot] = nifTexturePath(path);
        }
        for (size_t slot = 0; slot < next.size(); ++slot) partial = partial || (!written.count(slot) && !next[slot].empty());
        if (partial) { result.rejected[shape] = "alternate_texture_partial"; continue; }
        size_t slot = 0;
        for (auto& texture : textures->textures) texture.get() = next[slot++];
        result.applied[shape] = k;
    }
    return result;
}

struct Source {
    NifFile* nif{};
    BSTriShape* shape{};
    json reference;
    json path;
    MatTransform world;
    MatTransform placement, toRoot;  // world == placement.ComposeTransforms(toRoot)
    int swap = -1;                   // index of the reference's alternate texture applied to this shape
};
struct Group {
    std::vector<Source> sources;
    size_t vertices = 0, triangles = 0;
    Vector3 origin;
    std::string fade;  // exterior only: "small" or "large"
    bool lightSplit = false;  // produced by halving a bin that touched too many lights
};

// Optional light budget. Skyrim SE 1.5.97 adds a light to a geometry when the light's sphere touches
// the geometry's world bound (|light - centre| - radius < light radius, BSLight 0x131D000), keeps
// lights in attach order (0x1310A30), and shades only the first seven of them in its single lighting
// pass (BSLightingShaderProperty::GetRenderPasses 0x12C6200 clamps the technique to min(n - 1, 7)).
// A merged chunk's bound is larger than any source's, so it touches more lights, and past seven the
// engine drops lights that do reach its geometry. Scene "light_budget" B with "lights" [[x, y, z, r]]
// (world space, radius = base + XRDS) keeps every chunk at B lights or fewer, which leaves 7 - B
// slots for torches and spells. Lights that touch only the bound add nothing: attenuation is zero
// at the radius.
struct LightBudget {
    int budget = -1;
    std::vector<std::array<double, 4>> spheres;
    size_t count(const BoundingSphere& local, const Vector3& origin) const {
        const double cx = double(local.center.x) + origin.x, cy = double(local.center.y) + origin.y,
                     cz = double(local.center.z) + origin.z;
        size_t n = 0;
        for (const auto& l : spheres) {
            const double dx = l[0] - cx, dy = l[1] - cy, dz = l[2] - cz;
            if (std::sqrt(dx * dx + dy * dy + dz * dz) - local.radius < l[3]) ++n;
        }
        return n;
    }
};

// The bound the bake stores for a group (relative to its origin): the same vertex positions in the
// same order, nifly's Miniball, then the same growth to the farthest vertex.
static BoundingSphere groupBound(const Group& group, bool exterior) {
    std::vector<Vector3> points;
    points.reserve(group.vertices);
    for (const auto& source : group.sources)
        for (const auto& input : source.shape->vertData) {
            Vector3 v = input.vert;
            if (!exterior) v = source.world.ApplyTransform(v) - group.origin;
            else v = source.placement.rotation * (source.toRoot.ApplyTransform(v) * source.placement.scale)
                    + (source.placement.translation - group.origin);
            points.push_back(v);
        }
    BoundingSphere bounds(points);
    float reach = 0.f;
    for (const auto& p : points) reach = std::max(reach, (p - bounds.center).length());
    if (!(reach <= bounds.radius + 0.05f)) bounds.radius = reach * 1.0001f + 0.01f;
    return bounds;
}

// Splits a group whose bound touches more than `budget` lights. Greedy sweep: along each axis, sources
// are ordered by their bound centres and the longest prefix that fits is found by binary search; the
// longest prefix over the three axes becomes a part (a prefix of one source stays original; the
// caller skips singletons) and the rest is split again. A prefix is only taken after its bound was
// counted, so every part keeps the budget. Fewer parts than halving (simulated -9% to -12% draws in
// Bleak Falls Barrow and Bthardamz).
static void splitByLights(Group group, bool exterior, const LightBudget& lights, std::vector<Group>& out, size_t& splits) {
    const auto fits = [&](const Group& g) {
        return g.sources.size() < 2 || lights.count(groupBound(g, exterior), g.origin) <= size_t(lights.budget);
    };
    if (lights.budget < 0 || fits(group)) {
        out.push_back(std::move(group));
        return;
    }
    ++splits;
    const size_t n = group.sources.size();
    std::vector<Vector3> centres;
    for (const auto& s : group.sources) centres.push_back(s.world.ApplyTransform(s.shape->GetBounds().center));
    const auto part = [&](const std::vector<size_t>& order, size_t begin, size_t end) {
        Group g;
        g.origin = group.origin; g.fade = group.fade; g.lightSplit = true;
        for (size_t k = begin; k < end; ++k) {
            const auto& s = group.sources[order[k]];
            g.vertices += s.shape->GetNumVertices();
            g.triangles += s.shape->GetNumTriangles();
            g.sources.push_back(s);
        }
        return g;
    };
    size_t bestLength = 0;
    std::vector<size_t> bestOrder;
    for (int axis = 0; axis < 3; ++axis) {
        std::vector<size_t> order(n);
        for (size_t i = 0; i < n; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return centres[a][axis] < centres[b][axis]; });
        size_t lo = 1, hi = n - 1;  // the whole group does not fit
        while (lo < hi) {
            const size_t mid = (lo + hi + 1) / 2;
            if (fits(part(order, 0, mid))) lo = mid;
            else hi = mid - 1;
        }
        if (lo > bestLength) { bestLength = lo; bestOrder = std::move(order); }
    }
    out.push_back(part(bestOrder, 0, bestLength));
    splitByLights(part(bestOrder, bestLength, n), exterior, lights, out, splits);
}

static json bake(const json& scene, const fs::path& output) {
    const bool exterior = exteriorScene(scene);
    mergeableShaderTypes = {0};
    if (scene.contains("shader_types")) {
        mergeableShaderTypes.clear();
        for (const auto& t : scene.at("shader_types")) {
            const auto type = t.get<uint32_t>();
            require(type <= 3, "Shader type not allowed to merge");
            mergeableShaderTypes.insert(type);
        }
        require(mergeableShaderTypes.count(0), "Shader types must include the plain lighting shader");
    }
    const float chunkSize = scene.at("chunk_size");
    require(std::isfinite(chunkSize) && chunkSize >= 128 && chunkSize <= 4096, "Invalid chunk size");
    std::map<std::string, std::unique_ptr<NifFile>> models;
    std::map<std::string, size_t> skipped;
    for (auto it = scene.at("assets").begin(); it != scene.at("assets").end(); ++it) {
        const fs::path file = pathOf(it.value().at("file").get<std::string>());
        require(fs::file_size(file) == it.value().at("bytes").get<uint64_t>() && hashFile(file) == it.value().at("hash").get<std::string>(), "Source asset changed during generation");
        auto nif = std::make_unique<NifFile>();
        if (nif->Load(file) || !nif->IsValid() || nif->HasUnknown() || !nif->GetHeader().GetVersion().IsSSE()) { ++skipped["unsupported_nif"]; continue; }
        auto root = nif->GetRootNode();
        if (!root || !root->transform.IsNearlyEqualTo(MatTransform{})) { ++skipped["non_identity_root"]; continue; }
        models[it.key()] = std::move(nif);
    }
    std::map<std::string, float> modelRadius;
    if (exterior) for (auto& [name, nif] : models) modelRadius[name] = modelBoundRadius(*nif);
    // One private, swapped copy of a model per distinct alternate texture list (references of different bases can
    // share a model with different swaps).
    std::map<std::string, TextureSwaps> swapped;

    std::map<std::string, std::vector<Group>> groups;
    size_t allShapes = 0;
    for (const auto& ref : scene.at("references")) {
        auto it = models.find(ref.at("model").get<std::string>());
        if (it == models.end()) continue;
        const TextureSwaps* swaps = nullptr;
        if (ref.contains("alternate_textures")) {
            const auto key = it->first + '\n' + ref.at("alternate_textures").dump();
            auto variant = swapped.find(key);
            if (variant == swapped.end()) {
                const auto& asset = scene.at("assets").at(it->first);
                const fs::path file = pathOf(asset.at("file").get<std::string>());
                require(hashFile(file) == asset.at("hash").get<std::string>(), "Source asset changed during generation");
                variant = swapped.emplace(key, applyTextureSwaps(file, ref.at("alternate_textures"))).first;
            }
            swaps = &variant->second;
        }
        // Sources of one chunk share an enable parent and its flags, so the chunk follows one enable state.
        std::string parentKey;
        if (ref.contains("enable_parent")) {
            const auto& parent = ref.at("enable_parent");
            require(parent.is_object() && parent.contains("ref") && parent.at("opposite").is_boolean()
                && (!parent.contains("pop_in") || parent.at("pop_in").is_boolean()) && parent.size() <= 3, "Invalid enable parent");
            parentKey = ":parent=" + formKey(parent.at("ref")) + (parent.at("opposite").get<bool>() ? ":opposite" : ":same")
                + (parent.value("pop_in", false) ? ":pop_in" : "");
        }
        auto& nif = swaps ? *swaps->nif : *it->second;
        for (auto* generic : nif.GetShapes()) {
            ++allShapes;
            try {
                auto* shape = dynamic_cast<BSTriShape*>(generic);
                require(shape && std::string(shape->GetBlockName()) == "BSTriShape", "unsupported_shape_type");
                require(shape->GetNumVertices() && shape->GetNumTriangles() && !shape->IsSkinned() && !shape->HasSkinInstance(), "empty_or_skinned");
                require(shape->HasNormals() && shape->HasTangents() && shape->HasUVs() && !shape->HasSecondUVs() && !shape->HasEyeData(), "unsupported_vertex_layout");
                require(shape->extraDataRefs.GetSize() == 0 && shape->collisionRef.IsEmpty() && shape->propertyRefs.GetSize() == 0, "shape_extra_data");
                int swap = -1;
                if (swaps) {
                    if (auto reason = swaps->rejected.find(shape); reason != swaps->rejected.end()) throw std::runtime_error(reason->second);
                    if (auto applied = swaps->applied.find(shape); applied != swaps->applied.end()) swap = applied->second;
                }
                const auto key = materialKey(nif, *shape);
                MatTransform toRoot = shape->transform;
                std::vector<std::string> path;
                NiAVObject* child = shape;
                std::set<NiAVObject*> visited;
                while (child) {
                    require(visited.insert(child).second, "cyclic_scenegraph");
                    require(child->controllerRef.IsEmpty() && !(child->flags & 1), "controlled_or_hidden_node");
                    require(sane(child->transform), "non_rigid_transform");
                    if (child == nif.GetRootNode()) break;
                    require(!child->name.get().empty(), "unnamed_path");
                    path.push_back(child->name.get());
                    auto* parent = nif.GetParentNode(child);
                    require(parent, "orphan_shape");
                    const std::string type = parent->GetBlockName();
                    require(type == "NiNode" || type == "BSFadeNode", "special_node");
                    require(parent->effectRefs.GetSize() == 0, "node_effect");
                    size_t names = 0;
                    for (auto& sibling : parent->childRefs) {
                        auto* node = nif.GetHeader().GetBlock(sibling);
                        if (node && node->name.get() == child->name.get()) ++names;
                    }
                    require(names == 1, "ambiguous_shape_path");
                    toRoot = parent->transform.ComposeTransforms(toRoot);
                    child = parent;
                }
                require(child == nif.GetRootNode(), "unreachable_shape");
                std::reverse(path.begin(), path.end());
                const auto placement = referenceTransform(ref);
                const auto world = placement.ComposeTransforms(toRoot);
                require(sane(world), "invalid_world_transform");
                std::array<int, 3> grid{};
                const auto center = world.ApplyTransform(shape->GetBounds().center);
                for (int axis = 0; axis < 3; ++axis) grid[axis] = static_cast<int>(std::floor(center[axis] / chunkSize));
                std::string groupKey = key + ':' + std::to_string(grid[0]) + ',' + std::to_string(grid[1]) + ',' + std::to_string(grid[2]);
                // Optional observed room groups only split the existing material/grid bins.
                // The runtime still verifies actual shared parents before activation.
                if(ref.contains("room_group")) groupKey += ":room=" + ref.at("room_group").get<std::string>();
                groupKey += parentKey;
                // Exterior chunks never mix small-object and large-object fade classes (FO4 style).
                std::string fade;
                if (exterior) {
                    fade = double(modelRadius.at(it->first)) * ref.at("scale").get<double>() < kSmallObjectRadius ? "small" : "large";
                    groupKey += ":fade=" + fade;
                }
                auto& bins = groups[groupKey];
                if (bins.empty() || bins.back().vertices + shape->GetNumVertices() > 65535 || bins.back().triangles + shape->GetNumTriangles() > 65535) {
                    Group group;
                    group.origin = {grid[0] * chunkSize, grid[1] * chunkSize, grid[2] * chunkSize};
                    group.fade = fade;
                    bins.push_back(std::move(group));
                }
                auto& group = bins.back();
                group.sources.push_back({&nif, shape, ref, path, world, placement, toRoot, swap});
                group.vertices += shape->GetNumVertices(); group.triangles += shape->GetNumTriangles();
            } catch (const std::runtime_error& e) { ++skipped[e.what()]; }
        }
    }

    LightBudget lights;
    if (scene.contains("light_budget")) {
        lights.budget = scene.at("light_budget").get<int>();
        require(lights.budget >= 0 && lights.budget <= 16, "Invalid light budget");
        for (const auto& l : scene.at("lights")) {
            require(l.is_array() && l.size() == 4, "Invalid light sphere");
            const std::array<double, 4> sphere{l[0].get<double>(), l[1].get<double>(), l[2].get<double>(), l[3].get<double>()};
            require(std::isfinite(sphere[0]) && std::isfinite(sphere[1]) && std::isfinite(sphere[2]) && std::isfinite(sphere[3])
                && sphere[3] >= 0, "Invalid light sphere");
            lights.spheres.push_back(sphere);
        }
    } else require(!scene.contains("lights"), "Lights without a light budget");
    // Without a budget the groups and their order are exactly the bins' (outputs stay byte-identical).
    std::vector<Group> finalGroups;
    size_t lightSplits = 0;
    for (auto& [key, bins] : groups)
        for (auto& group : bins) splitByLights(std::move(group), exterior, lights, finalGroups, lightSplits);

    auto manifest = scene;
    manifest.erase("references"); manifest.erase("assets"); manifest.erase("lights");
    if (lights.budget >= 0)
        manifest["light_budget"] = {{"budget", lights.budget}, {"lights", lights.spheres.size()},
            {"rule", "chunk world bound touches light sphere (radius = base + XRDS); at most budget per chunk"}};
    manifest["chunks"] = json::array();
    std::map<size_t, size_t> chunkLightHistogram;
    size_t covered = 0, totalVertices = 0, totalBytes = 0, grownBounds = 0, swappedShapes = 0, parentedChunks = 0;
    std::map<std::string, size_t> fadeChunks{{"small", 0}, {"large", 0}}, fadeShapes{{"small", 0}, {"large", 0}};
    const std::string cellHex = cellStem(scene.at("cell"));
    fs::create_directories(output / "meshes/Precombined/Skyrim");
    fs::create_directories(output / "SKSE/Plugins/Precombines");
    for (auto& group : finalGroups) {
        if (group.sources.size() < 2) { ++skipped[group.lightSplit ? "light_budget_singleton" : "singleton"]; continue; }
        auto& first = group.sources.front();
        NifFile combined; combined.Create(first.nif->GetHeader().GetVersion());
        // Exterior chunks get a BSFadeNode root so the engine fades each chunk by its own bound.
        if (exterior) combined.GetHeader().ReplaceBlock(0, std::make_unique<BSFadeNode>());
        combined.GetRootNode()->name.get() = "SkyrimPrecombines";
        auto* result = dynamic_cast<BSTriShape*>(combined.CloneShape(first.shape, "Precombined", first.nif));
        require(result, "CloneShape failed");
        combined.GetRootNode()->childRefs.Clear();
        combined.GetRootNode()->childRefs.AddBlockRef(combined.GetBlockID(result));
        combined.DeleteUnreferencedBlocks();
        result->transform.Clear(); result->SetFullPrecision(true);
        std::vector<BSVertexData> vertices;
        std::vector<Vector3> normals, tangents, bitangents;
        std::vector<Triangle> triangles;
        json sources = json::array();
        for (auto& source : group.sources) {
            const auto& input = source.shape->vertData;
            const auto ns = source.shape->UpdateRawNormals();
            const auto ts = source.shape->UpdateRawTangents();
            const auto bs = source.shape->UpdateRawBitangents();
            const size_t base = vertices.size();
            require(ns.size() == input.size() && ts.size() == input.size() && bs.size() == input.size(), "Malformed tangent basis");
            for (size_t i = 0; i < input.size(); ++i) {
                auto v = input[i];
                if (!exterior) v.vert = source.world.ApplyTransform(v.vert) - group.origin;
                // Exterior coordinates reach about 250,000 units, where a float step is 1/64. Subtract
                // the chunk origin from the reference position first, so no sum is rounded at that size.
                else v.vert = source.placement.rotation * (source.toRoot.ApplyTransform(v.vert) * source.placement.scale)
                        + (source.placement.translation - group.origin);
                require(std::isfinite(v.vert.x) && std::isfinite(v.vert.y) && std::isfinite(v.vert.z), "Nonfinite vertex");
                vertices.push_back(v);
                normals.push_back(unitVector(source.world.rotation * ns[i]));
                tangents.push_back(unitVector(source.world.rotation * ts[i]));
                bitangents.push_back(unitVector(source.world.rotation * bs[i]));
            }
            for (const auto& tri : source.shape->triangles) {
                require(tri.p1 < input.size() && tri.p2 < input.size() && tri.p3 < input.size(), "Invalid source index");
                triangles.emplace_back(static_cast<uint16_t>(base + tri.p1), static_cast<uint16_t>(base + tri.p2), static_cast<uint16_t>(base + tri.p3));
            }
            auto record = source.reference;
            record["path"] = source.path; record["world"] = transformJson(source.world);
            if (source.swap >= 0) { record["texture_swap"] = source.swap; ++swappedShapes; }
            record["vertices"] = input.size(); record["triangles"] = source.shape->triangles.size();
            const auto& asset = scene.at("assets").at(record.at("model").get<std::string>());
            record["mesh_hash"] = asset.at("hash"); record["mesh_bytes"] = asset.at("bytes");
            if (exterior) record["model_bound_radius"] = modelRadius.at(record.at("model").get<std::string>());
            sources.push_back(std::move(record));
        }
        result->SetVertexData(vertices); result->SetTriangles(triangles);
        result->SetNormals(normals); result->SetTangentData(tangents); result->SetBitangentData(bitangents);
        result->UpdateBounds();
        // nifly's single-precision Miniball can miss vertices of near-degenerate point sets (one
        // two-instance Dead Men's Respite chunk was 18.8 units short). The engine culls by this
        // sphere, so never store one that excludes geometry: grow it to the farthest vertex.
        // Misses below 0.05 units are float noise (the verifier allows 0.1); leaving those spheres
        // untouched keeps every previously verified chunk byte-identical.
        std::optional<BoundingSphere> grown;
        {
            auto bounds = result->GetBounds();
            float reach = 0.f;
            for (const auto& v : vertices) reach = std::max(reach, (v.vert - bounds.center).length());
            if (!(reach <= bounds.radius + 0.05f)) { bounds.radius = reach * 1.0001f + 0.01f; grown = bounds; ++grownBounds; }
        }
        const std::string mesh = "Precombined/Skyrim/" + cellHex + '_' + std::to_string(manifest["chunks"].size()) + ".nif";
        const auto file = output / "meshes" / mesh;
        require(combined.Save(file) == 0, "Cannot save chunk " + utf8(file));
        if (grown) {
            // NifFile::Save recomputes every bound (Optimize), so store the grown sphere with a second
            // save of the written file that neither optimizes nor reorders blocks.
            NifFile fix(file);
            const auto fixShapes = fix.GetShapes();
            require(fix.IsValid() && fixShapes.size() == 1, "Cannot reopen chunk to store its bound");
            fixShapes[0]->SetBounds(*grown);
            NifSaveOptions keep;
            keep.optimize = false;
            keep.sortBlocks = false;
            require(fix.Save(file, keep) == 0, "Cannot store the grown bound " + utf8(file));
        }
        // Every output must round-trip through a separate NIF read before it is activated.
        NifFile check(file);
        const auto shapes = check.GetShapes();
        require(check.IsValid() && shapes.size() == 1, "Output failed round-trip");
        auto* verified = dynamic_cast<BSTriShape*>(shapes[0]);
        require(verified && verified->IsFullPrecision() && verified->GetNumVertices() == vertices.size() && verified->GetNumTriangles() == triangles.size(), "Output lost geometry or precision");
        const auto stored = verified->GetBounds();
        for (size_t i = 0; i < vertices.size(); ++i) {
            require((verified->vertData[i].vert - vertices[i].vert).length() < 0.001f, "Output vertex precision mismatch");
            require((verified->vertData[i].vert - stored.center).length() <= stored.radius + 0.1f, "Stored bound excludes a vertex");
        }
        covered += group.sources.size(); totalVertices += vertices.size(); totalBytes += fs::file_size(file);
        json chunk = {{"mesh", mesh}, {"mesh_bytes", fs::file_size(file)}, {"mesh_hash", hashFile(file)},
            {"origin", vec(group.origin)}, {"vertices", vertices.size()}, {"triangles", triangles.size()}, {"sources", std::move(sources)}};
        if (lights.budget >= 0) {
            // Counted again from the bound the engine will read; the split used the same computation.
            const size_t lit = lights.count(stored, group.origin);
            require(lit <= size_t(lights.budget), "Stored chunk bound exceeds the light budget");
            chunk["lights"] = lit;
            ++chunkLightHistogram[lit];
        }
        if (exterior) {
            chunk["fade_class"] = group.fade;
            ++fadeChunks.at(group.fade); fadeShapes.at(group.fade) += group.sources.size();
        }
        parentedChunks += first.reference.contains("enable_parent");
        manifest["chunks"].push_back(std::move(chunk));
    }
    const size_t chunks = manifest["chunks"].size();
    // An exterior cell without merges is reported (and skipped by the caller), not an error.
    require(chunks || exterior, "No eligible merges in this cell");
    json report = {{"cell", scene.at("editor_id")}, {"cell_form_id", cellHex}, {"eligible_references", scene.at("references").size()},
        {"examined_shapes", allShapes}, {"covered_shapes", covered}, {"chunks", chunks}, {"draws_saved_if_all_chunks_accepted", covered - chunks},
        {"covered_shape_reduction", chunks ? double(covered) / chunks : 0.0}, {"vertices", totalVertices}, {"mesh_bytes", totalBytes},
        {"shape_skips", skipped}, {"record_skips", scene.at("record_skips")},
        {"note", "Offline counts, not measured draw calls. Runtime room, source and fade checks can reject chunks."}};
    if(scene.contains("room_grouping")) report["room_grouping"]=scene.at("room_grouping");
    if (grownBounds) report["grown_bounds"] = grownBounds;  // chunks whose nifly sphere missed vertices
    // Load-order scenes and expansions only (absent from older reports).
    if (swappedShapes) report["covered_shapes_with_texture_swaps"] = swappedShapes;
    if (parentedChunks) report["chunks_with_enable_parent"] = parentedChunks;
    for (const char* field : {"expansions", "override_audit"})
        if (scene.contains(field)) report[field] = scene.at(field);
    if (lights.budget >= 0) {
        json histogram = json::object();
        for (const auto& [lit, count] : chunkLightHistogram) histogram[std::to_string(lit)] = count;
        report["light_budget"] = {{"budget", lights.budget}, {"lights", lights.spheres.size()}, {"split_bins", lightSplits},
            {"singletons", skipped.count("light_budget_singleton") ? skipped.at("light_budget_singleton") : 0},
            {"chunk_light_histogram", histogram}};
    }
    if (exterior) {
        report["interior"] = false; report["worldspace"] = scene.at("worldspace"); report["grid"] = scene.at("grid");
        report["small_object_radius"] = kSmallObjectRadius;
        report["chunks_by_fade_class"] = fadeChunks; report["covered_shapes_by_fade_class"] = fadeShapes;
    }
    if (chunks) {
        std::ofstream manifestOut(output / "SKSE/Plugins/Precombines" / (cellHex + ".json"));
        manifestOut << manifest.dump(2); require(bool(manifestOut), "Manifest write failed");
    }
    std::ofstream reportOut(output / "generation-report.json"); reportOut << report.dump(2); require(bool(reportOut), "Report write failed");
    return report;
}

// Exterior (schema 2) scenes: BSFadeNode roots, whole-model fade classes, vertex precision far
// from the world origin, cells without merges, and malformed exterior scenes.
static void exteriorSelfTest(const fs::path& dir) {
    const std::vector<Vector3> vertices{{0,0,0},{2,0,0},{0,2,0}}, normals(3,Vector3{0,0,1});
    const std::vector<Vector2> uv{{0,0},{1,0},{0,1}};
    const std::vector<Triangle> triangles{{0,1,2}};
    const auto model=[&](const fs::path& file,const std::vector<std::pair<std::string,Vector3>>& shapes){
        NifFile nif; nif.Create(NiVersion::getSSE());
        for(const auto& [name,offset]:shapes){
            auto* shape=dynamic_cast<BSTriShape*>(nif.CreateShapeFromData(name,&vertices,&triangles,&uv,&normals));
            require(shape,"Exterior fixture shape missing");
            shape->SetTangents(true); shape->CalcTangentSpace(); shape->UpdateBounds();
            shape->transform.translation=offset;
        }
        require(nif.Save(file)==0,"Exterior fixture save failed");
        return json{{"file",utf8(file)},{"bytes",fs::file_size(file)},{"hash",hashFile(file)}};
    };
    // Each triangle's own bound radius is ~1.41; the whole two-shape model's is 150+sqrt(2).
    json scene={{"schema",2},{"plugin","Skyrim.esm"},{"interior",false},{"cell",0x42246u},{"editor_id","FixtureExterior"},
        {"worldspace",{{"form_id",0x16bb4u},{"editor_id","FixtureWorld"}}},{"grid",{41,-25}},{"chunk_size",2048},
        {"record_skips",json::object()},{"references",json::array()},
        {"assets",{{"pair.nif",model(dir/"pair.nif",{{"A",{0,0,0}},{"B",{300,0,0}}})},{"single.nif",model(dir/"single.nif",{{"A",{0,0,0}}})}}}};
    // Near Riften's real coordinates, where one float step is 1/64 unit. Scale 0.5 makes the same model small.
    const float scales[]{1.f,1.f,0.5f,0.5f};
    for(uint32_t i=0;i<4;++i) scene["references"].push_back({{"ref",0x300u+i},{"base",50u},{"model","pair.nif"},
        {"position",{169000.5f+20*i,-101000.25f,100.f}},{"rotation",{0.f,0.f,0.3f*i}},{"scale",scales[i]}});
    const auto report=bake(scene,dir/"exterior");
    require(report.at("chunks")==2 && report.at("covered_shapes")==8 && report.at("chunks_by_fade_class")==json{{"large",1},{"small",1}}
        && report.at("covered_shapes_by_fade_class")==json{{"large",4},{"small",4}} && report.at("interior")==false
        && report.at("grid")==scene.at("grid") && report.at("worldspace")==scene.at("worldspace"),"Exterior fixture did not split by whole-model fade class");
    std::ifstream in(dir/"exterior/SKSE/Plugins/Precombines/00042246.json");
    const auto manifest=json::parse(in);
    require(manifest.at("schema")==2 && manifest.at("interior")==false && manifest.at("worldspace")==scene.at("worldspace")
        && manifest.at("grid")==scene.at("grid") && manifest.at("chunks").size()==2,"Exterior manifest header wrong");
    for(const auto& chunk:manifest.at("chunks")){
        const bool small=chunk.at("fade_class")=="small";
        require(small || chunk.at("fade_class")=="large","Exterior chunk without fade class");
        NifFile nif(dir/"exterior/meshes"/chunk.at("mesh").get<std::string>());
        require(nif.IsValid() && std::string(nif.GetRootNode()->GetBlockName())=="BSFadeNode" && nif.GetRootNode()->name.get()=="SkyrimPrecombines"
            && nif.GetHeader().GetNumBlocks()==4 && nif.GetRootNode()->childRefs.GetSize()==1,"Exterior chunk root is not a BSFadeNode");
        auto* merged=dynamic_cast<BSTriShape*>(nif.GetShapes().at(0));
        require(merged && merged->vertData.size()==3*chunk.at("sources").size(),"Exterior chunk layout changed");
        const auto origin=vec(chunk.at("origin"));
        size_t base=0;
        for(const auto& source:chunk.at("sources")){
            require(std::abs(source.at("model_bound_radius").get<double>()-151.41421)<0.01,"Fade class did not use the whole model bound");
            require((source.at("scale").get<double>()<1)==small,"Fade class does not follow bound radius x reference scale");
            // Expected positions in double precision, directly from the reference placement.
            const auto placement=referenceTransform(source);
            const double offset=source.at("path").at(0)=="B" ? 300 : 0;
            for(size_t k=0;k<3;++k) for(int axis=0;axis<3;++axis){
                const double local[3]{vertices[k].x+offset,vertices[k].y,vertices[k].z};
                double expected=double(placement.translation[axis])-origin[axis];
                for(int j=0;j<3;++j) expected+=double(placement.rotation[axis][j])*local[j]*double(placement.scale);
                require(std::abs(merged->vertData[base+k].vert[axis]-expected)<0.001,"Exterior vertex lost precision far from the origin");
            }
            base+=3;
        }
    }
    // A cell whose only shape is a singleton is reported with zero chunks and no manifest.
    auto lone=scene; lone["references"]=json::array();
    lone["references"].push_back({{"ref",0x310u},{"base",50u},{"model","single.nif"},{"position",{1000.f,1000.f,0.f}},{"rotation",{0.f,0.f,0.f}},{"scale",1.f}});
    const auto empty=bake(lone,dir/"exterior-empty");
    require(empty.at("chunks")==0 && empty.at("covered_shape_reduction")==0.0 && empty.at("shape_skips").at("singleton")==1
        && !fs::exists(dir/"exterior-empty/SKSE/Plugins/Precombines/00042246.json") && fs::is_empty(dir/"exterior-empty/meshes/Precombined/Skyrim"),
        "Exterior cell without merges was not reported cleanly");
    const std::vector<std::pair<std::string,json>> malformed{{"schema",1},{"interior",true},{"grid",json::array({41})},{"worldspace",json::object()}};
    for(const auto& [field,value]:malformed){
        auto bad=scene; bad[field]=value;
        std::string why; try{bake(bad,dir/("exterior-bad-"+field));}catch(const std::runtime_error& e){why=e.what();}
        require(why=="Unsupported scene schema or source" || why=="Invalid exterior worldspace or grid","Malformed exterior scene accepted: "+field);
    }
    auto noGrid=scene; noGrid.erase("grid");
    std::string why; try{bake(noGrid,dir/"exterior-no-grid");}catch(const std::runtime_error& e){why=e.what();}
    require(why=="Unsupported scene schema or source","Exterior scene without grid accepted");
}

// Load-order scenes: plugin-relative cells and file stems, enable-parent grouping and base alternate textures.
static void loadOrderSelfTest(const fs::path& dir) {
    const std::vector<Vector3> vertices{{0,0,0},{2,0,0},{0,2,0}}, normals(3,Vector3{0,0,1});
    const std::vector<Vector2> uv{{0,0},{1,0},{0,1}};
    const std::vector<Triangle> triangles{{0,1,2}};
    using Shapes=std::vector<std::pair<std::string,std::vector<std::string>>>;  // name, texture slots
    const auto model=[&](const fs::path& file,const Shapes& shapes,bool shareFirstMaterial=false){
        NifFile nif; nif.Create(NiVersion::getSSE());
        BSLightingShaderProperty* firstShader=nullptr;
        for(size_t i=0;i<shapes.size();++i){
            auto* shape=dynamic_cast<BSTriShape*>(nif.CreateShapeFromData(shapes[i].first,&vertices,&triangles,&uv,&normals));
            require(shape,"Load-order fixture shape missing");
            shape->SetTangents(true); shape->CalcTangentSpace(); shape->UpdateBounds();
            shape->transform.translation={float(10*i),0,0};
            auto* shader=dynamic_cast<BSLightingShaderProperty*>(nif.GetShader(shape));
            auto* set=nif.GetHeader().GetBlock(shader->textureSetRef);
            require(set && set->textures.size()==9,"Load-order fixture texture set");
            for(size_t t=0;t<shapes[i].second.size();++t) set->textures[uint32_t(t)].get()=shapes[i].second[t];
            if(i==0) firstShader=shader;
            else if(shareFirstMaterial) shape->ShaderPropertyRef()->index=nif.GetBlockID(firstShader);
        }
        require(nif.Save(file)==0,"Load-order fixture save failed");
        return json{{"file",utf8(file)},{"bytes",fs::file_size(file)},{"hash",hashFile(file)}};
    };
    const auto form=[](const char* plugin,uint32_t id){ return json{{"plugin",plugin},{"id",id}}; };
    const auto reference=[&](uint32_t id,const char* mesh,float x){
        return json{{"ref",form("Dragonborn.esm",id)},{"base",form("Dragonborn.esm",0x900)},{"model",mesh},
            {"position",{x,0.f,0.f}},{"rotation",{0.f,0.f,0.f}},{"scale",1.f}};
    };
    const Shapes rock{{"Rock:0",{"textures\\rock.dds","textures\\rock_n.dds"}},{"Rock:1",{"textures\\rock.dds","textures\\rock_n.dds"}}};
    json scene={{"schema",1},{"plugin","Dragonborn.esm"},{"interior",true},{"cell",form("Dragonborn.esm",0x17c1a)},
        {"editor_id","FixtureDragonborn"},{"chunk_size",2048},{"record_skips",json::object()},
        {"assets",{{"rock.nif",model(dir/"rock.nif",rock)}}},{"references",json::array()}};

    // Enable parents: two sources follow the parent, two the opposite state, two have none; each pair is one chunk.
    for(uint32_t i=0;i<6;++i){
        auto ref=reference(0x800+i,"rock.nif",100.f+30*i);
        if(i<4) ref["enable_parent"]={{"ref",i<2 ? json(0x1234) : form("Dawnguard.esm",0x2345)},{"opposite",i>=2},{"pop_in",false}};
        scene["references"].push_back(ref);
    }
    // One shape per reference, so every chunk is one pair of references.
    scene["assets"]["rock.nif"]=model(dir/"rock.nif",{rock[0]});
    const auto parents=bake(scene,dir/"parents");
    require(parents.at("chunks")==3 && parents.at("covered_shapes")==6 && parents.at("chunks_with_enable_parent")==2
        && parents.at("cell_form_id")=="dragonborn.esm_017c1a","Enable-parent fixture did not group by parent");
    require(fs::exists(dir/"parents/meshes/Precombined/Skyrim/dragonborn.esm_017c1a_2.nif"),"Plugin-relative chunk name");
    std::ifstream parentInput(dir/"parents/SKSE/Plugins/Precombines/dragonborn.esm_017c1a.json");
    const auto parentManifest=json::parse(parentInput);
    require(parentManifest.at("cell")==scene.at("cell") && parentManifest.at("plugin")=="Dragonborn.esm","Plugin-relative manifest header");
    for(const auto& chunk:parentManifest.at("chunks")){
        const auto& sources=chunk.at("sources");
        require(sources.size()==2 && sources[0].value("enable_parent",json())==sources[1].value("enable_parent",json())
            && sources[0].at("ref").is_object(),"A chunk mixes enable parents");
    }
    int malformed=0;
    for(const auto& [field,value,why]:std::vector<std::tuple<std::string,json,std::string>>{
            {"plugin",json("Skyrim.esm"),"Unsupported scene schema or source"},
            {"cell",form("../Dragonborn.esm",0x17c1a),"Invalid form reference"},
            {"cell",form("Dragonborn.esm",0x1000000),"Invalid form reference"}}){
        auto bad=scene; bad[field]=value;
        std::string error; try{bake(bad,dir/("relative-bad-"+std::to_string(malformed++)));}catch(const std::runtime_error& e){error=e.what();}
        require(error==why,"Malformed plugin-relative scene accepted: "+field+" ("+error+")");
    }

    // Alternate textures: Rock:0 of reference A is swapped to the moss texture set and merges with a native moss
    // shape; Rock:1 of A and both shapes of B (no swaps) merge as rock.
    const auto txst=[&](std::vector<std::string> textures,const char* name,int index){
        textures.resize(8);
        return json{{"name3d",name},{"index3d",index},{"texture_set",form("Skyrim.esm",0x10e9e4)},{"textures",textures}};
    };
    auto swaps=scene; swaps["references"]=json::array();
    swaps["assets"]={{"rock.nif",model(dir/"rock.nif",rock)},{"moss.nif",model(dir/"moss.nif",{{"Moss:0",{"textures\\moss.dds","textures\\moss_n.dds"}}})}};
    auto a=reference(0x810,"rock.nif",100); a["alternate_textures"]=json::array({txst({"moss.dds","moss_n.dds"},"Rock:0",0)});
    swaps["references"].push_back(a);
    swaps["references"].push_back(reference(0x811,"moss.nif",150));
    swaps["references"].push_back(reference(0x812,"rock.nif",200));
    const auto swapped=bake(swaps,dir/"swaps");
    require(swapped.at("chunks")==2 && swapped.at("covered_shapes")==5 && swapped.at("covered_shapes_with_texture_swaps")==1,
        "Alternate texture fixture did not merge by swapped material");
    std::ifstream swapInput(dir/"swaps/SKSE/Plugins/Precombines/dragonborn.esm_017c1a.json");
    const auto swapManifest=json::parse(swapInput);
    size_t checkedMoss=0;
    for(const auto& chunk:swapManifest.at("chunks")){
        NifFile nif(dir/"swaps/meshes"/chunk.at("mesh").get<std::string>());
        auto* shader=dynamic_cast<BSLightingShaderProperty*>(nif.GetShader(nif.GetShapes().at(0)));
        const auto diffuse=nif.GetHeader().GetBlock(shader->textureSetRef)->textures[0].get();
        for(const auto& source:chunk.at("sources")){
            const bool swappedSource=source.at("ref")==a.at("ref") && source.at("path")==json::array({"Rock:0"});
            require(swappedSource==source.contains("texture_swap") && (!swappedSource || source.at("texture_swap")==0),"Texture swap not recorded");
            require(diffuse==(swappedSource || source.at("model")=="moss.nif" ? "textures\\moss.dds" : "textures\\rock.dds"),"Swapped textures not merged");
            checkedMoss+=swappedSource;
        }
    }
    require(checkedMoss==1,"Swapped source missing");
    // Name and index disagree (both shapes stay), an unknown name and index swap nothing, a partial set and a shared
    // material are not swapped, and a swap without a 3D index is never applied.
    auto unsure=swaps; unsure["references"]=json::array();
    unsure["assets"]["back.nif"]=model(dir/"back.nif",{{"Back:0",{"textures\\rock.dds","textures\\rock_n.dds","","","","","","textures\\back.dds"}}});
    unsure["assets"]["shared.nif"]=model(dir/"shared.nif",rock,true);
    auto ambiguous=reference(0x820,"rock.nif",100); ambiguous["alternate_textures"]=json::array({txst({"moss.dds","moss_n.dds"},"Rock:0",1)});
    auto nothing=reference(0x821,"rock.nif",150); nothing["alternate_textures"]=json::array({txst({"moss.dds","moss_n.dds"},"Nope",9)});
    auto partial=reference(0x822,"back.nif",200); partial["alternate_textures"]=json::array({txst({"moss.dds","moss_n.dds"},"Back:0",0)});
    auto shared=reference(0x823,"shared.nif",250); shared["alternate_textures"]=json::array({txst({"moss.dds","moss_n.dds"},"Rock:1",1)});
    // Without a 3D index the game never applies a swap: both shapes keep the rock material.
    auto unindexed=reference(0x826,"rock.nif",400); unindexed["alternate_textures"]=json::array({txst({"moss.dds","moss_n.dds"},"Rock:0",-1)});
    for(const auto& ref:{ambiguous,nothing,partial,shared,unindexed,reference(0x824,"back.nif",300),reference(0x825,"moss.nif",350)})
        unsure["references"].push_back(ref);
    const auto unsureReport=bake(unsure,dir/"swaps-unsure");
    const auto& skips=unsureReport.at("shape_skips");
    require(skips.at("alternate_texture_ambiguous")==2 && skips.at("alternate_texture_partial")==1
        && skips.at("alternate_texture_shared_material")==1 && !unsureReport.contains("covered_shapes_with_texture_swaps")
        && unsureReport.at("covered_shapes")==5,"Unsafe texture swaps were not kept original");
}

static void selfTest(const fs::path& dir) {
    require(!fs::exists(dir), "Self-test needs a fresh directory");
    fs::create_directories(dir);
    NifFile fixture; fixture.Create(NiVersion::getSSE());
    std::vector<Vector3> vertices{{0,0,0},{2,0,0},{0,2,0}}, normals(3,Vector3{0,0,1});
    std::vector<Vector2> uv{{0,0},{1,0},{0,1}};
    std::vector<Triangle> triangles{{0,1,2}};
    auto* shape=dynamic_cast<BSTriShape*>(fixture.CreateShapeFromData("Triangle",&vertices,&triangles,&uv,&normals));
    require(shape, "Fixture shape missing");
    shape->SetTangents(true); shape->CalcTangentSpace();
    shape->SetVertexColors(true);
    shape->vertData[0].colorData[0]=42;
    auto* shader=dynamic_cast<BSLightingShaderProperty*>(fixture.GetShader(shape));
    require(shader, "Fixture shader missing");
    const auto originalKey=materialKey(fixture,*shape);
    shader->glossiness+=1;
    require(materialKey(fixture,*shape)!=originalKey,"Different glossiness merged");
    shader->glossiness-=1;
    auto alpha=std::make_unique<NiAlphaProperty>(); alpha->flags=4844; alpha->threshold=128;
    fixture.AssignAlphaProperty(shape,std::move(alpha));
    const auto alphaKey=materialKey(fixture,*shape);
    auto* alphaPtr=fixture.GetAlphaProperty(shape); alphaPtr->threshold=127;
    require(materialKey(fixture,*shape)!=alphaKey,"Different alpha threshold merged");
    alphaPtr->flags|=1;
    bool rejected=false; try{materialKey(fixture,*shape);}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"Alpha blend accepted"); alphaPtr->flags&=~1;
    // Environment-mapped (type 1) materials merge only when the scene allows them, and the type is
    // part of the material key.
    shader->SetShaderType(1);
    rejected=false; try{materialKey(fixture,*shape);}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"Environment-mapped material merged without shader_types");
    mergeableShaderTypes={0,1};
    const auto envKey=materialKey(fixture,*shape);
    shader->SetShaderType(0);
    require(materialKey(fixture,*shape)!=envKey,"Shader type is not part of the material key");
    mergeableShaderTypes={0};
    const fs::path mesh=dir/"source.nif";
    require(fixture.Save(mesh)==0,"Fixture save failed");
    const json asset={{"file",utf8(mesh)},{"bytes",fs::file_size(mesh)},{"hash",hashFile(mesh)}};
    json scene={{"schema",1},{"plugin","Skyrim.esm"},{"interior",true},{"cell",42},{"editor_id","Fixture"},
        {"chunk_size",2048},{"record_skips",json::object()},{"assets",{{"source.nif",asset}}},{"references",json::array()}};
    for(uint32_t i=0;i<3;++i) scene["references"].push_back({{"ref",100+i},{"base",50},{"model","source.nif"},
        {"position",{100.f+10*i,100.f,100.f}},{"rotation",{0.f,0.f,1.57079632679f}},{"scale",2.f}});
    const auto report=bake(scene,dir/"rotated");
    require(report.at("chunks")==1 && report.at("covered_shapes")==3,"Fixture did not merge");
    NifFile output(dir/"rotated/meshes/Precombined/Skyrim/0000002a_0.nif");
    require(std::string(output.GetRootNode()->GetBlockName())=="NiNode" && output.GetRootNode()->name.get()=="SkyrimPrecombines","Interior root changed");
    auto* actual=dynamic_cast<BSTriShape*>(output.GetShapes().at(0));
    require(actual && actual->IsFullPrecision() && actual->vertData.size()==9,"Fixture round-trip changed layout");
    // Positive Skyrim Z rotation is clockwise, independently checked here.
    require((actual->vertData[1].vert-Vector3{100,96,100}).length()<0.001f,"Incorrect reference rotation/scale");
    require(actual->vertData[0].colorData[0]==42 && actual->vertData[1].uv==Vector2{1,0},"UV or color lost");
    require(actual->triangles[1].p1==3 && actual->triangles[2].p3==8,"Triangle offsets corrupted");
    // Non-ASCII folders (an accented Latin letter and Cyrillic): the scene names its assets in UTF-8, survives
    // JSON's strict dump and parse, and bakes into an output folder of the same name on any ANSI code page.
    const fs::path unicodeDir=dir/fs::path(u8"Cache \u00e9 \u0418\u0433\u0440\u044b");
    fs::create_directories(unicodeDir);
    fs::copy_file(mesh,unicodeDir/"source.nif");
    auto unicode=scene;
    unicode["assets"]["source.nif"]["file"]=utf8(unicodeDir/"source.nif");
    const auto unicodeReport=bake(json::parse(unicode.dump()),unicodeDir/"out");
    require(unicodeReport.at("chunks")==1 && fs::exists(unicodeDir/"out/meshes/Precombined/Skyrim/0000002a_0.nif")
        && fs::exists(unicodeDir/"out/SKSE/Plugins/Precombines/0000002a.json"),"Non-ASCII asset or output folder failed");
    auto rooms=scene;
    rooms["references"][0]["room_group"]="observed:a";
    rooms["references"][1]["room_group"]="observed:a";
    rooms["references"][2]["room_group"]="observed:b";
    const auto roomSplit=bake(rooms,dir/"rooms");
    require(roomSplit.at("chunks")==1 && roomSplit.at("covered_shapes")==2,"Cross-room sources merged");
    std::ifstream roomInput(dir/"rooms/SKSE/Plugins/Precombines/0000002a.json");
    const auto roomManifest=json::parse(roomInput);
    for(const auto& source:roomManifest.at("chunks").at(0).at("sources"))
        require(source.at("room_group")=="observed:a","Wrong room source in output");
    // Light budget: three sources 600 units apart in one bin, each touching only its own light.
    // Merged they touch all three; a budget of two merges the first two (the longest prefix that
    // fits) and leaves the last original.
    auto lit=scene;
    lit["references"]=json::array();
    lit["lights"]=json::array();
    for(uint32_t i=0;i<3;++i) {
        lit["references"].push_back({{"ref",300+i},{"base",50},{"model","source.nif"},
            {"position",{100.f+600*i,100.f,100.f}},{"rotation",{0.f,0.f,0.f}},{"scale",1.f}});
        lit["lights"].push_back({101.0+600*i,101.0,100.0,50.0});
    }
    lit["light_budget"]=3;
    const auto litAll=bake(lit,dir/"lights3");
    require(litAll.at("chunks")==1 && litAll.at("covered_shapes")==3 && litAll.at("light_budget").at("split_bins")==0
        && litAll.at("light_budget").at("chunk_light_histogram")==json{{"3",1}},"Light budget split a fitting chunk");
    lit["light_budget"]=2;
    const auto litTwo=bake(lit,dir/"lights2");
    require(litTwo.at("chunks")==1 && litTwo.at("covered_shapes")==2 && litTwo.at("shape_skips").at("light_budget_singleton")==1
        && litTwo.at("light_budget").at("chunk_light_histogram")==json{{"2",1}},"Light budget did not split");
    std::ifstream litInput(dir/"lights2/SKSE/Plugins/Precombines/0000002a.json");
    const auto litManifest=json::parse(litInput);
    require(litManifest.at("chunks").at(0).at("lights")==2 && !litManifest.contains("lights")
        && litManifest.at("light_budget").at("budget")==2 && litManifest.at("chunks").at(0).at("sources").at(0).at("ref")==300
        && litManifest.at("chunks").at(0).at("sources").at(1).at("ref")==301,
        "Light budget manifest wrong");
    auto unlit=lit; unlit.erase("light_budget");
    std::string litWhy; try{bake(unlit,dir/"lights-without-budget");}catch(const std::runtime_error& e){litWhy=e.what();}
    require(litWhy=="Lights without a light budget","Lights accepted without a budget");
    // Vanilla tangent-basis bytes decode up to ~1% long. Rotations that align such
    // a vector with an axis used to round to 256 and wrap to 0, flipping it.
    const auto plainVertices=shape->vertData;
    for(auto& v:shape->vertData){
        v.normal[0]=81;v.normal[1]=8;v.normal[2]=138;          // |n| = 1.009 (Ustengrav pillar)
        v.tangent[0]=247;v.tangent[1]=81;v.tangent[2]=128;     // |t| = 1.006
        v.bitangentX=0.08f;v.bitangentY=133;v.bitangentZ=255;  // |b| = 1.004
    }
    require(fixture.Save(mesh)==0,"Tangent-basis fixture save failed");
    auto basis=scene;
    basis["assets"]["source.nif"]={{"file",utf8(mesh)},{"bytes",fs::file_size(mesh)},{"hash",hashFile(mesh)}};
    basis["references"]=json::array();
    std::vector<Vector3> angles;
    for(int i=0;i<72;++i) angles.push_back({0.f,0.f,i*0.0872664626f});
    for(const Vector3& a:{Vector3{0.5f,0.f,0.3f},Vector3{0.f,1.2f,2.f},Vector3{3.14159265f,0.f,0.f},
            Vector3{1.57079633f,0.f,0.f},Vector3{0.f,1.57079633f,0.f},Vector3{-0.7f,0.4f,4.38739347f}}) angles.push_back(a);
    for(uint32_t i=0;i<angles.size();++i) basis["references"].push_back({{"ref",200+i},{"base",50},{"model","source.nif"},
        {"position",{100.f,100.f,100.f}},{"rotation",{angles[i].x,angles[i].y,angles[i].z}},{"scale",1.f}});
    const auto basisReport=bake(basis,dir/"basis");
    require(basisReport.at("chunks")==1 && basisReport.at("covered_shapes")==angles.size(),"Tangent-basis fixture did not merge");
    NifFile basisOut(dir/"basis/meshes/Precombined/Skyrim/0000002a_0.nif");
    auto* merged=dynamic_cast<BSTriShape*>(basisOut.GetShapes().at(0));
    require(merged && merged->vertData.size()==3*angles.size(),"Tangent-basis fixture layout changed");
    const auto unpack=[](const uint8_t* b){return Vector3(b[0]/255.f*2-1,b[1]/255.f*2-1,b[2]/255.f*2-1);};
    for(uint32_t i=0;i<angles.size();++i) {
        const auto rotation=referenceTransform(basis["references"][i]).rotation;
        for(uint32_t k=0;k<3;++k) {
            const auto& in=shape->vertData[k];
            const auto& out=merged->vertData[3*i+k];
            const Vector3 bitIn(in.bitangentX,in.bitangentY/255.f*2-1,in.bitangentZ/255.f*2-1);
            const Vector3 bitOut(out.bitangentX,out.bitangentY/255.f*2-1,out.bitangentZ/255.f*2-1);
            for(const auto& [expected,actual]:{std::pair{rotation*unpack(in.normal),unpack(out.normal)},
                    std::pair{rotation*unpack(in.tangent),unpack(out.tangent)},std::pair{rotation*bitIn,bitOut}})
                require(expected.dot(actual)>0.999f*expected.length()*actual.length(),"Rotated tangent basis flipped or bent");
        }
    }
    shape->vertData=plainVertices;
    // A single source with many triangles tests the independent 16-bit triangle
    // limit: three instances must split into a two-instance chunk plus singleton.
    shape->SetTriangles(std::vector<Triangle>(30000,Triangle{0,1,2}));
    require(fixture.Save(mesh)==0,"Large fixture save failed");
    scene["assets"]["source.nif"]={{"file",utf8(mesh)},{"bytes",fs::file_size(mesh)},{"hash",hashFile(mesh)}};
    const auto split=bake(scene,dir/"triangles");
    require(split.at("covered_shapes")==2 && split.at("chunks")==1,"Triangle limit failed to split");
    NifFile splitNif(dir/"triangles/meshes/Precombined/Skyrim/0000002a_0.nif");
    require(splitNif.GetShapes().at(0)->GetNumTriangles()==60000,"Triangle count overflow");
    const auto originalVertices=shape->vertData;
    std::vector<BSVertexData> manyVertices(22000,originalVertices[0]);
    manyVertices[1]=originalVertices[1]; manyVertices[2]=originalVertices[2];
    shape->SetVertexData(manyVertices); shape->SetTriangles(triangles);
    scene["references"].push_back({{"ref",103},{"base",50},{"model","source.nif"},
        {"position",{130.f,100.f,100.f}},{"rotation",{0.f,0.f,1.57079632679f}},{"scale",2.f}});
    require(fixture.Save(mesh)==0,"Vertex fixture save failed");
    scene["assets"]["source.nif"]={{"file",utf8(mesh)},{"bytes",fs::file_size(mesh)},{"hash",hashFile(mesh)}};
    const auto vertexSplit=bake(scene,dir/"vertices");
    require(vertexSplit.at("chunks")==2 && vertexSplit.at("covered_shapes")==4,"Vertex limit failed to split");
    shape->SetVertexData(originalVertices);
    MatTransform branchTransform; branchTransform.translation={3,4,5};
    branchTransform.rotation=Matrix3(0,-1,0,1,0,0,0,0,1);
    auto* branch=fixture.AddNode("Branch",branchTransform);
    fixture.SetParentNode(shape,branch); shape->transform.translation={7,8,9};
    require(fixture.Save(mesh)==0,"Nested fixture save failed");
    scene["assets"]["source.nif"]={{"file",utf8(mesh)},{"bytes",fs::file_size(mesh)},{"hash",hashFile(mesh)}};
    bake(scene,dir/"nested");
    NifFile nested(dir/"nested/meshes/Precombined/Skyrim/0000002a_0.nif");
    auto* nestedShape=dynamic_cast<BSTriShape*>(nested.GetShapes().at(0));
    require((nestedShape->vertData[1].vert-Vector3{126,110,128}).length()<0.001f,"Nested transform order incorrect");
    const auto nestedNormal=nestedShape->UpdateRawNormals().at(0);
    require(std::abs(nestedNormal.z-1.f)<0.01f,"Normal rotation corrupted");
    auto withGrid=scene; withGrid["grid"]={0,0};
    std::string why; try{bake(withGrid,dir/"interior-grid");}catch(const std::runtime_error& e){why=e.what();}
    require(why=="Unsupported scene schema or source","Interior scene with exterior fields accepted");
    scene["assets"]["source.nif"]["hash"]="0000000000000000";
    rejected=false; try{bake(scene,dir/"stale");}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"Stale source fingerprint accepted");
    exteriorSelfTest(dir);
    fs::create_directories(dir/"load-order");
    loadOrderSelfTest(dir/"load-order");
    std::cout<<"PASS: material properties, alpha threshold/blend, rotated/scaled positions, nested transforms, UV/color preservation, index offsets, non-ASCII paths, room grouping, light budget, over-length rotated tangent bases, vertex/triangle limits, full precision, round-trip, stale source rejection, "
               "exterior schema 2 (BSFadeNode root, whole-model fade classes, far-from-origin precision, cells without merges, malformed scenes), "
               "load-order scenes (plugin-relative cells and file names, enable-parent grouping, alternate textures: applied, ambiguous, partial, shared, no-op)\n";
}

namespace precombine {
    json Bake(const json& scene, const fs::path& output) { return bake(scene, output); }
    void SelfTest(const fs::path& directory) { selfTest(directory); }
}
