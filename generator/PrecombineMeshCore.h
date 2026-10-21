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

#pragma once
// In-process entry points of the precombine baker (generator/PrecombineMeshCore.cpp), linked into the SKSE plugin (its
// runtime bake), the PrecombineMesh executable and the offline tools. nifly is GPL-3.0, as this mod is.
#include <filesystem>
#include <nlohmann/json.hpp>

namespace precombine {
    // The baker's output version. Bump it whenever Bake's output changes (bytes, bounds, encodings, merge or eligibility
    // rules): the runtime bake puts it in every cell's cache key, index entry and part, so each cached cell is baked again
    // on its next visit instead of keeping chunks from the older baker.
    inline constexpr int kOutputVersion = 1;
    // Bakes one scene (the JSON scene: references, assets, chunk size, optional room groups and
    // light budget) into `output` (meshes/Precombined/Skyrim, SKSE/Plugins/Precombines/<cell>.json) and returns the
    // generation report. Throws std::runtime_error on invalid input. Safe to call from several threads at once. An
    // asset's "file" is UTF-8, as all JSON text is, whatever the system's ANSI code page.
    // Optional, for load-order scenes: "cell" (and "worldspace.form_id", source "ref"/"base") may be plugin-relative
    // {"plugin": file, "id": local form ID}; a cell of another plugin than Skyrim.esm is written as
    // "<plugin lower case>_<6-digit local ID>". A reference may carry "enable_parent" {"ref", "opposite", "pop_in"}
    // (chunks never mix parents or flags) and "alternate_textures" (base MODS swaps the baker applies; manifest
    // sources get "texture_swap"). Scenes without these fields bake exactly as before.
    nlohmann::json Bake(const nlohmann::json& scene, const std::filesystem::path& output);
    // The baker's own fixtures (merging rules, encodings, exterior schema); throws on the first failure.
    void SelfTest(const std::filesystem::path& directory);
}
