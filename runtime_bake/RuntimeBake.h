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

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// Runtime bake. The first time a cell is attached, its eligible static references are collected from the loaded game
// (the load order's winning records, any plugin), their meshes are read through the resource system (so mesh replacers
// are what gets merged), and the baker (precombine::Bake, linked in) bakes the chunks on a below-normal-priority worker
// thread; "runtime_bake_tool" names an external PrecombineMesh.exe instead (development). Interiors are
// baked nearest-first: room groups ordered by distance from the player are packed into parts, and each part is
// published on its own (Data/SKSE/Plugins/Precombines/RuntimeBake/parts), naming forms by plugin and local ID. A
// SkyrimPrecombines.dll with the hand-off export (SkyrimPrecombines_RuntimeBakePart) activates each part during play;
// without it, only Skyrim.esm records are baked and the parts are merged into Data/SKSE/Plugins/Precombines/<cell>.json
// for the next data load. On unless SkyrimPrecombines.ini sets bRuntimeBake=0, or bEnabled=0 outside developer mode (whose
// key can switch precombines on in play).
namespace RuntimeBake {
    struct Settings {
        bool enabled=false;                 // "runtime_bake", default true in Load (which also needs "enabled" or "developer_mode")
        std::filesystem::path tool;         // "runtime_bake_tool": an external baker; empty (default): in-process
        float chunkSize=4096.f;             // "runtime_bake_chunk_size"
        int lightBudget=6;                  // "runtime_bake_light_budget"; -1 = none
        uint32_t timeoutSeconds=600;        // "runtime_bake_timeout"
        uint32_t firstPartReferences=150;   // "runtime_bake_first_part": references in the first (nearest) part
        uint32_t partReferences=400;        // "runtime_bake_part_size": references in each later part; 0 = one part
        bool prebakeDoors=false;            // "runtime_prebake_doors"
        float doorDistance=768.f;           // "runtime_prebake_door_distance": doors this close to the player count
        // "runtime_bake_detect_llf" (default true): with Community Shaders' Light Limit Fix active, the 7-light limit
        // the budget works around does not apply, so cells are baked without a light budget (fewer chunks).
        bool detectLightLimitFix=true;
        // Previs for interiors, rendered after a cell's chunks (generator/VisibilityEstimate.cpp on the GPU).
        bool previs=false;                  // "runtime_previs"
        std::filesystem::path previsTool;   // "runtime_previs_tool"; default Data/SKSE/Plugins/Precombines/VisibilityEstimate.exe
        std::string previsGpu="hardware";   // "runtime_previs_gpu": "hardware", "warp", or "" for the CPU rasterizer
        int previsFaceSize=128;             // "runtime_previs_face_size": cube-map face resolution
        double previsGpuDuty=0.5;           // "runtime_previs_gpu_duty": share of time the bake keeps the GPU busy
        uint32_t previsTimeoutSeconds=14400; // "runtime_previs_timeout": a background job on a busy GPU can take an hour
        // "runtime_previs_method": "render" (cube maps at every sample point) or "portals" (a voxel cell-and-portal graph
        // answers each point; points near or behind single-sided surfaces are still rendered, on runtime_previs_gpu).
        // Ustengrav02: 240 s against 1,204 s for the paced GPU render.
        std::string previsMethod="render";
        int previsThreads=4;                // "runtime_previs_threads": portal-method threads (idle priority)
        // "runtime_previs_exterior": exteriors too (with "runtime_previs"): each camera cell's view cells against the
        // chunks within two cells, with their references and terrain as occluders (see ExteriorPrevisWorker).
        bool previsExterior=false;
    };

    // A form as plugin file plus local form ID: stable across load orders, resolved by the precombine plugin.
    struct FormKey {
        std::string plugin;
        uint32_t local=0;
    };

    // One eligible reference, copied out of the game on a task thread; the worker never touches game objects.
    struct Reference {
        uint32_t ref=0, base=0;             // runtime form IDs (this load order)
        FormKey refKey, baseKey;
        std::string model;                  // lowercase, '/' separators, relative to meshes/
        std::array<float,3> position{}, rotation{};
        float scale=1.f;
        std::string roomGroup;              // interiors: the loaded root's parent chain, hashed; plus the enable parent
        FormKey enableParent;               // empty plugin: none
        bool enableOpposite=false;
    };

    struct Job {
        uint32_t cell=0;
        FormKey cellKey;
        std::string editorID;
        bool interior=true;
        uint32_t worldspace=0;              // exteriors
        FormKey worldspaceKey;
        std::string worldspaceEditorID;
        std::array<int32_t,2> grid{};
        std::vector<Reference> references;
        // x, y, z, radius (base + XRDS): one per placed light reference, in value order. Interiors: the cell's lights;
        // exteriors: those of the cell and of its attached neighbours that can reach its chunks (Exterior::LightReachesCell).
        std::vector<std::array<double,4>> lights;
        // The lights the cache key covers: interiors all of them, exteriors the cell's own, so the neighbours that happened
        // to be loaded when the cell was gathered do not change its key.
        std::vector<std::array<double,4>> keyLights;
        uint32_t skippedWithout3D=0;
        uint32_t skippedRuntimeMaterial=0;  // live material differs from the NIF (runtime snow): left original
        std::array<float,3> origin{};       // nearest-first: the player at gather time, or a door's arrival point
        double distance=0;                  // queue order: exteriors by distance from the player at gather time
        bool prebake=false;                 // a load door's destination, gathered before the player enters
        // Exterior previs: a cell with nothing to merge (under two eligible references) or with a shipped manifest is
        // still handed to the exterior previs thread for its terrain, navmesh and occluders, so its neighbours' camera
        // cells can be rendered (such cells kept the cell around Whiterun's marker from ever being rendered).
        bool previsOnly=false;
        std::vector<std::array<float,9>> navTriangles;  // previs: the cell's live navmesh triangles (world xyz x 3)
        std::vector<float> terrain;         // exterior previs: 33 x 33 heights from the cell's south-west corner (empty: none)
        // Exterior previs: static references that occlude but are not merged (visible when distant, a material or
        // alternate textures, links): see Occluder.
        std::vector<Reference> occluders;
    };

    // Pure helpers, testable without the game.
    std::string NormalizeModel(std::string model);
    std::string HashHex(const void* data,size_t size,uint64_t seed=14695981039346656037ull);
    // A cell's bake inputs, hashed: the baker's output version, the settings, the references, the meshes and, with a light
    // budget, the lights; independent of the order the game lists references and lights in.
    std::string CacheKey(const Job& job,const std::vector<std::pair<std::string,std::string>>& meshHashes,const Settings& settings);
    // Indices into job.references, one list per part, each in the references' original order. Interiors: whole room
    // groups ordered by their nearest reference's distance from job.origin, packed into a first part of at least
    // `first` references and later parts of at least `later` (0: one part). Exteriors: one part. Because the baker
    // bins by material, grid and room group, a part produces exactly the chunks the whole-cell bake makes for its
    // groups (checked byte for byte for Ustengrav02).
    std::vector<std::vector<size_t>> PlanParts(const Job& job,size_t first,size_t later);
    // File-name key of a cell: its form ID (8 hex) for a Skyrim.esm cell, otherwise FNV-1a of "<plugin>|<local>" (16 hex).
    std::string CellKey(const FormKey& cell);

    void Configure(const Settings& settings);
    // SKSE kDataLoaded: starts the poller and worker when enabled. A cache folder that cannot be created leaves the bake
    // off for the session, logged; nothing is thrown for it.
    void Start();
    // SKSE kDataLoaded: reads the runtime-bake settings from SkyrimPrecombines.ini, then Configure and Start
    // (BakePlugin.cpp).
    void Load();
}
