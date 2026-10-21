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

// Skyrim VR 1.4.15 only: the engine's own occlusion culling, which VR ships switched off
// (bDepthBufferCulling:VRPerformance = 0). With it on, the GPU tests the oriented boxes of up to 4096 geometries
// against each frame's depth prepass, and the next frame skips the draws of those it found hidden (-43% draws in
// Ustengrav02; Community Shaders measured CPU 19.5 -> 11.3 ms and GPU 11.7 -> 8.4 ms in Whiterun).
//
// When Community Shaders is loaded it manages this setting itself, so nothing here is installed. Otherwise this plugin
// switches the culling on, fixes the bound copy the box test depends on, and restores hidden objects a moving view
// may uncover before the next test (the method of Community Shaders' "Advanced" mode).
namespace VRDepthCulling {
    // kPostLoad (every plugin loaded): reads the settings, applies the bound fix and installs the recovery hooks.
    void Install();
    // kDataLoaded: switches the culling on and sets the smallest tested box.
    void Apply();
    // Diagnostics ("vr_depth_culling_log"), once per frame from the Hi-Z Prepare on VR: the census of what the game's
    // culling and this plugin's Hi-Z each test, hide and cost, one log line every "vr_depth_culling_log_frames" frames.
    void OnFrame();
}
