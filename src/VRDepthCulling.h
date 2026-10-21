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
