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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <dxgiformat.h>
#include <vector>

// Hi-Z occlusion culling for the main camera only (SE 1.5.97, AE 1.6.1170 / 1.6.1179 GOG / 1.7.104; VR 1.4.15 behind
// "hiz_vr", off by default: both eyes, see Occlusion.cpp StereoProjection).
//
// After the engine's depth prepass, a compute shader reduces the prepass depth copy to a small grid of
// farthest depths, which is read back a frame or two later without stalling. Before the main view is
// culled, the newest grid is reprojected into the current camera and turned into a max pyramid. Fade
// nodes and appended geometry are skipped when that pyramid proves them hidden, but only in culling
// processes whose camera is the world root camera: reflections, shadow maps and cubemaps never see a
// skipped object, so they stay correct by construction.

// Engine-free part, also compiled by tests/hiz (define PRECOMBINES_OCCLUSION_REDUCE_ONLY before including):
// the reduction shader, its constants, the rendered area of the depth copy and the shader-view format.
namespace Occlusion::Reduce {
    // Reduces the rendered area of the prepass depth copy (Sizes.xy, anchored at the top left) to one farthest depth
    // per grid texel (Sizes.zw), so the grid always spans the viewport, NDC -1..1. Depths nearer than the exemption
    // (first-person arms, anything at the eye) count as far: they never hide anything. So do texels touching an
    // excluded rectangle (NDC x0, y0, x1, y1, y up): the screen bounds of actors and doors,
    // which must not hide what stands behind them once they move.
    // Stereo (Skyrim VR, Options.w = 2): the depth copy holds both eyes side by side, and the grid's left and right halves
    // reduce the left and right eye's halves. Each half spans its own eye's NDC, and its rectangles are in that NDC: the
    // first eye's in Excluded[0..63], the second's in Excluded[64..127], Options.z counting first + 256 x second.
    inline constexpr uint32_t kMaxExcluded=128, kMaxExcludedPerEye=kMaxExcluded/2;
    inline constexpr char kShader[]=R"(
Texture2D<float> Depth : register(t0);
RWTexture2D<float> Grid : register(u0);
cbuffer Params : register(b0) { uint4 Sizes; float4 Options; float4 Excluded[128]; };
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= Sizes.z || id.y >= Sizes.w) return;
    bool stereo = Options.w > 1.5;
    uint width = stereo ? Sizes.z / 2 : Sizes.z;
    uint eye = stereo && id.x >= width ? 1 : 0;
    uint gx = id.x - eye * width;
    uint total = (uint)Options.z;
    uint count = stereo ? min(eye ? total / 256 : total % 256, 64u) : min(total, 128u);
    if (count > 0) {
        float4 t = float4(gx * 2.0 / width - 1, 1 - (id.y + 1) * 2.0 / Sizes.w, (gx + 1) * 2.0 / width - 1, 1 - id.y * 2.0 / Sizes.w);
        for (uint i = 0; i < count; ++i) {
            float4 r = Excluded[eye * 64 + i];
            if (r.x <= t.z && r.z >= t.x && r.y <= t.w && r.w >= t.y) { Grid[id.xy] = 1; return; }
        }
    }
    uint x0 = id.x * Sizes.x / Sizes.z, x1 = max(x0 + 1, (id.x + 1) * Sizes.x / Sizes.z);
    uint y0 = id.y * Sizes.y / Sizes.w, y1 = max(y0 + 1, (id.y + 1) * Sizes.y / Sizes.w);
    float farthest = 0;
    for (uint y = y0; y < y1; ++y)
        for (uint x = x0; x < x1; ++x) {
            float d = Depth.Load(int3(x, y, 0));
            if (Options.y > 0.5) d = 1 - d;
            if (d < Options.x) d = 1;
            farthest = max(farthest, d);
        }
    Grid[id.xy] = farthest;
}
)";

    // kShader's constant buffer: Sizes = rendered width and height, grid width and height; Options = near-exemption
    // NDC depth, reversed-depth flag, number of excluded rectangles, 2 for stereo; Excluded = those rectangles.
    struct alignas(16) Params {
        uint32_t sizes[4];
        float options[4];
        float excluded[kMaxExcluded][4];
    };
    static_assert(sizeof(Params)==32+16*kMaxExcluded);

    // The part of the depth copy that this frame's prepass rendered.
    struct Area {
        uint32_t width=0,height=0;
        bool partial=false;  // dynamic resolution: smaller than the texture
        bool clamped=false;  // a ratio above 1 was clamped to the texture
    };

    // Dynamic resolution (the engine's own, Community Shaders upscaling, Skyrim Upscaler) renders the world into the
    // top-left widthRatio x heightRatio of full-size targets; the rest of the depth keeps its far clear value. The area
    // is the texture's own size times the ratios, rounded up: a partly covered column or row belongs to the area, and
    // one column past it holds the cleared far depth, which only makes the grid farther. A ratio that is not a finite
    // number in (0, 1] (above 1 is clamped), or an area below 1/8 of the texture, is unusable: skip the frame.
    inline bool RenderedArea(uint32_t textureWidth,uint32_t textureHeight,float widthRatio,float heightRatio,Area& area) {
        area=Area{};
        if(!textureWidth || !textureHeight || !std::isfinite(widthRatio) || !std::isfinite(heightRatio) || !(widthRatio>0.f) || !(heightRatio>0.f))
            return false;
        const auto side=[&](uint32_t size,float ratio) {
            if(ratio>=1.f) { area.clamped|=ratio>1.f+1e-4f; return size; }
            const double scaled=std::ceil(double(size)*double(ratio)-1e-3);
            return static_cast<uint32_t>(std::clamp(scaled,1.0,double(size)));
        };
        area.width=side(textureWidth,widthRatio);
        area.height=side(textureHeight,heightRatio);
        area.partial=area.width<textureWidth || area.height<textureHeight;
        return uint64_t(area.width)*8>=textureWidth && uint64_t(area.height)*8>=textureHeight;
    }

    inline Params MakeParams(const Area& area,uint32_t gridWidth,uint32_t gridHeight,float nearExemptNdc,bool reversed,bool stereo=false) {
        return Params{{area.width,area.height,gridWidth,gridHeight},{nearExemptNdc,reversed ? 1.f:0.f,0.f,stereo ? 2.f:0.f},{}};
    }

    // Adds one excluded rectangle; rectangles wholly off screen are dropped. False when the list is full.
    inline bool Exclude(Params& params,const float rect[4]) {
        if(rect[2]<-1.f || rect[0]>1.f || rect[3]<-1.f || rect[1]>1.f) return true;
        const auto count=static_cast<uint32_t>(params.options[2]);
        if(count>=kMaxExcluded) return false;
        for(int k=0;k<4;++k) params.excluded[count][k]=rect[k];
        params.options[2]=static_cast<float>(count+1);
        return true;
    }
    // Stereo: one excluded rectangle in eye `eye`'s NDC (0 or 1); false when that eye's kMaxExcludedPerEye are taken.
    inline bool ExcludeEye(Params& params,uint32_t eye,const float rect[4]) {
        if(rect[2]<-1.f || rect[0]>1.f || rect[3]<-1.f || rect[1]>1.f) return true;
        const auto total=static_cast<uint32_t>(params.options[2]);
        const uint32_t count=eye ? total/256 : total%256;
        if(count>=kMaxExcludedPerEye) return false;
        for(int k=0;k<4;++k) params.excluded[eye*kMaxExcludedPerEye+count][k]=rect[k];
        params.options[2]=static_cast<float>(total+(eye ? 256u : 1u));
        return true;
    }

    // Format of a shader view that reads the depth plane of a depth texture, or DXGI_FORMAT_UNKNOWN. Only the
    // typeless formats a depth-stencil texture with a shader view must have are accepted; anything else in the
    // engine's depth slot is not the prepass depth (vanilla SE: R24G8_TYPELESS).
    inline DXGI_FORMAT DepthViewFormat(DXGI_FORMAT textureFormat) {
        switch(textureFormat) {
        case DXGI_FORMAT_R24G8_TYPELESS: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
        case DXGI_FORMAT_R32G8X24_TYPELESS: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
        case DXGI_FORMAT_R32_TYPELESS: return DXGI_FORMAT_R32_FLOAT;
        case DXGI_FORMAT_R16_TYPELESS: return DXGI_FORMAT_R16_UNORM;
        default: return DXGI_FORMAT_UNKNOWN;
        }
    }
}

// The GPU box test (developer "box_occlusion", off by default), engine-free part: Skyrim VR's own depth-buffer culling
// tests each geometry's box against the prepass depth and hid 3-5 times as many draws as the sphere test of the Hi-Z grid;
// SE and AE keep only a sphere per geometry, so the boxes come from each mesh's own vertices (Occlusion.cpp).
// After Main::RenderDepth the depth copy is reduced to a depth target of farthest depths (kFillShader), eroded for the eye's
// motion (kErodeShader), and every box the main view registered this frame is drawn against it with the depth test on
// (kBoxShader): a box any of whose fragments passes is visible. The results are read back a frame or two later without stalling.
// The false-cull self-check (developer "hiz_audit"), engine-free part. Hi-Z hides an object only when every texel its sphere
// covers is nearer than the sphere's nearest point, and the box test only when no fragment of its box passed an older depth.
// So on an audit frame every sphere a culler hid is drawn against that same frame's prepass depth, each pixel's view ray is
// intersected with the sphere, and where the sphere's front lies in front of the depth there the cull was wrong: the sphere
// (and with it, possibly, the object) was in view. Image comparisons of ON against OFF captures missed a missing rock of
// 18x15 pixels among swaying trees near Fort Amol; this counts such culls directly, at any draw distance.
namespace Occlusion::Audit {
    // Spheres per audit frame; a sphere with at least kMinPixels pixels in front of the depth is a false cull, kMargin the
    // share of view depth (plus a unit) it must be in front by, above the depth buffer's precision at 20,000 units.
    inline constexpr uint32_t kMaxSpheres=4096, kMinPixels=2;
    inline constexpr float kMargin=0.002f;
    enum Culler : uint8_t { kHiZGeometry, kHiZReference, kBoxTest, kCullers };
    // Each instance is one sphere (centre relative to the eye of the frame's culling camera, radius), drawn as its bounding
    // cube (both faces); the pixel shader adds one to the sphere's count where its ray meets the sphere in front of the
    // depth copy (standard depth after Model.z's flip: view depth = Model.y / (z - Model.x)). Area: the rendered area's width
    // and height in pixels and their reciprocals; FromClip inverts ToClip.
    inline constexpr char kShader[]=R"(
StructuredBuffer<float4> Spheres : register(t0);
Texture2D<float> Depth : register(t0);
cbuffer Params : register(b0) { row_major float4x4 ToClip; row_major float4x4 FromClip; float4 Area; float4 Model; };
RWByteAddressBuffer Visible : register(u0);
struct Fragment { float4 position : SV_Position; nointerpolation float4 sphere : SPHERE; nointerpolation uint index : INDEX; };
static const uint kIndex[36] = { 0,2,1, 1,2,3, 4,5,6, 5,7,6, 0,1,4, 1,5,4, 2,6,3, 3,6,7, 0,4,2, 2,4,6, 1,3,5, 3,7,5 };
Fragment vs(uint id : SV_VertexID, uint sphere : SV_InstanceID) {
    uint c = kIndex[id];
    float4 s = Spheres[sphere];
    float3 corner = s.xyz + s.w * float3((c & 1) ? 1 : -1, (c & 2) ? 1 : -1, (c & 4) ? 1 : -1);
    Fragment f;
    f.position = mul(ToClip, float4(corner, 1));
    f.sphere = s;
    f.index = sphere;
    return f;
}
void ps(Fragment f) {
    uint2 p = uint2(f.position.xy);
    float d = Depth.Load(int3(p, 0));
    if (Model.z > 0.5) d = 1 - d;
    float2 ndc = float2((p.x + 0.5) * Area.z * 2 - 1, 1 - (p.y + 0.5) * Area.w * 2);
    float4 a = mul(FromClip, float4(ndc, 0, 1)), b = mul(FromClip, float4(ndc, 1, 1));
    float3 o = a.xyz / a.w, dir = normalize(b.xyz / b.w - o), oc = o - f.sphere.xyz;
    float half_b = dot(oc, dir), disc = half_b * half_b - (dot(oc, oc) - f.sphere.w * f.sphere.w);
    if (disc < 0) return;
    float w = mul(ToClip, float4(o + dir * max(0, -half_b - sqrt(disc)), 1)).w;
    float scene = d < 1 ? Model.y / (d - Model.x) : 3.0e38;
    if (w < scene * (1 - Model.w) - 1) Visible.InterlockedAdd(f.index * 4, 1);
}
)";
    struct alignas(16) Params { float toClip[4][4]; float fromClip[4][4]; float area[4]; float model[4]; };
    static_assert(sizeof(Params)==160);
}

namespace Occlusion::Boxes {
    inline constexpr uint32_t kMaxBoxes=16384;
    // One box relative to the eye of the frame's culling camera (Occlusion.cpp RegisterBox): the centre and its three
    // half-extent axes (w unused). Relative, because a world-space float projection far from the origin is off by about
    // 0.15% in depth, close to a box's own growth. An empty slot is all zeros and rasterizes nothing, so it
    // reads as hidden and is ignored (it names no object).
    struct alignas(16) Box { float center[4]; float axis[3][4]; };
    static_assert(sizeof(Box)==64);
    // False for a slot that holds no box: an axis of zero length (a slot taken but not yet written, or an empty one) draws
    // nothing, which reads as hidden.
    inline bool Solid(const Box& box) {
        for(const auto& axis:box.axis) if(axis[0]==0.f && axis[1]==0.f && axis[2]==0.f) return false;
        return true;
    }
    // The depth target: one farthest standard depth per texel of the depth copy's rendered area (Sizes.xy) reduced to
    // the target (Sizes.zw), with Reduce::kShader's near exemption, reversed depth and excluded rectangles
    // (Reduce::Params), written as SV_Depth by one full-screen triangle (vs, ps).
    inline constexpr char kFillShader[]=R"(
Texture2D<float> Depth : register(t0);
cbuffer Params : register(b0) { uint4 Sizes; float4 Options; float4 Excluded[128]; };
float4 vs(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
float ps(float4 position : SV_Position) : SV_Depth {
    uint2 t = uint2(position.xy);
    uint count = min((uint)Options.z, 128u);
    if (count > 0) {
        float4 own = float4(t.x * 2.0 / Sizes.z - 1, 1 - (t.y + 1) * 2.0 / Sizes.w, (t.x + 1) * 2.0 / Sizes.z - 1, 1 - t.y * 2.0 / Sizes.w);
        for (uint i = 0; i < count; ++i) {
            float4 r = Excluded[i];
            if (r.x <= own.z && r.z >= own.x && r.y <= own.w && r.w >= own.y) return 1;
        }
    }
    uint x0 = t.x * Sizes.x / Sizes.z, x1 = max(x0 + 1, (t.x + 1) * Sizes.x / Sizes.z);
    uint y0 = t.y * Sizes.y / Sizes.w, y1 = max(y0 + 1, (t.y + 1) * Sizes.y / Sizes.w);
    float farthest = 0;
    for (uint y = y0; y < y1; ++y)
        for (uint x = x0; x < x1; ++x) {
            float d = Depth.Load(int3(x, y, 0));
            if (Options.y > 0.5) d = 1 - d;
            if (d < Options.x) d = 1;
            farthest = max(farthest, d);
        }
    return farthest;
}
)";
    // Each instance is one box (36 vertices, both faces drawn); a fragment that passes the early depth test marks its
    // box visible. Standard depth: the clip matrix is the main camera's standard projection (Occlusion.cpp Projection) for
    // points relative to the eye the boxes are given relative to (HiZ::Relative).
    inline constexpr char kBoxShader[]=R"(
struct Box { float4 center; float4 axis[3]; };
StructuredBuffer<Box> Boxes : register(t0);
cbuffer Params : register(b0) { row_major float4x4 ToClip; uint4 Count; };
RWByteAddressBuffer Visible : register(u0);
struct Fragment { float4 position : SV_Position; nointerpolation uint box : BOX; };
static const uint kIndex[36] = { 0,2,1, 1,2,3, 4,5,6, 5,7,6, 0,1,4, 1,5,4, 2,6,3, 3,6,7, 0,4,2, 2,4,6, 1,3,5, 3,7,5 };
Fragment vs(uint id : SV_VertexID, uint box : SV_InstanceID) {
    uint c = kIndex[id];
    float3 s = float3((c & 1) ? 1 : -1, (c & 2) ? 1 : -1, (c & 4) ? 1 : -1);
    Box b = Boxes[box];
    float3 at = b.center.xyz + s.x * b.axis[0].xyz + s.y * b.axis[1].xyz + s.z * b.axis[2].xyz;
    Fragment f;
    f.position = mul(ToClip, float4(at, 1));
    f.box = box;
    return f;
}
[earlydepthstencil] void ps(Fragment f) { Visible.Store(f.box * 4, 1); }
)";
    // toClip: HiZ::Relative of the main camera's world-to-clip for the boxes' eye.
    struct alignas(16) BoxParams { float toClip[4][4]; uint32_t count[4]; };
    static_assert(sizeof(BoxParams)==80);

    // Motion erosion. A result drawn from eye E is used while the eye stays within S of E (box_occlusion_max_shift) for up to
    // 4 frames, and moving uncovers what lay behind an occluder's edge: through the gaps between the planks of stairs and
    // bridges, ground went missing while walking and came back when standing still or with the box test off.
    // So before the boxes are drawn, every texel of the depth target that an eye within S of E might see past an edge counts
    // as far:
    //  - A cliff is a texel nearer than one of its eight neighbours by more than kCliffRatio in view depth (or than a far one),
    //    or by enough that the two shift against each other by over a texel for an eye moved by S (f S k (1/d - 1/d_n) > 1,
    //    k as below at the texel), or on the border of the target (nothing is known beyond it). Its depth goes into a mip
    //    chain of the nearest cliff per block (edges, down).
    //  - A texel t is exposed when a cliff n lies within f S k / d_n + 2 texels of it: d_n is the cliff's view depth, f the
    //    target's texels per unit of view tangent, k = sqrt(1 + a^2) with a = ((|tangent at t| + 3 / f) z0 + S) / (z0 - S),
    //    and z0 the nearest view depth a registered box reaches (the near plane or the near exemption, at least 2S:
    //    MotionShift). The search (expose) runs per level-1 texel and writes 1 where it finds one; apply makes those far.
    // Why no box an eye within S sees is hidden. Let q be a point of a geometry visible from E' = E + s, |s| <= S, that from E
    // lies behind the depth target at its texel; its view depth z > z0. In E's view the line of sight from E' to q, at view
    // depth w, has the view tangent a + (s_xy - s_z a) / w (a the tangent of its direction): as w falls from z the line runs
    // straight across the screen, at f |s_xy - s_z a| (1/w - 1/z) < f S k / w texels from q (|s_xy - s_z a| <= |s|
    // sqrt(1 + |a|^2), and |a| <= (|tangent of q| z + S) / (z - S), which is largest at z0). It starts behind the target and
    // must end in front of it (every depth in the target is beyond z0 > S), so at some depth w it passes from a texel nearer
    // than w to a neighbour farther than w, or leaves the screen behind a border texel nearer than w. If that step is no
    // cliff, its two depths shift against each other by at most a texel for an eye within S, so a q between them (z up to
    // the farther depth) is within a texel of the step, where its grown box covers the farther texel's centre: q's box is
    // visible from E already. A q beyond both depths would need a second opening hidden behind the near texel. Here lies the
    // one assumption: a step of less than 6.25% in view depth is one continuous surface, or a background that continues
    // behind the nearer one (the occluders themselves stay put, as moving references are cut out of the target and a
    // geometry whose bound changed gets no result). Otherwise the near texel is a cliff with d_n < w, within f S k / d_n + 1
    // texels of q's texel. The other texel of slack covers the grown box's texel centre near q (RegisterBox grows boxes by
    // about a texel). That texel is exposed, so q's box is visible. The erosion only ever makes texels far: it never hides
    // what the plain target shows. (The texel rule: with 6.25% alone, slats near the eye with a wall a few
    // units behind could hide from a moved eye what lay between them.)
    // The search: a mip texel holds the nearest cliff of its block (the last row and column of a level take what is left
    // over). For the level-1 texel T and each level L >= 1, the 5x5 blocks around the one holding T cover everything within 2
    // blocks of it, so a cliff not found at level L - 1 is more than 2^L texels from T (L > 1), and no cliff in a block is
    // nearer than the block. A block with cliff depth d whose distance bound D gives d (D - 2) < reach (f S k at T's farthest
    // corner) exposes T: a cliff that exposes a texel of T passes this at the first level that holds it. Above the level
    // where reach < z0 (D - 2), nothing can pass (no cliff is nearer than z0) and the search stops. Compared with an exact
    // search over every cliff, this exposes 1.3-1.5 times the area and misses nothing (a CPU model of these passes on the
    // HiZTest scenes).
    inline constexpr char kErodeShader[]=R"(
Texture2D<float> Input : register(t0);
RWTexture2D<float> Output : register(u0);
cbuffer Params : register(b0) { uint4 Sizes; float4 Model; float4 Frustum; float4 Reach; };
float View(float z) { return z < 1 ? Model.y / (z - Model.x) : 3.0e38; }
[numthreads(8, 8, 1)]
void edges(uint3 id : SV_DispatchThreadID) {
    int2 size = int2(Sizes.xy), at = int2(id.xy);
    if (at.x >= size.x || at.y >= size.y) return;
    float z = Input.Load(int3(at, 0)), cliff = 1;
    if (z < 1) {
        float d = View(z), limit = d * Reach.w;
        // f S k at this texel, with expose's bound on a.
        float2 tangent = float2(Frustum.x + (at.x + 0.5) * (Frustum.y - Frustum.x) / size.x, Frustum.z - (at.y + 0.5) * (Frustum.z - Frustum.w) / size.y);
        float a = ((length(tangent) + 3 / Reach.x) * Reach.z + Reach.y) / (Reach.z - Reach.y);
        float shift = Reach.x * Reach.y * sqrt(1 + a * a);
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int2 p = at + int2(dx, dy);
                float n = p.x < 0 || p.y < 0 || p.x >= size.x || p.y >= size.y ? 1 : Input.Load(int3(p, 0));
                if (n >= 1 || View(n) > limit || shift * (1 / d - 1 / View(n)) > 1) cliff = z;
            }
    }
    Output[at] = cliff;
}
[numthreads(8, 8, 1)]
void down(uint3 id : SV_DispatchThreadID) {
    uint w, h, sw, sh, levels;
    Output.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    Input.GetDimensions(0, sw, sh, levels);
    uint x1 = id.x + 1 == w ? sw - 1 : 2 * id.x + 1, y1 = id.y + 1 == h ? sh - 1 : 2 * id.y + 1;
    float m = 1;
    for (uint y = 2 * id.y; y <= y1; ++y)
        for (uint x = 2 * id.x; x <= x1; ++x) m = min(m, Input.Load(int3(x, y, 0)));
    Output[id.xy] = m;
}
[numthreads(8, 8, 1)]
void expose(uint3 id : SV_DispatchThreadID) {
    uint tw, th, levels, mw, mh;
    Input.GetDimensions(0, tw, th, levels);
    Output.GetDimensions(mw, mh);
    if (id.x >= mw || id.y >= mh) return;
    int w = (int)tw, h = (int)th;
    int2 lo = int2(id.xy) * 2, hi = int2(id.x + 1 == mw ? w - 1 : lo.x + 1, id.y + 1 == mh ? h - 1 : lo.y + 1);
    float reach = 0;
    for (uint c = 0; c < 4; ++c) {
        float2 at = float2((c & 1) ? hi.x : lo.x, (c & 2) ? hi.y : lo.y) + 0.5;
        float2 tangent = float2(Frustum.x + at.x * (Frustum.y - Frustum.x) / w, Frustum.z - at.y * (Frustum.z - Frustum.w) / h);
        float a = ((length(tangent) + 3 / Reach.x) * Reach.z + Reach.y) / (Reach.z - Reach.y);
        reach = max(reach, Reach.x * Reach.y * sqrt(1 + a * a));
    }
    float exposed = 0;
    for (uint level = 1; level < levels && exposed == 0; ++level) {
        uint lwu, lhu, unused;
        Input.GetDimensions(level, lwu, lhu, unused);
        int lw = (int)lwu, lh = (int)lhu, size = 1 << level;
        float annulus = level == 1 ? 0 : (float)size;
        if (annulus > 2 && reach < Reach.z * (annulus - 2)) break;
        int2 at = min(lo >> level, int2(lw - 1, lh - 1));
        for (int by = max(at.y - 2, 0); by <= min(at.y + 2, lh - 1); ++by)
            for (int bx = max(at.x - 2, 0); bx <= min(at.x + 2, lw - 1); ++bx) {
                float m = Input.Load(int3(bx, by, level));
                if (!(m < 1)) continue;
                int2 b0 = int2(bx, by) * size, b1 = int2(bx == lw - 1 ? w - 1 : b0.x + size - 1, by == lh - 1 ? h - 1 : b0.y + size - 1);
                float2 gap = (float2)max(max(b0 - hi, lo - b1), 0);
                if (View(m) * (max(length(gap), annulus) - 2) < reach) exposed = 1;
            }
    }
    Output[id.xy] = exposed;
}
float apply(float4 position : SV_Position) : SV_Depth {
    uint mw, mh;
    Input.GetDimensions(mw, mh);
    uint2 at = min(uint2(position.xy) / 2, uint2(mw - 1, mh - 1));
    if (Input.Load(int3(at, 0)) < 0.5) discard;
    return 1;
}
)";
    // kErodeShader's constant buffer: Sizes = target width, height and mip levels; Model = depth model a, b (NDC depth
    // a + b / w); Frustum = view tangents left, right, top, bottom; Reach = target texels per unit of tangent f, the shift S,
    // the nearest box depth z0, kCliffRatio.
    struct alignas(16) ErodeParams { uint32_t sizes[4]; float model[4]; float frustum[4]; float reach[4]; };
    static_assert(sizeof(ErodeParams)==64);
    inline constexpr float kCliffRatio=1.0625f;
    // Levels of a full mip chain down to 1x1.
    inline uint32_t Levels(uint32_t width,uint32_t height) {
        uint32_t levels=1;
        for(uint32_t side=std::max(width,height);side>1;side>>=1) ++levels;
        return levels;
    }
    // The shift a result is eroded for and then trusted with: box_occlusion_max_shift, at most half the nearest depth a box
    // reaches (the bound on a needs z0 >= 2S); 0, no motion at all, for anything but a positive number.
    inline float MotionShift(float shift,float nearest) { return shift>0.f && nearest>0.f ? std::min(shift,0.5f*nearest) : 0.f; }
    inline ErodeParams MakeErodeParams(uint32_t width,uint32_t height,float a,float b,float left,float right,float top,float bottom,float shift,float nearest) {
        const float f=std::max(float(width)/(right-left),float(height)/(top-bottom));
        return ErodeParams{{width,height,Levels(width,height),0},{a,b,0.f,0.f},{left,right,top,bottom},{f,MotionShift(shift,nearest),nearest,kCliffRatio}};
    }
    // The depth target's side for a rendered side and the downsample factor (rounded up).
    inline uint32_t TargetSide(uint32_t side,uint32_t factor) { return factor ? std::max(1u,(side+factor-1)/factor) : side; }
    // A texel of the depth target in world units per unit of view distance, for a view frustum spanning spanX x spanY at
    // unit distance (tangents). A box thinner than two texels on screen can fall between texel centres and draw no
    // fragment at all, which reads as hidden: every half extent is made at least distance x TexelSpan, so the box holds a
    // sphere two texels across and always rasterizes.
    inline float TexelSpan(float spanX,float spanY,uint32_t width,uint32_t height) {
        return width && height ? std::max(spanX/float(width),spanY/float(height)) : 0.f;
    }
}

#ifndef PRECOMBINES_OCCLUSION_REDUCE_ONLY
namespace Occlusion {
    struct Settings {
        bool enabled=false;     // config "hiz_occlusion"
        bool fadeNodes=true;    // "hiz_fade_nodes": skip whole fade-node subtrees (references)
        uint32_t fadeRule=1;    // "hiz_fade_rule": which fade nodes may be skipped whole (Occlusion.cpp FadeSafe)
        bool geometry=true;     // "hiz_geometry": test each appended geometry (merged chunks, parts)
        // "hiz_grid_width"; the height is half. 128: the main view's culling jobs start about 160 us after the pyramid
        // build, and a 256x128 build took 245-310 us, so about 670 of 700 Ustengrav objects went untested;
        // a 128x64 build is published at 90-130 us and culled as much (-29% and -11% draws at two views).
        uint32_t gridWidth=128;
        float nearExempt=64.f;  // "hiz_near_exempt": never cull within this distance of the eye
        bool excludeMoving=true;  // "hiz_exclude_moving": actors and doors never occlude
        uint32_t waitMicros=0;    // "hiz_wait_us": how long culling tests wait for this frame's pyramid (0: never wait)
        bool splitBuild=false;    // "hiz_split_build": build the pyramid on two threads (slower under load)
        // The two gates' startup values; the in-game menu changes them in play (SetMinFrameMs, SetMinYield).
        float minFrameMs=5.f;    // "hiz_min_frame_ms": work only while the smoothed frame time is at least this (0: always)
        float minYield=0.01f;     // "hiz_min_yield": below this share of tested objects hidden, test only in probe bursts (0: always test)
        uint32_t probeInterval=32, probeBurst=4;  // "hiz_probe_interval", "hiz_probe_burst": frames per probe cycle, active frames in it
        // "hiz_visible_retest": an object found visible is tested again after 1 to this many frames (HiZ::VisibleCache);
        // 0 or 1 tests every object every frame.
        uint32_t visibleRetest=8;
        // "hiz_diagnostics" (developer log): time the culling hooks every 8th frame and name the actors behind widened
        // near exemptions. It costs about 0.01-0.02 ms per frame while Hi-Z is on, so timing runs leave it off.
        bool diagnostics=false;
        // "hiz_audit" (developer): every "hiz_audit_every"-th frame, check every sphere a culler hid against that frame's
        // own depth (Occlusion::Audit) and log the false culls. It draws on the GPU on those frames: timing runs leave it off.
        bool audit=false;
        uint32_t auditEvery=8;
        bool frameLog=false;      // "frame_log": keep every main-view frame's interval for TakeFrameTimes
        // "hiz_ab" (developer): while Hi-Z is on, every block of abFrames frames runs one of four states, in a shuffled
        // order per round of four blocks: off (no capture, nothing culled), as in 1.0, as before (the original occlusion
        // test, no visible-object table, shared statistics counters), and eligibility first (as in 1.0 with the
        // eligibility checks before the visible-object table, as in runtime 5dd7002c). The frame-time and yield gates are
        // ignored. The first abSkip frames of a block are not counted (a stale grid, a cold table, the previous
        // state's last frames still in flight). "hiz_ab_frames", "hiz_ab_skip".
        bool ab=false;
        uint32_t abFrames=60, abSkip=8;
        // "hiz_dump_seconds" (developer, the grid-size study; 0: off): every that many seconds one frame is kept for
        // offline replay: its main-view culling records every Hi-Z candidate and culls nothing,
        // and its whole depth copy is read back; both go into SkyrimPrecombines_hiz_NNN.bin in the log directory, at most
        // "hiz_dump_max" files.
        float dumpSeconds=0.f;
        uint32_t dumpMax=40;
        // "box_occlusion" (developer, off by default; SE and AE): the GPU box test (Occlusion::Boxes). A geometry whose
        // box was hidden in the newest result is left out of the main view while the eye has moved at most
        // "box_occlusion_max_shift" units since that frame (at most half the larger of the near plane and the near exemption:
        // Boxes::MotionShift) and the result is at most 4 frames old. Results are a frame or two old: turning the view never
        // uncovers anything (occlusion from one eye position does not depend on where it looks; boxes near the screen edge are
        // not tested), moving does, near an occluder's edge. So the depth target is eroded for that shift first
        // (Boxes::kErodeShader): what an eye that far away could see past an edge counts as visible, up to S f k / d texels
        // from an edge at view depth d (f about 500 texels per unit of tangent at 1080p and downsample 2, k 1 to 2.4), and the
        // smaller the shift, the more of the nearer occluders still hide something (16: with 32 and no erosion, ground went missing through
        // stair and bridge gaps while walking; Precombines.cpp reads the same default). SPS allows 64
        // units, Bottled Shaders a motion margin. The depth target is the rendered area divided by "box_occlusion_downsample"
        // (1-8).
        bool boxes=false;
        float boxMaxShift=16.f;
        uint32_t boxDownsample=2;
    };
    struct Stats {
        bool enabled=false, calibrated=false;
        uint64_t captures=0, builds=0;
        // Last completed frame.
        uint32_t fadeTested=0, fadeCulled=0, geometryTested=0, geometryCulled=0;
        uint32_t prepareMicros=0, gridAge=0, landed=0;
        uint32_t previsCulled=0;  // geometries previs kept out of the main view in the last frame
        uint32_t referencePrevisCulled=0;  // references (fade nodes, trees) reference previs skipped in the last frame
        uint32_t buildMicros=0;   // pyramid reprojection and mips on the helper thread
        uint64_t lateBuilds=0, skippedFrames=0;  // pyramids finished after their frame; frames the helper was still busy
        uint64_t skippedCaptures=0;  // frames whose depth could not be captured (unusable depth copy or resolution state)
        uint32_t depthWidth=0, depthHeight=0;  // rendered area of the latest capture (dynamic resolution)
        uint32_t movingExcluded=0;  // moving references whose screen bounds the latest capture excluded
        uint64_t movingSkipped=0;   // captures dropped because more moving references were visible than the list holds
        uint64_t eyeExempted=0;     // captures whose near exemption was widened for a moving reference at the eye
        uint64_t eyeAtEye=0, eyeRect=0;  // of those: a reference at the eye; bounds reaching the camera plane
        uint64_t boundsReplaced=0;  // actor root bounds larger than 1.5x the actor's box plus 64 units, replaced by that box
        uint32_t untested=0;        // last frame: main-camera tests made while the frame's pyramid was still being built
        uint32_t skipped=0;         // last frame: recently visible objects not tested again (hiz_visible_retest)
        uint32_t fadeKept=0;        // last frame: fade nodes not skipped whole so their fade stays the engine's (hiz_fade_rule)
        uint64_t waits=0, waitTimeouts=0;  // culling tests that waited for the frame's pyramid; of those, waits that timed out
        uint64_t fastFrames=0;  // frames skipped because the frame rate was above the hiz_min_frame_ms threshold
        uint64_t yieldSkipped=0;  // frames skipped between probe bursts because Hi-Z hid too little (hiz_min_yield)
        float frameMs=0.f;      // smoothed frame time seen by Hi-Z
        // The box test ("box_occlusion"). Last frame: boxes registered; geometries it hid; main-view geometries without a
        // box (no vertex data or another geometry type); boxes not registered because they reach the screen edge or the
        // near plane. The hidden boxes in the result in use, and since the start: box passes drawn, results read back,
        // frames without a usable result because the eye moved too far or the result was too old.
        bool boxes=false;
        uint32_t boxRegistered=0, boxCulled=0, boxNoModel=0, boxEdge=0, boxTableHidden=0;
        uint64_t boxPasses=0, boxReadbacks=0, boxShifted=0, boxStale=0;
        // Of the geometries without a box last frame: another geometry type or no renderer data; vertices still on their
        // way (a GPU copy queued or in flight); vertices that do not fit the model bound. Since the start: boxes made from
        // GPU copies of vertex buffers, copies refused, boxes from the engine's CPU copy, requests dropped, requests
        // waiting now, and geometries without a box by BSGeometry::Type.
        uint32_t boxType=0, boxWaiting=0, boxInvalid=0, boxQueued=0;
        uint64_t boxMeshReads=0, boxMeshInvalid=0, boxCpuBoxes=0, boxMeshDropped=0;
        std::array<uint64_t,16> boxTypes{};
    };
    // At plugin load: installs the hooks (inert until enabled). SE and AE. Hooks another plugin installed first
    // are chained: their entry is called in place of the engine's, and the log names the module that owns each one.
    // main.cpp does not call it on SE and AE when nothing that needs the hooks can run this session (HooksUnneeded: Hi-Z,
    // previs, the box test, developer mode and the test logs all off); a new setting that needs them must be added there.
    void Install();
    // True when the main draw hooks and the main view's culler (BSGeometryListCullingProcess::AppendVirtual) are
    // installed: previs can then hide chunks from the main camera only.
    bool Installed();
    // Chunk roots that previs hides from the main camera only: their geometry is never appended to the main
    // view, while reflections, shadow maps and cubemaps still draw it. The snapshot keeps every root alive.
    void SetPrevisHidden(std::vector<RE::NiPointer<RE::NiAVObject>> roots);
    // Reference previs (all references, as in Fallout 4): form IDs of references hidden from the main camera, valid
    // while the camera stays within one view cell (edge `size`) of `cell`, the view cell they were computed for.
    // Each frame the camera's own view cell is checked; anywhere else nothing is hidden until the next update.
    void SetReferencePrevisHidden(std::vector<uint32_t> forms,const std::array<int64_t,3>& cell,float size);
    void Configure(const Settings& settings);
    // When a game is loaded or started (other plugins' menus change their settings in play): Hi-Z occlusion pauses while
    // Bottled Shaders' HiZ Occlusion Culling culls or SPS's "Hide occluded objects" is on, and returns to the configured
    // state when neither does. Reported in the log only.
    void CheckOtherCullers();
    // Hi-Z on or off from the in-game menu and the Hi-Z shortcut key (through Settings). While another plugin's occlusion
    // culling is on, Hi-Z stays paused and resumes with this value. Any thread.
    void SetEnabled(bool on);
    // The gates "hiz_min_frame_ms" (0-100) and "hiz_min_yield" (0-1), live: read from the next frame. Non-finite values are
    // ignored. Any thread.
    void SetMinFrameMs(float ms);
    void SetMinYield(float share);
    // SetEnabled with the opposite of the value set last; returns whether Hi-Z runs now. Settings does not see this change:
    // the shortcut key and the menu go through Settings::SetBool("bHiZ").
    bool Toggle();
    // Developer: switch the per-geometry test on or off (reference roots are always tested); returns the new state.
    bool ToggleGeometry();
    // Developer: switch the GPU box test ("box_occlusion") on or off; returns the new state.
    bool ToggleBoxes();
    // Developer ("hiz_probe_key", needs "hiz_diagnostics"): the next frame with a pyramid logs every geometry whose screen
    // rectangle covers the screen centre, with what Hi-Z decided and why. False without diagnostics.
    bool Probe();
    Stats Read();
    // Diagnostics since the previous call (developer log only; each call resets them). Per sampled frame (one in 8):
    // microseconds in the Hi-Z test, in the moving-reference notes and in the previs lookups of the culling hooks, and
    // Hi-Z tests per frame. Of the sampled frames that started a pyramid build and tested: those whose first test came
    // before the pyramid was published, and the mean time from the build start to publication and to the first test.
    struct Cost {
        uint64_t sampledFrames=0, raceFrames=0, raceLost=0;
        double hizMicros=0.0, movingMicros=0.0, previsMicros=0.0, hizCallsPerFrame=0.0, publishMicros=0.0, firstTestMicros=0.0;
        // hizMicros in stages: camera (every call), waiting for the pyramid and the visible-object table (main-camera
        // calls), eligibility and the occlusion test (calls past the table); main-camera calls and calls past the table
        // per sampled frame (candidatesPerFrame, testsPerFrame).
        double stageMicros[4]{}, candidatesPerFrame=0.0, testsPerFrame=0.0;
        float eyeWidest=0.f;  // widest near exemption since the previous call (units)
    };
    Cost ReadCost();
    // "hiz_diagnostics": since the previous call, the render thread's CPU time in Capture (the reduction's dispatch and
    // readback copy) and in Prepare (readback and job hand-off), the helper's pyramid builds, and the reduction's GPU
    // time (timestamp queries; frames whose queries have not come back are left out). Means per call, in microseconds.
    struct CaptureCost {
        uint64_t captures=0, prepares=0, builds=0, gpuSpans=0;
        double captureMicros=0.0, prepareMicros=0.0, buildMicros=0.0, gpuMicros=0.0;
    };
    CaptureCost TakeCaptureCost();
    // "frame_log": the main-view frame intervals (ms) since the previous call, moved into `out` (which is cleared first).
    void TakeFrameTimes(std::vector<float>& out);
    enum class AbState : uint8_t { Off=0, New=1, Old=2, EligibleFirst=3 };
    // One finished block of "hiz_ab": its round, state, counted frames and their times, Hi-Z tests and objects hidden
    // over those frames, and the camera position at its end.
    struct AbBlock {
        uint64_t round=0;
        AbState state=AbState::New;
        uint32_t frames=0;
        double sumMs=0.0, maxMs=0.0;
        uint64_t tested=0, culled=0;
        std::array<float,3> eye{};
    };
    // The blocks finished since the previous call, moved into `out` (cleared first).
    void TakeAbBlocks(std::vector<AbBlock>& out);
}
#endif
