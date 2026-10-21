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
#include <atomic>
#include <memory>
#include <cstdint>
#include <functional>
#include <vector>

// Hierarchical-depth (Hi-Z) occlusion test. Engine-independent, so tests/hiz exercises it natively.
// Conventions: clip = m * (x, y, z, 1) with rows as output components; NDC depth runs from 0 at the
// near plane to 1 at the far plane (a reversed projection is flipped before it gets here); grid row 0
// is the top of the screen.
namespace HiZ {
    struct Mat4 {
        float m[4][4]{};
        static Mat4 Identity();
    };
    Mat4 Multiply(const Mat4& a,const Mat4& b);
    Mat4 Transpose(const Mat4& a);
    bool Invert(const Mat4& in,Mat4& out);
    // Local-to-world matrix of a rotation (rows of r), uniform scale and translation.
    Mat4 FromTransform(const float r[3][3],float scale,const float t[3]);

    // The same in double. World coordinates reach hundreds of thousands of units: a reprojection composed in float from two
    // world-to-clip matrices there is off by 1e-4 to 1e-3 in NDC depth even for a camera standing still, which reads
    // occluders several percent too near at 3,000 units and tens of percent at 20,000 (Fort Amol, about (98000,-18500):
    // rocks in clear view hidden). Matrices built from the camera pose in double and composed in double are
    // rounded to float once, near the identity.
    struct Mat4d {
        double m[4][4]{};
    };
    Mat4d ToDouble(const Mat4& a);
    Mat4 ToFloat(const Mat4d& a);
    Mat4d Multiply(const Mat4d& a,const Mat4d& b);
    bool Invert(const Mat4d& in,Mat4d& out);
    // Captured clip space to current clip space: current times the inverse of captured, in double, rounded once. False
    // when captured cannot be inverted, an entry is not finite, or the two matrices are the same and the result is not the
    // identity to 1e-6 (the reprojection is then not to be trusted: cull nothing).
    bool Reprojection(const Mat4d& current,const Mat4d& captured,Mat4& out);
    // worldToClip for points given relative to `origin` (point minus origin), in double and rounded once: its translation column
    // takes origin's part, so for the eye as origin it is rotation and projection only, and float clip coordinates computed from
    // eye-relative points keep their precision far from the world origin (the box test, Occlusion::Boxes).
    Mat4 Relative(const Mat4d& worldToClip,const std::array<float,3>& origin);

    // Conservative NDC rectangle {x0, y0, x1, y1} (y up) covering a sphere seen through worldToClip. False when the
    // sphere reaches within one unit of the eye or lies behind it; the rectangle may lie partly or wholly off screen.
    bool ScreenRect(const Mat4& worldToClip,const float center[3],float radius,float rect[4]);
    // The largest view depth (clip w) over a sphere: no point of it lies farther in front of the eye.
    float FarthestW(const Mat4& worldToClip,const float center[3],float radius);

    // NDC depth as a function of view depth w: z = a + b / w (standard perspective, fitted once).
    struct DepthModel {
        float a=1.f,b=0.f;
        float Ndc(float w) const { return a+b/w; }
        float ViewDepth(float z) const { return b/(z-a); }  // valid for z < a
    };

    // One depth grid read back from the GPU: per texel the farthest NDC depth of its block of pixels
    // (1 = far plane, sky or no data), seen through worldToClip.
    struct Grid {
        uint32_t width=0,height=0;
        std::vector<float> depth;
        Mat4 worldToClip;
        // worldToClip built in double from the capture's camera pose; without it worldToClip is widened to double.
        Mat4d exact{};
        bool hasExact=false;
    };

    class Pyramid {
    public:
        // Reprojects grid into the camera worldToClip (texels nothing lands on count as far; after a camera move each
        // texel takes the farthest of its 3x3 neighbourhood) and builds max mips. Returns the number of source texels that landed. `exact` is worldToClip built in double from the camera
        // pose, when known (the reprojection is composed from it and grid.exact; see Mat4d).
        uint32_t Build(const Grid& grid,const Mat4& worldToClip,const DepthModel& depth,const std::array<float,3>& eye,float nearExempt,
                       const Mat4d* exact=nullptr);
        // The same, with the reprojection split in two row halves: both(first, second) runs both tasks, possibly at the same
        // time on two threads, and returns when both are done. The halves scatter into separate buffers merged by the
        // farthest depth, so the result equals the single-thread build bit for bit.
        using Both=std::function<void(const std::function<void()>&,const std::function<void()>&)>;
        uint32_t Build(const Grid& grid,const Mat4& worldToClip,const DepthModel& depth,const std::array<float,3>& eye,float nearExempt,
                       const Both& both,const Mat4d* exact=nullptr);
        // True only when every texel covering the sphere's screen rectangle is nearer than the sphere's
        // nearest point. Spheres near the eye, crossing the near plane or off screen are never occluded.
        bool Occluded(const float center[3],float radius) const;
        // The original formulation (the screen rectangle from all eight quotients), kept for the tests that check
        // Occluded returns the same answer.
        bool OccludedReference(const float center[3],float radius) const;
        // Occluded's decision for one sphere with what it rests on (developer probe, Occlusion::Probe): the level-0 texel
        // rectangle, the level read, the farthest depth found there and its view depth, and the sphere's nearest view
        // depth. `why` names the step that decided.
        struct Explanation {
            bool hidden=false;
            const char* why="";
            uint32_t tx0=0, ty0=0, tx1=0, ty1=0, level=0;
            float wmin=0.f, occluder=0.f, occluderView=0.f, distance=0.f;
        };
        Explanation Explain(const float center[3],float radius) const;
        // The farthest NDC depth at level-0 texel (x, y), or 1 outside the grid.
        float At(uint32_t x,uint32_t y) const { return ready_ && x<widths_[0] && y<heights_[0] ? levels_[0][size_t(y)*widths_[0]+x] : 1.f; }
        float ViewDepth(float z) const { return depth_.ViewDepth(z); }
        bool Ready() const { return ready_; }
        uint32_t Width() const { return ready_ ? widths_[0] : 0; }
        uint32_t Height() const { return ready_ ? heights_[0] : 0; }
        const std::vector<float>& Level(size_t i) const { return levels_[i]; }
    private:
        float MaxOver(size_t level,uint32_t x0,uint32_t y0,uint32_t x1,uint32_t y1) const;
        Mat4 clip_;
        float rowNorm_[4]{};
        DepthModel depth_;
        std::array<float,3> eye_{};
        float nearExempt_=64.f;
        // Storage is kept between builds: a pyramid is rebuilt every frame.
        std::vector<std::vector<float>> levels_;
        std::vector<float> scratch_;  // the second half's scatter target, then the 3x3 maximum's row pass
        std::vector<float> padded_;   // a moved view's landing grid, one texel wider on every side
        std::vector<uint32_t> widths_,heights_;
        bool ready_=false;
    };

    // Hi-Z only pays when it hides enough (Occlusion's "hiz_min_yield"): after `lowFrames` frames that hid less than
    // `minYield` of what they tested (with at least `minTests` tests), only `burst` frames of every `interval` test; a
    // testing frame that hides enough brings every frame back. At one Whiterun view: 4 of 2,424 objects
    // hidden per frame, 3.7% slower than precombines alone.
    struct YieldGate {
        float minYield=0.05f;
        uint32_t interval=32, burst=4, lowFrames=120, minTests=64;
        uint32_t low=0, frame=0;
        bool probing=false;
        // Once per frame with the previous frame's counts; true when this frame tests.
        bool Step(uint32_t tested,uint32_t culled);
    };

    // Objects found visible are tested again only after 1 to `interval` frames, a fixed phase per object, so the
    // re-tests spread over frames (about a third of the tests remain at interval 8). Drawing a visible object is
    // always correct, so a skipped test can only cost culling for those frames, never hide anything; a key collision
    // or two culling threads writing one slot likewise only skip or repeat a test. One 64-bit word per slot (key,
    // frame until which the object is skipped), relaxed atomics; 2^14 slots (128 KiB) stay in the L2 cache.
    class VisibleCache {
    public:
        explicit VisibleCache(uint32_t slotsLog2=14);
        bool Skip(const void* object,uint32_t frame) const;
        void Visible(const void* object,uint32_t frame,uint32_t interval);
    private:
        static uint64_t Hash(const void* object);
        std::unique_ptr<std::atomic<uint64_t>[]> slots_;
        uint32_t shift_;
    };

    // The frame each object was last tested and found visible (the developer flicker log, Occlusion.cpp NoteFlicker), lock free
    // like VisibleCache: one 64-bit word per slot (key, frame), relaxed atomics. Lossy: an object whose slot another takes loses its
    // frame, so its flicker is under-counted, never invented (a false match needs 32 equal hash bits).
    class ShownFrames {
    public:
        explicit ShownFrames(uint32_t slotsLog2=14);
        void Shown(const void* object,uint32_t frame);
        // The frame recorded for object, or 0 when none is (never found visible, or its slot taken by another object since).
        uint32_t Last(const void* object) const;
    private:
        std::unique_ptr<std::atomic<uint64_t>[]> slots_;
        uint32_t shift_;
    };
}
