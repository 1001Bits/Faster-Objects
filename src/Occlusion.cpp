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

#include "Occlusion.h"
#include "Config.h"
#include "GpuTimer.h"
#include "HiZ.h"
#include "VRDepthCulling.h"
#include "RE/S/SendHUDMessage.h"

#include <d3d11.h>
#include <d3dcompiler.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <immintrin.h>
#include <intrin.h>
#include <set>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <functional>
#include <memory>
#include <cctype>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

namespace Occlusion {
    namespace {
        using Clock=std::chrono::steady_clock;
        constexpr auto relaxed=std::memory_order_relaxed;
        constexpr uint32_t kSlots=3, kMaxAge=6, kAbStates=4;

        // Immutable snapshot of previs-hidden chunk roots; readers on culling threads never free one.
        struct PrevisSet {
            std::vector<RE::NiPointer<RE::NiAVObject>> held;
            std::vector<const RE::NiAVObject*> sorted;
        };

        struct ReferenceSet {
            std::vector<uint32_t> sorted;
            std::array<int64_t,3> cell{};
            float size=0.f;
        };

        struct Slot {
            uint64_t capture=0;
            HiZ::Mat4 clip;
            std::array<float,3> eye{};
            bool pending=false;
            HiZ::Mat4 eyeClip[2];  // VR: each eye's world-to-clip (clip is unused there)
            HiZ::Mat4d exact{};    // clip built in double from the camera pose (SE and AE; see HiZ::Mat4d)
        };

        // One frame's pyramids: one, or one per eye on VR. An object is hidden only when every eye's pyramid proves it
        // (it may be in view of one eye only, or seen past an occluder's edge by one).
        struct Views {
            HiZ::Pyramid eye[2];
            uint32_t count=1;
            bool Ready() const { for(uint32_t i=0;i<count;++i) if(!eye[i].Ready()) return false; return true; }
            bool Occluded(const float center[3],float radius) const {
                for(uint32_t i=0;i<count;++i) if(!eye[i].Occluded(center,radius)) return false;
                return true;
            }
            bool OccludedReference(const float center[3],float radius) const {
                for(uint32_t i=0;i<count;++i) if(!eye[i].OccludedReference(center,radius)) return false;
                return true;
            }
        };

        // Capture problems, each logged once.
        enum Problem : uint32_t {
            kNoProjection=1u<<0, kNoTexture=1u<<1, kMultisampled=1u<<2, kTextureArray=1u<<3, kNotReadable=1u<<4,
            kFormat=1u<<5, kViewFailed=1u<<6, kResolution=1u<<7, kClamped=1u<<8, kLetterbox=1u<<9, kViewReplaced=1u<<10,
            kNoEngineView=1u<<11, kStereoArea=1u<<12,
        };

        // How the prepass writes depth, read from the renderer's own projection (CrossCheck); nothing is captured until
        // it is known. Bottled Shaders' Reverse-Z (on by default) publishes a reversed projection and writes 1 - z into
        // a float depth copy: read as standard depth, every occluder beyond ~20 units counted as far (no culling) and a
        // near plane above half the near exemption could hide visible objects.
        enum class Convention : uint8_t { Unknown, Standard, Reversed, Mismatch };

        // Statistics counted per culling thread, each thread on its own cache line, and summed once per frame by Prepare:
        // shared counters cost about 45 ns per increment with four culling threads at once, against 4 ns alone (HiZTest).
        enum Count : uint32_t { kFadeTested,kFadeCulled,kGeometryTested,kGeometryCulled,kPrevisCulled,kReferencesCulled,kUntested,kSkipped,kFadeKept,
                                kBoxRegistered,kBoxCulled,kBoxNoModel,kBoxEdge,kBoxType,kBoxWaiting,kBoxInvalid,kCounts };
        struct alignas(64) Tally { std::array<std::atomic<uint32_t>,kCounts> count{}; };
        constexpr uint32_t kTallies=64;

        struct State {
            Settings settings;
            std::atomic_bool enabled{false};
            uint32_t gridW=256,gridH=128;
            bool initTried=false,initOk=false;
            bool tracedCapture=false,tracedReadback=false,tracedBuild=false;  // one-time progress lines
            Convention convention=Convention::Unknown;  // render thread (Prepare sets it, Capture and Projection read it)
            uint32_t conventionTries=0;                 // frames CrossCheck could not compare (another camera in ViewData)
            uint32_t conventionWaits=0;                 // frames CrossCheck skipped (ViewData not this camera, or moving); logged once
            std::atomic_bool yielded{false};  // Hi-Z paused for another plugin's occlusion culling (CheckOtherCullers)
            // The value Configure, the in-game menu and the Hi-Z shortcut set (SetEnabled); `enabled` follows it unless
            // yielded. The switch mutex orders SetEnabled (any thread) against CheckOtherCullers' switches.
            std::atomic_bool configured{false};
            std::mutex switchMutex;
            // The frame-time and yield gates, live (Configure, SetMinFrameMs, SetMinYield); Paced and YieldGate read them
            // every frame. `settings` keeps the startup values.
            std::atomic<float> minFrameMs{5.f},minYield{0.01f};
            uint32_t reported=0;  // Problem bits already logged
            ID3D11Device* device{};
            ID3D11DeviceContext* context{};
            ID3D11ComputeShader* shader{};
            ID3D11Buffer* params{};
            ID3D11Texture2D* gridTexture{};
            ID3D11UnorderedAccessView* gridView{};
            ID3D11Texture2D* staging[kSlots]{};
            Slot slots[kSlots];
            uint64_t captures=0;
            // The depth copy is read through this plugin's own view of the copy texture, never through the engine's
            // depthSRV: CS Terrain Blending points that view at its own blended texture, which it fills only after
            // RenderDepth returns. The texture is held (one reference) while it is the current one.
            ID3D11Texture2D* depthTexture{};
            ID3D11ShaderResourceView* depthView{};  // null while depthTexture is unusable
            D3D11_TEXTURE2D_DESC depthDesc{};
            const void* engineView{};  // the engine's depthSRV at the last capture (not held)
            uint32_t depthViews=0;     // views created (one per depth copy texture object)
            uint32_t areaWidth=0,areaHeight=0,areaLogs=0;
            // Where this plugin's hooks were written and what they point to, re-read once at the first capture to log
            // plugins that hooked the same places later (they wrap this plugin's hooks).
            struct Written { const char* name=nullptr; std::uintptr_t at=0; bool call=false; std::uintptr_t ours=0; };
            std::vector<Written> written;
            bool checkedLater=false;
            // From the renderer's own camera data at the latest capture.
            bool calibrated=false,rowVector=false,reversed=false,loggedProjection=false;
            HiZ::DepthModel depth;
            HiZ::Mat4 cameraToClip;  // world-to-clip times the camera's local-to-world: the projection in camera space
            RE::NiFrustum frustum{};
            // Skyrim VR: the prepass renders both eyes side by side into one double-wide depth copy. There `eyes` is 2:
            // the grid texture's left and right halves hold the left and right eye, each with its own projection
            // (StereoProjection), kept in the world camera's local space for Prepare to reproject.
            uint32_t eyes=1;
            HiZ::Mat4 eyeCameraToClip[2];
            HiZ::DepthModel eyeDepth[2];
            // Latest grid on the CPU (one per eye), rebuilt into a pyramid for every frame's camera.
            HiZ::Grid grids[2];
            uint64_t gridCapture=0;
            std::array<float,3> gridEye{};
            bool haveGrid=false;
            Views views[2];
            int next=0;
            std::atomic<const Views*> active{nullptr};
            std::atomic<const RE::NiCamera*> camera{nullptr};
            std::array<Tally,kTallies> tallies{};
            std::atomic<uint32_t> tallyThreads{0};
            HiZ::VisibleCache visible;
            std::atomic<uint32_t> cullFrame{0};   // this frame's number for VisibleCache (Prepare)
            std::atomic<uint32_t> lastSkipped{0};
            // "hiz_ab" (Prepare, render thread): this frame's state, the block and round in progress, and the finished
            // blocks for TakeAbBlocks. The "before" state counts into shared counters, as the code did then.
            std::atomic<uint8_t> abState{uint8_t(AbState::New)};
            uint32_t abInBlock=~0u, abSlot=kAbStates, abOrder[kAbStates]{0,1,2,3}, abSeed=0x9e3779b9u;
            uint64_t abRound=0;
            Clock::time_point abLast{};
            AbBlock abCurrent;
            std::mutex abMutex;
            std::vector<AbBlock> abDone;  // guarded by abMutex
            // "hiz_dump_seconds" (Settings::dumpSeconds): Prepare starts a recording frame, the culling hooks record its
            // candidates (each its own slot), Capture copies that frame's depth and ends it, a later Prepare reads the copy
            // back and a writer thread saves the file. A geometry's or a root's parent root is its nearest fade-node
            // ancestor, so the replay knows which draws a hidden root takes with it.
            struct DumpEntry { float center[3]{},radius=0.f; uint32_t root=0,reserved=0; uint64_t self=0,parentRoot=0; };
            static constexpr uint32_t kMaxDumpEntries=65536;
            std::atomic_bool dumpRecording{false};
            std::atomic<uint32_t> dumpCount{0};
            std::vector<DumpEntry> dumpEntries;
            uint32_t dumps=0, dumpRecorded=0;
            Clock::time_point lastDump{};
            bool dumpPendingCapture=false, dumpPendingRead=false;
            ID3D11Texture2D* dumpStaging{};
            D3D11_TEXTURE2D_DESC dumpDesc{};
            Reduce::Params dumpParams{};
            Reduce::Area dumpArea;
            HiZ::Mat4 dumpClip;
            HiZ::DepthModel dumpDepth;
            std::array<float,3> dumpEye{};
            float dumpNear=0.f,dumpFar=0.f,dumpEyeExempt=0.f;
            std::vector<std::array<float,4>> dumpMoving;  // the frame's moving references (center, radius), as Capture excludes them
            std::array<std::atomic<uint32_t>,kCounts> sharedCounts{};
            // "frame_log": the interval between consecutive main-view frames (Prepare runs once per frame).
            std::mutex frameMutex;
            std::vector<float> frameTimes;  // guarded by frameMutex
            Clock::time_point lastFrame{};  // render thread only
            std::atomic<uint32_t> lastFadeTested{0},lastFadeCulled{0},lastGeometryTested{0},lastGeometryCulled{0};
            std::atomic<uint32_t> prepareMicros{0},buildMicros{0},gridAge{0},landed{0},depthWidth{0},depthHeight{0};
            std::atomic<uint64_t> builds{0},publishedCaptures{0},lateBuilds{0},skippedFrames{0},skippedCaptures{0};
            // Moving references (actors, doors) the main view's culling visited this frame (world bound center and radius),
            // recorded by OnVisible from the culling jobs and excluded from this frame's capture: an actor or an opening door in the
            // depth copy would otherwise keep hiding what stands behind it for the frames the grid is reused.
            static constexpr uint32_t kMaxMoving=Reduce::kMaxExcluded;
            std::array<std::array<float,4>,kMaxMoving> moving{};
            std::atomic<uint32_t> movingCount{0},movingExcluded{0};
            std::atomic_bool movingAtEye{false};  // a moving reference other than the player reached the eye
            std::atomic<float> eyeExempt{0.f};    // farthest distance from the eye of those references' bounds this frame
            std::atomic<uint64_t> eyeExempted{0}; // captures whose near exemption was widened for them
            std::atomic<uint64_t> movingSkipped{0};
            std::array<uint32_t,kMaxMoving> movingForm{};  // form ID of each recorded moving reference
            std::atomic<uint64_t> eyeAtEye{0},eyeRect{0};    // widened captures: a reference at the eye; bounds reaching the camera plane
            std::atomic<uint64_t> boundsReplaced{0};         // actor root bounds replaced by the actor's box (see NoteMovingBody)
            std::atomic<float> eyeWidest{0.f};               // widest exemption since the last read
            std::mutex eyeLogMutex;
            std::set<uint32_t> eyeLogged;                    // references already named in the log (at most kEyeLogLimit)
            static constexpr size_t kEyeLogLimit=24;
            // Cost diagnostics: every kSampleEvery-th frame the culling hooks are timed with the TSC (the cost of reading
            // it subtracted), and the time from the start of the pyramid build to its publication and to the frame's first
            // main-camera test is taken. Tests made while the frame's pyramid was still being built are counted every frame.
            static constexpr uint64_t kSampleEvery=8;
            std::atomic_bool sampling{false};
            std::atomic<uint64_t> movingTicks{0},previsTicks{0},hizCalls{0},movingCalls{0},previsCalls{0},sampledFrames{0};
            // The Hi-Z test in stages: camera, waiting for the pyramid, the visible-object table, eligibility and the occlusion
            // test; and the main-camera calls and those that passed the table.
            std::array<std::atomic<uint64_t>,4> stageTicks{};
            std::atomic<uint64_t> candidateCalls{0},testCalls{0};
            std::atomic<uint64_t> buildStartTsc{0},publishTsc{0},firstTestTsc{0};
            std::atomic<uint64_t> raceFrames{0},raceLost{0},publishTicks{0},firstTestTicks{0};
            // TakeCaptureCost (hiz_diagnostics): Capture and Prepare in TSC ticks, the helper's builds in microseconds,
            // and the reduction's GPU time.
            std::atomic<uint64_t> captureTicks{0},prepareTicks{0},captureCalls{0},prepareCalls{0},buildMicrosSum{0},buildCalls{0};
            GpuTimer reduceTimer;  // render thread
            std::atomic<uint32_t> lastUntested{0};
            std::atomic<uint32_t> lastFadeKept{0};  // last frame: fade nodes not skipped whole to keep their fade (FadeSafe)
            // Developer probe (Probe, "hiz_probe_key"): the frame after the key names every geometry whose screen rectangle
            // covers the screen centre, with Hi-Z's numbers for it; frames without a pyramid pass it on (at most kProbeTries).
            static constexpr uint32_t kProbeLines=48, kProbeTries=600;
            // The false-cull self-check (hiz_audit, Occlusion::Audit): the culls of the current audit frame (culling threads
            // append them, the render thread draws them after RenderDepth) and a readback ring of their visible-pixel counts.
            struct AuditEntry { float center[3]{}; float radius=0.f; uint32_t form=0; uint8_t culler=0; char name[47]{}; };
            struct AuditSlot { ID3D11Buffer* staging=nullptr; std::vector<AuditEntry> entries; std::array<float,3> eye{}; bool pending=false; };
            std::unique_ptr<AuditEntry[]> auditEntries;
            std::atomic_bool auditing{false};
            std::atomic<uint32_t> auditCount{0};
            std::array<float,3> auditEye{};
            AuditSlot auditSlots[kSlots];
            uint32_t auditNext=0;
            bool auditInitTried=false, auditInitOk=false;
            ID3D11VertexShader* auditVertex{}; ID3D11PixelShader* auditPixel{}; ID3D11Buffer* auditParams{}; ID3D11Buffer* auditSpheres{};
            ID3D11ShaderResourceView* auditSphereView{}; ID3D11Buffer* auditCounts{}; ID3D11UnorderedAccessView* auditCountView{};
            ID3D11RasterizerState* auditRaster{};
            uint64_t auditFrames=0, auditChecked=0, auditFalse=0;
            std::array<uint64_t,Audit::kCullers> auditCulled{}, auditFalseBy{};
            // False culls by visible pixels (2-3, 4-15, 16-63, 64 and more), and the largest since the last summary line: a
            // sliver of a few pixels can be the depth's TAA jitter against the unjittered projection, a large one cannot.
            std::array<uint64_t,4> auditBuckets{};
            struct AuditWorst { uint32_t pixels=0; float distance=0.f; AuditEntry entry; };
            std::vector<AuditWorst> auditWorst;
            Clock::time_point auditLogged{};
            std::atomic_bool probeArmed{false};
            std::atomic<uint32_t> probeFrame{0},probeLines{0};
            uint32_t probeTries=0;  // render thread
            uint64_t tscStart=0,tscCost=0;
            Clock::time_point clockStart{};
            bool installed=false;
            // The pyramid is built on a helper thread (building it on the render thread before the culling
            // jobs cost about 0.3 ms of main-thread CPU per frame, more than the culled draws saved in Ustengrav02).
            // The render thread only maps the readback and hands over a job; culling starts when the helper publishes
            // a pyramid for the same frame, and a pyramid finished after its frame is discarded.
            std::mutex jobMutex;
            std::condition_variable jobReady;
            struct Job { HiZ::Mat4 clip[2]; std::array<float,3> eye{}; uint64_t frame=0; uint64_t age=0; Views* target=nullptr;
                         HiZ::DepthModel depth[2]; float nearExempt=64.f; uint32_t count=1; HiZ::Mat4d exact[2]{}; };
            Job job;
            bool jobPending=false;
            std::atomic_bool busy{false};
            std::atomic<int64_t> waitDeadline{0};
            Clock::time_point lastPrepare{};  // render thread only
            double frameMs=0.0;
            bool paced=true;
            std::atomic<uint64_t> fastFrames{0};  // frames Hi-Z sat out because the frame rate was already high
            HiZ::YieldGate gate;                    // render thread only (YieldGate)
            std::atomic<uint64_t> yieldSkipped{0};  // frames Hi-Z sat out between probe bursts
            // Second builder thread for the pyramid's lower half (hiz_split_build): one build took 300-400 us on one
            // thread, longer than the engine's culling at high frame rates (most waits timed out).
            std::mutex splitMutex;
            std::condition_variable splitReady,splitDone;
            const std::function<void()>* splitTask=nullptr;
            bool splitFinished=true,splitStarted=false;  // steady-clock ticks until which culling tests wait for this frame's pyramid
            std::atomic<uint64_t> waits{0},waitTimeouts{0};
            uint64_t frame=0;  // guarded by jobMutex
            bool helperStarted=false;
            std::atomic<const PrevisSet*> previs{nullptr};
            std::vector<std::shared_ptr<const PrevisSet>> previsRetained;  // the writer keeps recent snapshots alive
            std::mutex previsMutex;
            std::atomic<uint32_t> lastPrevisCulled{0};
            // Reference previs: the hidden form IDs and the view cell they hold for (see SetReferencePrevisHidden).
            std::atomic<const ReferenceSet*> references{nullptr};
            std::vector<std::shared_ptr<const ReferenceSet>> referencesRetained;
            std::atomic_bool referencesUsable{false};  // this frame's camera is within one view cell of the set's
            std::atomic<uint32_t> lastReferencesCulled{0};
            // ---- The GPU box test ("box_occlusion", Occlusion::Boxes) ----
            // A registered geometry, and the world bound it had then: a result applies only while that bound is unchanged. `epoch`
            // is the frame's boxEpoch, written last (RegisterBox): the snapshot (BoxCaptureBody) takes only slots written this frame.
            struct BoxKey { const RE::NiAVObject* object=nullptr; float center[3]{},radius=0.f; uint32_t epoch=0; };
            // This frame's registrations: the culling threads take slots in batches (BoxBatch), Capture draws them.
            std::vector<Boxes::Box> boxes;
            std::vector<BoxKey> boxKeys;
            std::atomic<uint32_t> boxCount{0};
            std::atomic<uint32_t> boxEpoch{0};  // advanced by Prepare: batches taken in an earlier frame are dropped
            // The culling camera of this frame (Prepare): boxes reaching the near plane or the screen edge are not tested. boxClip
            // takes points relative to boxEye (HiZ::Relative), as the boxes are registered.
            HiZ::Mat4 boxClip;
            std::array<float,3> boxEye{};
            float boxNear=0.f;
            float boxTexel=0.f;  // Boxes::TexelSpan for this frame's camera and depth target
            std::atomic_bool boxRegistering{false};
            // Readbacks in flight: the keys drawn, the frame and the camera they were drawn for, and the eye shift their depth
            // target was eroded for (Boxes::kErodeShader), which is all the result may be used with.
            struct BoxSlot { ID3D11Buffer* staging=nullptr; std::vector<BoxKey> keys; uint64_t frame=0; std::array<float,3> eye{};
                             RE::NiFrustum frustum{}; float shift=0.f; bool pending=false; };
            BoxSlot boxSlots[kSlots];
            uint64_t boxFrame=0;  // render thread: main-view frames (Prepare)
            // The hidden geometries of the newest result, by object (open addressing; a null object marks an empty entry).
            struct BoxTable { std::vector<BoxKey> entries; uint32_t mask=0, hidden=0; uint64_t frame=0; std::array<float,3> eye{};
                              RE::NiFrustum frustum{}; float shift=0.f; bool valid=false; };
            BoxTable boxTables[2];
            int boxCurrent=-1;
            std::atomic<const BoxTable*> boxActive{nullptr};
            bool boxInitTried=false, boxInitOk=false;
            ID3D11VertexShader* fillVertex{}; ID3D11PixelShader* fillPixel{};
            ID3D11VertexShader* boxVertex{}; ID3D11PixelShader* boxPixel{};
            ID3D11Buffer* fillParams{}; ID3D11Buffer* boxParams{};
            ID3D11Buffer* boxBuffer{}; ID3D11ShaderResourceView* boxView{};
            ID3D11Buffer* boxResults{}; ID3D11UnorderedAccessView* boxResultsView{};
            ID3D11DepthStencilState* fillState{}; ID3D11DepthStencilState* boxState{};
            ID3D11RasterizerState* boxRaster{};
            ID3D11Texture2D* boxDepth{}; ID3D11DepthStencilView* boxDepthView{};
            uint32_t boxDepthWidth=0, boxDepthHeight=0;
            // Motion erosion (Boxes::kErodeShader): the depth target as the edge pass reads it, the cliff map with its mips (a view
            // of each level to build them, one of the whole chain to search them) and the exposed level-1 texels.
            ID3D11ShaderResourceView* boxDepthRead{};
            ID3D11ComputeShader* erodeEdges{}; ID3D11ComputeShader* erodeDown{}; ID3D11ComputeShader* erodeExpose{}; ID3D11PixelShader* erodeApply{};
            ID3D11Buffer* erodeParams{};
            ID3D11Texture2D* cliffs{}; ID3D11ShaderResourceView* cliffChain{};
            std::vector<ID3D11ShaderResourceView*> cliffRead; std::vector<ID3D11UnorderedAccessView*> cliffWrite;
            ID3D11Texture2D* exposed{}; ID3D11ShaderResourceView* exposedRead{}; ID3D11UnorderedAccessView* exposedWrite{};
            std::atomic<int64_t> boxVertexBudget{0};  // mesh vertices the culling threads may still read this frame
            std::atomic<uint32_t> lastBoxRegistered{0},lastBoxCulled{0},lastBoxNoModel{0},lastBoxEdge{0},boxTableHidden{0};
            std::atomic<uint32_t> lastBoxType{0},lastBoxWaiting{0},lastBoxInvalid{0};
            std::atomic<uint64_t> boxPasses{0},boxReadbacks{0},boxShifted{0},boxStale{0};
            GpuTimer boxTimer;  // render thread
            // "hiz_diagnostics": the box pass's render-thread time, logged with its GPU time every kBoxLogFrames frames.
            double boxCpuMicros=0.0; uint32_t boxCpuPasses=0; uint64_t boxLogFrame=0;
        };
        // Never destroyed, like the precombine Manager: the retained previs snapshots hold NiPointers to chunk roots,
        // and releasing them from the C runtime's exit handlers called engine code after the engine had shut down
        // (test runs that ended with previs on hit an access violation at SkyrimSE.exe+0xC67886 under this DLL's
        // static destructors, and the game lingered past its shutdown).
        State& S() { static auto* state=new State; return *state; }
        Tally& OwnTally() { thread_local Tally* own=&S().tallies[S().tallyThreads.fetch_add(1,relaxed)%kTallies]; return *own; }
        // Only this thread writes its tally; a Take racing with it can carry one count into the next frame.
        void Bump(Count c) { auto& v=OwnTally().count[c]; v.store(v.load(relaxed)+1,relaxed); }
        uint32_t Take(Count c) {
            auto& s=S();
            const uint32_t used=std::min(s.tallyThreads.load(relaxed),kTallies);
            uint32_t sum=0;
            for(uint32_t i=0;i<used;++i) sum+=s.tallies[i].count[c].exchange(0,relaxed);
            return sum+s.sharedCounts[c].exchange(0,relaxed);
        }
        void NameEyeReference(uint32_t formID,const char* why,float radius,float distance,float reach);
        void NoteMovingBody(const RE::NiAVObject& node,const RE::NiCullingProcess& process);

        // True the first time a problem is seen (log it then).
        bool Report(Problem problem) {
            auto& s=S();
            if(s.reported&problem) return false;
            s.reported|=problem;
            return true;
        }

        HiZ::Mat4 FromMatrix(const DirectX::SimpleMath::Matrix& matrix) {
            HiZ::Mat4 out;
            for(int r=0;r<4;++r) for(int c=0;c<4;++c) out.m[r][c]=matrix.m[r][c];
            return out;
        }
        HiZ::Mat4 Translation(float x,float y,float z) {
            auto out=HiZ::Mat4::Identity();
            out.m[0][3]=x; out.m[1][3]=y; out.m[2][3]=z;
            return out;
        }
        void Apply(const HiZ::Mat4& m,const std::array<float,3>& p,float out[4]) {
            for(int r=0;r<4;++r) out[r]=m.m[r][0]*p[0]+m.m[r][1]*p[1]+m.m[r][2]*p[2]+m.m[r][3];
        }
        HiZ::Mat4 CameraWorld(const RE::NiCamera* camera) {
            float rotation[3][3],translation[3]={camera->world.translate.x,camera->world.translate.y,camera->world.translate.z};
            for(int i=0;i<3;++i) for(int j=0;j<3;++j) rotation[i][j]=camera->world.rotate.entry[i][j];
            return HiZ::FromTransform(rotation,camera->world.scale,translation);
        }
        // The same in double from the camera's float pose (exact: a float widens to double without rounding).
        HiZ::Mat4d CameraWorldExact(const RE::NiCamera* camera) {
            HiZ::Mat4d out;
            const auto& world=camera->world;
            const double t[3]={world.translate.x,world.translate.y,world.translate.z};
            for(int i=0;i<3;++i) {
                for(int j=0;j<3;++j) out.m[i][j]=double(world.rotate.entry[i][j])*double(world.scale);
                out.m[i][3]=t[i];
            }
            out.m[3][3]=1.0;
            return out;
        }
        bool SameFrustum(const RE::NiFrustum& a,const RE::NiFrustum& b) {
            const auto close=[](float x,float y) { return std::abs(x-y)<=1e-4f*std::max(1.f,std::abs(x)); };
            return close(a.fLeft,b.fLeft) && close(a.fRight,b.fRight) && close(a.fTop,b.fTop) && close(a.fBottom,b.fBottom) &&
                close(a.fNear,b.fNear) && close(a.fFar,b.fFar) && a.bOrtho==b.bOrtho;
        }

        // The main camera's world-to-clip, built from the NiCamera itself: local X forward, Y up, Z right (SE 1.5.97,
        // read from a running game), and a standard D3D depth range from the view frustum. The renderer's own
        // ViewData cannot be used at capture time: right after Main::RenderDepth it holds another camera's data
        // (calibrating against it there failed), while between frames it equals this matrix to 5e-7 in NDC.
        // NDC does not depend on the viewport, so dynamic resolution
        // (a smaller viewport into full-size targets) leaves this matrix unchanged; Capture maps the grid to the viewport.
        // Camera-space projection and depth model of one frustum; false for an orthographic or degenerate frustum.
        bool CameraToClip(const RE::NiFrustum& f,HiZ::Mat4d& cameraToClip) {
            if(f.bOrtho || !(f.fRight>f.fLeft) || !(f.fTop>f.fBottom) || !(f.fNear>0.f) || !(f.fFar>f.fNear)) return false;
            const double l=f.fLeft,r=f.fRight,t=f.fTop,b=f.fBottom,n=f.fNear,fa=f.fFar;
            cameraToClip=HiZ::Mat4d{};
            cameraToClip.m[0][0]=-(r+l)/(r-l); cameraToClip.m[0][2]=2.0/(r-l);
            cameraToClip.m[1][0]=-(t+b)/(t-b); cameraToClip.m[1][1]=2.0/(t-b);
            cameraToClip.m[2][0]=fa/(fa-n); cameraToClip.m[2][3]=-n*fa/(fa-n);
            cameraToClip.m[3][0]=1.0;
            return true;
        }
        bool CameraToClip(const RE::NiFrustum& f,HiZ::Mat4& cameraToClip,HiZ::DepthModel& depth) {
            HiZ::Mat4d exact;
            if(!CameraToClip(f,exact)) return false;
            cameraToClip=HiZ::ToFloat(exact);
            depth.a=cameraToClip.m[2][0]; depth.b=cameraToClip.m[2][3];
            return true;
        }
        // World-to-clip of the main camera, composed in double and rounded once; `exact` receives the double matrix.
        bool Projection(const RE::NiCamera* camera,HiZ::Mat4& worldToClip,HiZ::Mat4d* exact=nullptr) {
            auto& s=S();
            const auto& f=camera->GetRuntimeData2().viewFrustum;
            HiZ::Mat4d cameraToClip,inverse;
            if(!CameraToClip(f,cameraToClip) || !HiZ::Invert(CameraWorldExact(camera),inverse)) return false;
            s.cameraToClip=HiZ::ToFloat(cameraToClip);
            s.depth=HiZ::DepthModel{s.cameraToClip.m[2][0],s.cameraToClip.m[2][3]};
            s.reversed=s.convention==Convention::Reversed;
            s.frustum=f;
            s.calibrated=true;
            const auto clip=HiZ::Multiply(cameraToClip,inverse);
            worldToClip=HiZ::ToFloat(clip);
            if(exact) *exact=clip;
            return true;
        }

        // The renderer's eye relative to posAdjust: the point its projection takes to clip x = y = w = 0. posAdjust is the
        // eye in most cells, but 0 in some interiors (Understone Keep, walk240: 8,295 units off, the eye's distance from
        // the world origin), where the matrix carries the eye's translation itself.
        bool ProjectionCentre(const HiZ::Mat4& m,float out[3]) {
            constexpr int kRows[3]={0,1,3};
            double a[3][3],b[3];
            for(int i=0;i<3;++i) {
                for(int j=0;j<3;++j) a[i][j]=m.m[kRows[i]][j];
                b[i]=-double(m.m[kRows[i]][3]);
            }
            const auto det=[](const double c[3][3]) {
                return c[0][0]*(c[1][1]*c[2][2]-c[1][2]*c[2][1])-c[0][1]*(c[1][0]*c[2][2]-c[1][2]*c[2][0])+c[0][2]*(c[1][0]*c[2][1]-c[1][1]*c[2][0]);
            };
            const double d=det(a);
            if(!(std::abs(d)>1e-9)) return false;
            for(int k=0;k<3;++k) {
                double c[3][3];
                for(int i=0;i<3;++i) for(int j=0;j<3;++j) c[i][j]=j==k ? b[i] : a[i][j];
                out[k]=float(det(c)/d);
            }
            return true;
        }

        // The renderer's own data for this camera on SE and AE: the entry of BSGraphics::State's camera cache that
        // SetCameraData copies into the current view (SE 1.5.97 FUN_140d7d7b0 searches State+0xA0, AE
        // 1.6.1170 State+0xA8, AE 1.7.104 State+0xB8, CommonLib's cameraDataCacheA in all three; entries of 0x290 bytes,
        // the camera at 0 and the jitter flag at 0x284; BuildCameraStateData writes the ViewData at 0x10 and posAdjust,
        // the camera's position, at 0x260). The current view holds whichever camera was set last: in Understone Keep a
        // view at the world origin, so Hi-Z never calibrated from it there.
        const std::uint8_t* CameraEntry(const RE::NiCamera* camera) {
            if(REL::Module::IsVR()) return nullptr;
            auto* state=RE::BSGraphics::State::GetSingleton();
            if(!state) return nullptr;
            auto& cache=state->GetRuntimeData().cameraDataCacheA;
            const auto* data=reinterpret_cast<const std::uint8_t*>(cache.data());
            constexpr size_t kStride=0x290;
            for(uint32_t i=0;data && i<cache.size() && i<64;++i)
                if(*reinterpret_cast<const RE::NiCamera* const*>(data+kStride*i)==camera) return data+kStride*i;
            return nullptr;
        }

        // Once, when the renderer's ViewData holds this camera: read the depth convention by comparing its projection with
        // this plugin's (Convention). ViewData holds the pose the renderer last drew, a frame behind the camera while the
        // player moves or turns, so both projections are compared at that pose (its projection's centre and the ViewData
        // axes): the exact pose match this once required never held during walks, and Hi-Z then never captured.
        // A pose within 512 units and about 6 degrees on every axis is this camera; a cubemap face at the eye is 90
        // degrees off. Screen x and y must then agree, or ViewData holds another camera and the next frame tries again;
        // depth agrees as z (standard) or 1 - z (reversed), or matches neither and Hi-Z stays off. A mismatch counts only
        // with the camera standing still, when the ViewData is this frame's main view for certain. Agreement until now:
        // 5e-7 to 1.8e-3 in NDC, against about 0.9 between the two conventions at these
        // distances. Called from Prepare, between frames; at capture time the ViewData holds another camera.
        void CrossCheck(const RE::NiCamera* camera,const HiZ::Mat4& cameraToClip) {
            constexpr uint32_t kWaitReport=600;
            auto& s=S();
            // This camera's cache entry, else the renderer's current view (SE and AE: ViewData at 0x380, posAdjust 0x35C).
            const auto* entry=CameraEntry(camera);
            const auto* state=reinterpret_cast<const std::uint8_t*>(RE::BSGraphics::RendererShadowState::GetSingleton());
            const auto& view=*reinterpret_cast<const RE::BSGraphics::ViewData*>(entry ? entry+0x10 : state+0x380);
            const auto& adjust=*reinterpret_cast<const RE::NiPoint3*>(entry ? entry+0x260 : state+0x35C);
            const auto& eye=camera->world.translate;
            const auto& r=camera->world.rotate.entry;
            // Everything relative to posAdjust, where the renderer's matrix is (and float keeps its precision far out).
            const auto renderer=HiZ::Transpose(FromMatrix(view.viewProjMatrixUnjittered));
            float centre[3];
            if(!ProjectionCentre(renderer,centre)) return;
            // Camera-local axes as the camera's rotation columns: forward, up, right.
            const float axes[3][3]={{view.viewForward.x,view.viewForward.y,view.viewForward.z},{view.viewUp.x,view.viewUp.y,view.viewUp.z},
                                    {view.viewRight.x,view.viewRight.y,view.viewRight.z}};
            const float dx=adjust.x+centre[0]-eye.x,dy=adjust.y+centre[1]-eye.y,dz=adjust.z+centre[2]-eye.z;
            const bool nearby=dx*dx+dy*dy+dz*dz<=512.f*512.f;
            float aligned=1.f;
            for(int c=0;c<3;++c) aligned=std::min(aligned,axes[c][0]*r[0][c]+axes[c][1]*r[1][c]+axes[c][2]*r[2][c]);
            if(!nearby || !(aligned>=0.995f)) {
                if(++s.conventionWaits==kWaitReport)
                    logger::info("Hi-Z calibration waiting: for {} frames the renderer's view was not the main camera's (last {:.0f} units and {:.4f} axis "
                                 "agreement away; {})",kWaitReport,std::sqrt(dx*dx+dy*dy+dz*dz),aligned,entry ? "camera cache" : "current view");
                return;
            }
            const bool still=std::abs(dx)<=1.f && std::abs(dy)<=1.f && std::abs(dz)<=1.f && aligned>=1.f-1e-6f;
            float rotation[3][3];
            for(int i=0;i<3;++i) for(int c=0;c<3;++c) rotation[i][c]=axes[c][i];
            HiZ::Mat4 inverse;
            if(!HiZ::Invert(HiZ::FromTransform(rotation,1.f,centre),inverse)) return;
            const auto ours=HiZ::Multiply(cameraToClip,inverse);
            float screen=0.f,standard=0.f,reversed=0.f;
            for(int i=0;i<8;++i) {
                const float d=500.f+i*1500.f,side=(i%2 ? 1.f:-1.f)*0.3f*d,up=(i%3-1)*0.2f*d;
                std::array<float,3> point;
                for(int k=0;k<3;++k) point[k]=centre[k]+axes[0][k]*d+axes[2][k]*side+axes[1][k]*up;
                float a[4],b[4];
                Apply(ours,point,a); Apply(renderer,point,b);
                for(int c=0;c<2;++c) screen=std::max(screen,std::abs(a[c]/a[3]-b[c]/b[3]));
                standard=std::max(standard,std::abs(a[2]/a[3]-b[2]/b[3]));
                reversed=std::max(reversed,std::abs(1.f-a[2]/a[3]-b[2]/b[3]));
            }
            constexpr float kAgree=0.02f;
            constexpr uint32_t kTries=600;
            const bool depthAgrees=standard<=kAgree || reversed<=kAgree;
            if(!(screen<=kAgree) || (!depthAgrees && !still)) {
                if(!still) {
                    if(++s.conventionWaits==kWaitReport)
                        logger::info("Hi-Z calibration waiting: for {} frames the renderer's projection differed by {:.2e} on screen and {:.2e} in depth "
                                     "(camera moving)",kWaitReport,screen,std::min(standard,reversed));
                    return;
                }
                if(++s.conventionTries==kTries) {
                    s.convention=Convention::Mismatch;
                    logger::warn("Hi-Z occlusion stays off: for {} frames the renderer's projection did not match the main camera's on screen "
                                 "(by {:.2e} in NDC)",kTries,screen);
                }
                return;
            }
            s.convention=standard<=kAgree ? Convention::Standard : reversed<=kAgree ? Convention::Reversed : Convention::Mismatch;
            if(s.convention==Convention::Mismatch)
                logger::warn("Hi-Z occlusion stays off: the renderer's depth is neither standard nor reversed (camera near {:.1f}, far {:.0f}; "
                             "depth differs by {:.2e} from standard and {:.2e} from reversed)",camera->GetRuntimeData2().viewFrustum.fNear,
                             camera->GetRuntimeData2().viewFrustum.fFar,standard,reversed);
            else
                logger::info("Hi-Z projection from the camera frustum (near {:.1f}, far {:.0f}); renderer ViewData agrees to {:.2e} in NDC with {} depth{} "
                             "(from the {}; renderer eye {:.1f} units from posAdjust)",
                    camera->GetRuntimeData2().viewFrustum.fNear,camera->GetRuntimeData2().viewFrustum.fFar,std::max(screen,std::min(standard,reversed)),
                    s.convention==Convention::Reversed ? "reversed":"standard",s.convention==Convention::Reversed ? " (Reverse-Z, e.g. Bottled Shaders)":"",
                    entry ? "camera cache" : "current view",std::sqrt(centre[0]*centre[0]+centre[1]*centre[1]+centre[2]*centre[2]));
        }

        // Skyrim VR: both eyes' world-to-clip, as the engine's own occlusion test projects. The renderer's camera
        // cache (FUN_140dd1ad0: State*, camera, useJitter; it only searches) holds the world root camera's entry: +0x08 a
        // BSTArray<ViewData> (stride 0x250) and +0x20 a BSTArray<NiPoint3> of eye position adjustments, two of each.
        // Eye e: Transpose(viewProjMatrixUnjittered) after subtracting its adjustment. Eye 0 must lie left of eye 1 along
        // the camera's right axis (local Z): it renders into the left half of the double-wide depth copy. Each eye's
        // depth model is fitted to its matrix (clip z = a w + b); reversed depth is turned into standard depth so the
        // pyramid sees 0 at the near plane. The projections are kept in the camera's local space (eyeCameraToClip),
        // which stays fixed while the headset moves: Prepare reprojects them into each frame's camera. Anything
        // unexpected (no entry, other sizes, eyes the wrong way round, a depth row that is no function of view depth)
        // refuses the capture; reported once.
        bool StereoProjection(const RE::NiCamera* camera,HiZ::Mat4 clips[2]) {
            auto& s=S();
            using Lookup=const std::uint8_t* (*)(RE::BSGraphics::State*,const RE::NiCamera*,bool);
            static const REL::Relocation<Lookup> lookup{REL::Offset(0xDD1AD0)};
            static uint32_t refused=0;
            const auto refuse=[&](const char* why) {
                if(!(refused&1u)) { refused|=1u; logger::warn("Hi-Z capture skipped on VR: {}",why); }
                return false;
            };
            auto* state=RE::BSGraphics::State::GetSingleton();
            const auto* entry=state ? lookup(state,camera,false) : nullptr;
            if(!entry || *reinterpret_cast<const RE::NiCamera* const*>(entry)!=camera) return refuse("the renderer holds no camera data for the world camera");
            const auto* views=*reinterpret_cast<const std::uint8_t* const*>(entry+0x08);
            const auto* adjust=*reinterpret_cast<const RE::NiPoint3* const*>(entry+0x20);
            if(!views || !adjust || *reinterpret_cast<const std::uint32_t*>(entry+0x18)!=2 || *reinterpret_cast<const std::uint32_t*>(entry+0x30)!=2)
                return refuse("the world camera's data does not hold two eyes");
            const auto& r=camera->world.rotate.entry;
            const float side=(adjust[1].x-adjust[0].x)*r[0][2]+(adjust[1].y-adjust[0].y)*r[1][2]+(adjust[1].z-adjust[0].z)*r[2][2];
            if(!(side>0.f)) return refuse("eye 0 is not the left eye");
            const auto world=CameraWorld(camera);
            HiZ::DepthModel models[2];
            HiZ::Mat4 local[2];
            for(int e=0;e<2;++e) {
                const auto& view=*reinterpret_cast<const RE::BSGraphics::ViewData*>(views+size_t(0x250)*e);
                clips[e]=HiZ::Multiply(HiZ::Transpose(FromMatrix(view.viewProjMatrixUnjittered)),Translation(-adjust[e].x,-adjust[e].y,-adjust[e].z));
                local[e]=HiZ::Multiply(clips[e],world);
                // Fitted in camera space, where the numbers stay small: z = a w + b for every point.
                const auto& m=local[e].m;
                const float ww=m[3][0]*m[3][0]+m[3][1]*m[3][1]+m[3][2]*m[3][2];
                if(!(ww>0.f) || !std::isfinite(ww)) return refuse("an eye projection has no view depth");
                const float a=(m[2][0]*m[3][0]+m[2][1]*m[3][1]+m[2][2]*m[3][2])/ww,b=m[2][3]-a*m[3][3];
                float off=0.f;
                for(int k=0;k<3;++k) off=std::max(off,std::abs(m[2][k]-a*m[3][k]));
                if(!(off<=1e-4f*std::max(1.f,std::abs(a))) || !std::isfinite(b) || b==0.f) return refuse("an eye's depth is no function of view depth");
                models[e]=HiZ::DepthModel{a,b};
            }
            // Standard depth grows with view depth (b < 0); reversed depth falls with it.
            const bool reversed=models[0].b>0.f;
            if((models[1].b>0.f)!=reversed) return refuse("the eyes' depth conventions differ");
            for(int e=0;e<2;++e) {
                if(reversed) {
                    // 1 - z: the z row becomes w - z.
                    for(int k=0;k<4;++k) { clips[e].m[2][k]=clips[e].m[3][k]-clips[e].m[2][k]; local[e].m[2][k]=local[e].m[3][k]-local[e].m[2][k]; }
                    models[e]=HiZ::DepthModel{1.f-models[e].a,-models[e].b};
                }
                s.eyeCameraToClip[e]=local[e];
                s.eyeDepth[e]=models[e];
            }
            s.depth=models[0];
            s.reversed=reversed;
            if(s.convention==Convention::Unknown) {
                s.convention=reversed ? Convention::Reversed : Convention::Standard;
                logger::info("Hi-Z on VR: both eyes from the renderer's world-camera data ({} depth, eyes {:.1f} units apart; near {:.1f}, far {:.0f})",
                    reversed ? "reversed":"standard",std::sqrt((adjust[1]-adjust[0]).SqrLength()),camera->GetRuntimeData2().viewFrustum.fNear,
                    camera->GetRuntimeData2().viewFrustum.fFar);
            }
            s.calibrated=true;
            return true;
        }

        bool InitGpu(RE::BSGraphics::Renderer* renderer) {
            auto& s=S();
            if(s.initTried) return s.initOk;
            s.initTried=true;
            const auto fail=[](const char* what,HRESULT hr=S_OK) { logger::error("Hi-Z occlusion disabled: {} (0x{:08X})",what,static_cast<uint32_t>(hr)); return false; };
            s.device=reinterpret_cast<ID3D11Device*>(renderer->GetRuntimeData().forwarder);
            s.context=reinterpret_cast<ID3D11DeviceContext*>(renderer->GetRuntimeData().context);
            if(!s.device || !s.context) return fail("no D3D11 device");
            const HMODULE compiler=LoadLibraryW(L"d3dcompiler_47.dll");
            const auto compile=compiler ? reinterpret_cast<pD3DCompile>(GetProcAddress(compiler,"D3DCompile")) : nullptr;
            if(!compile) return fail("d3dcompiler_47.dll unavailable");
            ID3DBlob* code{}; ID3DBlob* errors{};
            HRESULT hr=compile(Reduce::kShader,sizeof(Reduce::kShader)-1,"PrecombinesHiZ",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&code,&errors);
            if(FAILED(hr)) {
                if(errors) logger::error("Hi-Z shader: {}",static_cast<const char*>(errors->GetBufferPointer()));
                if(errors) errors->Release();
                return fail("shader compilation failed",hr);
            }
            hr=s.device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&s.shader);
            code->Release();
            if(FAILED(hr)) return fail("CreateComputeShader",hr);
            D3D11_BUFFER_DESC buffer{};
            buffer.ByteWidth=sizeof(Reduce::Params); buffer.Usage=D3D11_USAGE_DEFAULT; buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            if(FAILED(hr=s.device->CreateBuffer(&buffer,nullptr,&s.params))) return fail("CreateBuffer",hr);
            D3D11_TEXTURE2D_DESC texture{};
            texture.Width=s.gridW; texture.Height=s.gridH; texture.MipLevels=1; texture.ArraySize=1;
            texture.Format=DXGI_FORMAT_R32_FLOAT; texture.SampleDesc.Count=1;
            texture.Usage=D3D11_USAGE_DEFAULT; texture.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
            if(FAILED(hr=s.device->CreateTexture2D(&texture,nullptr,&s.gridTexture))) return fail("CreateTexture2D",hr);
            if(FAILED(hr=s.device->CreateUnorderedAccessView(s.gridTexture,nullptr,&s.gridView))) return fail("CreateUnorderedAccessView",hr);
            texture.Usage=D3D11_USAGE_STAGING; texture.BindFlags=0; texture.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
            for(auto& staging:s.staging) if(FAILED(hr=s.device->CreateTexture2D(&texture,nullptr,&staging))) return fail("CreateTexture2D (staging)",hr);
            s.initOk=true;
            logger::info("Hi-Z occlusion ready: {}x{} grid, {} readback slots",s.gridW,s.gridH,kSlots);
            return true;
        }

        // A frame without a capture still counts toward the newest grid's age, so a run of skipped (or disabled)
        // frames stops culling after kMaxAge instead of reusing an old grid indefinitely. The slot this capture would
        // have filled is dropped, as a real capture would overwrite it.
        void Age(bool skipped) {
            auto& s=S();
            s.slots[s.captures%kSlots].pending=false;
            ++s.captures;
            if(skipped) s.skippedCaptures.fetch_add(1,relaxed);
        }

        // Once per change of the engine's depthSRV: note when another plugin has pointed it at a different resource
        // (CS Terrain Blending swaps it for its own blended R32F texture). Hi-Z keeps reading the depth copy itself.
        void CheckEngineView(ID3D11ShaderResourceView* engineView,ID3D11Texture2D* texture) {
            if(!engineView) {
                if(Report(kNoEngineView)) logger::info("Hi-Z: the engine's post-prepass depth view is empty; reading the depth copy texture directly");
                return;
            }
            ID3D11Resource* resource{};
            engineView->GetResource(&resource);
            if(resource && resource!=texture && Report(kViewReplaced)) {
                D3D11_SHADER_RESOURCE_VIEW_DESC view{};
                engineView->GetDesc(&view);
                D3D11_TEXTURE2D_DESC desc{};
                ID3D11Texture2D* other{};
                if(SUCCEEDED(resource->QueryInterface(__uuidof(ID3D11Texture2D),reinterpret_cast<void**>(&other))) && other) { other->GetDesc(&desc); other->Release(); }
                logger::info("Hi-Z: another plugin replaced the post-prepass depth view (format {}, {}x{}); Hi-Z reads the engine's depth copy, "
                    "which RenderDepth has just written, through its own view",static_cast<uint32_t>(view.Format),desc.Width,desc.Height);
            }
            if(resource) resource->Release();
        }

        // This plugin's view of the post-prepass depth copy, or null when the copy is unusable this frame. The texture
        // is validated once per texture object: multisampled, array, unreadable or non-depth formats are refused.
        ID3D11ShaderResourceView* DepthView(const RE::BSGraphics::DepthStencilData& target) {
            auto& s=S();
            auto* texture=reinterpret_cast<ID3D11Texture2D*>(target.texture);
            auto* engineView=reinterpret_cast<ID3D11ShaderResourceView*>(target.depthSRV);
            if(engineView!=s.engineView) { s.engineView=engineView; CheckEngineView(engineView,texture); }
            if(!texture) {
                if(Report(kNoTexture)) logger::warn("Hi-Z capture skipped: the post-prepass depth copy has no texture");
                return nullptr;
            }
            if(texture==s.depthTexture) return s.depthView;
            // A new texture object (first capture, or the engine or a plugin recreated the targets).
            if(s.depthView) { s.depthView->Release(); s.depthView=nullptr; }
            if(s.depthTexture) s.depthTexture->Release();
            texture->AddRef();
            s.depthTexture=texture;
            texture->GetDesc(&s.depthDesc);
            const auto& d=s.depthDesc;
            const auto format=Reduce::DepthViewFormat(d.Format);
            const auto refuse=[&](Problem problem,const char* why) {
                if(Report(problem)) logger::warn("Hi-Z capture skipped: the post-prepass depth copy ({}x{}, format {}, {} samples, {} slices) {}",
                    d.Width,d.Height,static_cast<uint32_t>(d.Format),d.SampleDesc.Count,d.ArraySize,why);
                return nullptr;
            };
            if(d.SampleDesc.Count>1) return refuse(kMultisampled,"is multisampled");
            if(d.ArraySize!=1) return refuse(kTextureArray,"is a texture array");
            if(!(d.BindFlags&D3D11_BIND_SHADER_RESOURCE)) return refuse(kNotReadable,"cannot be read by shaders");
            if(format==DXGI_FORMAT_UNKNOWN) return refuse(kFormat,"is not a depth format");
            D3D11_SHADER_RESOURCE_VIEW_DESC view{};
            view.Format=format; view.ViewDimension=D3D11_SRV_DIMENSION_TEXTURE2D; view.Texture2D.MostDetailedMip=0; view.Texture2D.MipLevels=1;
            if(const HRESULT hr=s.device->CreateShaderResourceView(texture,&view,&s.depthView);FAILED(hr)) {
                s.depthView=nullptr;
                if(Report(kViewFailed)) logger::warn("Hi-Z capture skipped: no shader view of the {}x{} depth copy (format {}): 0x{:08X}",
                    d.Width,d.Height,static_cast<uint32_t>(d.Format),static_cast<uint32_t>(hr));
                return nullptr;
            }
            if(++s.depthViews<=4)
                logger::info("Hi-Z reads the {}x{} post-prepass depth copy (format {}) through its own view (format {}){}",d.Width,d.Height,
                    static_cast<uint32_t>(d.Format),static_cast<uint32_t>(format),s.depthViews==4 ? "; later texture changes are not logged":"");
            return s.depthView;
        }

        // The part of the depth copy that this frame's prepass rendered: the texture times the engine's
        // dynamic-resolution ratios (BSGraphics::State; written by CS upscaling and Skyrim Upscaler).
        bool CurrentArea(Reduce::Area& area) {
            auto& s=S();
            auto* state=RE::BSGraphics::State::GetSingleton();
            if(!state) return false;
            // Letterboxing sets the prepass camera's port (main draw 35560 writes it when State::letterbox is set); that
            // area is not modelled, so nothing is captured.
            // CommonLib 7+: State's frame flags moved on AE 1.7.99+ (0x61; legacy AE 0x55; SE/VR 0x51), so builds with AE
            // read them through accessors.
            if(state->GetLetterbox()) {
                if(Report(kLetterbox)) logger::warn("Hi-Z capture skipped: letterboxed rendering");
                return false;
            }
            const auto& data=state->GetRuntimeData();
            const float widthRatio=data.dynamicResolutionWidthRatio,heightRatio=data.dynamicResolutionHeightRatio;
            if(!Reduce::RenderedArea(s.depthDesc.Width,s.depthDesc.Height,widthRatio,heightRatio,area)) {
                if(Report(kResolution)) logger::warn("Hi-Z capture skipped: dynamic-resolution ratios {} x {} give no usable area of the {}x{} depth copy",
                    widthRatio,heightRatio,s.depthDesc.Width,s.depthDesc.Height);
                return false;
            }
            if(area.clamped && Report(kClamped))
                logger::warn("Hi-Z: dynamic-resolution ratios {} x {} exceed 1; the whole depth copy is used",widthRatio,heightRatio);
            if(area.width!=s.areaWidth || area.height!=s.areaHeight) {
                s.areaWidth=area.width; s.areaHeight=area.height;
                s.depthWidth.store(area.width,relaxed); s.depthHeight.store(area.height,relaxed);
                // Evidence for compatibility tests: the first areas, against the viewport the prepass left bound. Only
                // logged, never acted on (a pass without draws leaves an earlier pass's viewport bound).
                if(s.areaLogs<8) {
                    ++s.areaLogs;
                    D3D11_VIEWPORT viewport{};
                    UINT count=1;
                    s.context->RSGetViewports(&count,&viewport);
                    const bool matches=count==1 && viewport.TopLeftX==0.f && viewport.TopLeftY==0.f &&
                        std::abs(viewport.Width-float(area.width))<=1.f && std::abs(viewport.Height-float(area.height))<=1.f;
                    logger::info("Hi-Z depth area {}x{} of the {}x{} depth copy (dynamic resolution {:.4f} x {:.4f}, screen {}x{}); "
                        "viewport bound after the prepass: {}",area.width,area.height,s.depthDesc.Width,s.depthDesc.Height,widthRatio,heightRatio,
                        state->screenWidth,state->screenHeight,count ? std::format("{:.1f},{:.1f} {:.1f}x{:.1f} ({})",viewport.TopLeftX,viewport.TopLeftY,
                        viewport.Width,viewport.Height,matches ? "matches":"differs") : std::string("none"));
                }
            }
            return true;
        }

        void LogLaterHooks();

        // Main draw, right after Main::RenderDepth: the prepass depth copy and the main camera both belong to this
        // frame's main view. Only the rendered area is reduced, so the grid spans the viewport (NDC -1..1).
        // Bytes per texel of the depth formats DepthViewFormat accepts.
        uint32_t TexelBytes(DXGI_FORMAT format) {
            switch(format) {
            case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_R32_TYPELESS: return 4;
            case DXGI_FORMAT_R32G8X24_TYPELESS: return 8;
            case DXGI_FORMAT_R16_TYPELESS: return 2;
            default: return 0;
            }
        }

        // A dump frame (State::DumpEntry), right after RenderDepth: the recording ends; the whole depth copy goes into a
        // staging texture, with this frame's camera and moving references.
        void DumpCapture() {
            auto& s=S();
            s.dumpRecording.store(false,relaxed);
            s.dumpPendingCapture=false;
            s.dumpRecorded=std::min(s.dumpCount.load(relaxed),State::kMaxDumpEntries);
            auto* renderer=RE::BSGraphics::Renderer::GetSingleton();
            auto* camera=RE::Main::WorldRootCamera();
            HiZ::Mat4 clip;
            if(!renderer || !InitGpu(renderer) || !camera || !Projection(camera,clip)) return;
            Reduce::Area area;
            if(!DepthView(renderer->GetDepthStencilData().depthStencils[RE::RENDER_TARGETS_DEPTHSTENCIL::kPOST_ZPREPASS_COPY]) || !CurrentArea(area) ||
               !TexelBytes(s.depthDesc.Format)) return;
            const auto& d=s.depthDesc;
            if(!s.dumpStaging || s.dumpDesc.Width!=d.Width || s.dumpDesc.Height!=d.Height || s.dumpDesc.Format!=d.Format) {
                if(s.dumpStaging) { s.dumpStaging->Release(); s.dumpStaging=nullptr; }
                s.dumpDesc=d;
                s.dumpDesc.Usage=D3D11_USAGE_STAGING; s.dumpDesc.BindFlags=0; s.dumpDesc.CPUAccessFlags=D3D11_CPU_ACCESS_READ; s.dumpDesc.MiscFlags=0;
                if(FAILED(s.device->CreateTexture2D(&s.dumpDesc,nullptr,&s.dumpStaging))) { s.dumpStaging=nullptr; logger::warn("Hi-Z dump: no staging texture"); return; }
            }
            s.context->CopyResource(s.dumpStaging,s.depthTexture);
            s.dumpArea=area; s.dumpClip=clip; s.dumpDepth=s.depth;
            s.dumpEye={camera->world.translate.x,camera->world.translate.y,camera->world.translate.z};
            s.dumpNear=camera->GetRuntimeData2().viewFrustum.fNear; s.dumpFar=camera->GetRuntimeData2().viewFrustum.fFar;
            s.dumpEyeExempt=s.eyeExempt.load(relaxed);
            s.dumpMoving.assign(s.moving.begin(),s.moving.begin()+std::min(s.movingCount.load(relaxed),State::kMaxMoving));
            s.dumpPendingRead=true;
        }

        // From Prepare: the dump frame's depth once the GPU has copied it, written by a thread of its own. Layout (little
        // endian): "HIZD", version 1; u32 texture width, height, DXGI format, rendered width, height, bytes per texel,
        // reversed, entries, moving references; f32 world-to-clip (16, rows), depth model a and b, near exemption, eye
        // exemption, eye (3), near and far plane; moving references (f32 x 4 each); the rendered rows of the depth copy;
        // the entries (State::DumpEntry, 40 bytes each).
        void DumpRead() {
            auto& s=S();
            D3D11_MAPPED_SUBRESOURCE mapped{};
            const HRESULT hr=s.context->Map(s.dumpStaging,0,D3D11_MAP_READ,D3D11_MAP_FLAG_DO_NOT_WAIT,&mapped);
            if(hr==DXGI_ERROR_WAS_STILL_DRAWING) return;
            s.dumpPendingRead=false;
            if(FAILED(hr)) return;
            static_assert(sizeof(State::DumpEntry)==40);
            std::vector<std::uint8_t> file;
            const auto put=[&](const void* data,size_t size) { const auto* b=static_cast<const std::uint8_t*>(data); file.insert(file.end(),b,b+size); };
            const auto u32=[&](uint32_t v) { put(&v,4); };
            const auto f32=[&](float v) { put(&v,4); };
            const uint32_t bytes=TexelBytes(s.dumpDesc.Format);
            put("HIZD",4); u32(1);
            for(const uint32_t v:{s.dumpDesc.Width,s.dumpDesc.Height,static_cast<uint32_t>(s.dumpDesc.Format),s.dumpArea.width,s.dumpArea.height,bytes,
                                  s.reversed ? 1u : 0u,s.dumpRecorded,static_cast<uint32_t>(s.dumpMoving.size())}) u32(v);
            for(int r=0;r<4;++r) for(int c=0;c<4;++c) f32(s.dumpClip.m[r][c]);
            for(const float v:{s.dumpDepth.a,s.dumpDepth.b,s.settings.nearExempt,s.dumpEyeExempt,s.dumpEye[0],s.dumpEye[1],s.dumpEye[2],s.dumpNear,s.dumpFar}) f32(v);
            for(const auto& m:s.dumpMoving) put(m.data(),16);
            for(uint32_t y=0;y<s.dumpArea.height;++y) put(static_cast<const std::uint8_t*>(mapped.pData)+size_t(y)*mapped.RowPitch,size_t(s.dumpArea.width)*bytes);
            s.context->Unmap(s.dumpStaging,0);
            put(s.dumpEntries.data(),size_t(s.dumpRecorded)*sizeof(State::DumpEntry));
            const auto dir=SKSE::log::log_directory();
            if(!dir) return;
            const auto path=*dir/std::format("SkyrimPrecombines_hiz_{:03}.bin",s.dumps++);
            logger::info("Hi-Z dump {}: {}x{} of a {}x{} depth copy (format {}), {} candidates, {} moving references",path.filename().string(),
                s.dumpArea.width,s.dumpArea.height,s.dumpDesc.Width,s.dumpDesc.Height,static_cast<uint32_t>(s.dumpDesc.Format),s.dumpRecorded,s.dumpMoving.size());
            std::thread([path,data=std::move(file)] { std::ofstream out(path,std::ios::binary); out.write(reinterpret_cast<const char*>(data.data()),std::streamsize(data.size())); }).detach();
        }

        // This frame's moving references (State::moving) cut out of `params` by their screen rectangles, in each eye's NDC
        // on VR (`views`: one world-to-clip per eye). A moving reference without a screen rectangle (its bounds hold the
        // eye, as a mount's or a follower's beside the camera, or reach the camera plane) cannot be cut out by area. Every
        // depth nearer than the far side of its bounds counts as far instead: its own depths are all within that distance,
        // and the occluders beyond it still cull. (Dropping the capture for them left Hi-Z culling nothing for
        // whole sessions at Whiterun.) False when more of them are visible than the list holds: no capture from this frame
        // (the older grid keeps aging and stops culling after kMaxAge frames). `record`: count the widened exemptions and
        // the excluded references in the statistics (the Hi-Z capture; the box pass repeats the same cut-outs).
        bool ExcludeMoving(Reduce::Params& params,const RE::NiCamera* camera,const HiZ::Mat4* views,bool stereo,bool record) {
            auto& s=S();
            const uint32_t count=s.movingCount.load(relaxed);
            bool usable=count<=State::kMaxMoving, rectWidened=false;
            float exempt=std::max(s.settings.nearExempt,s.eyeExempt.load(relaxed));
            const auto& eye=camera->world.translate;
            const uint32_t eyes=stereo ? 2u : 1u;
            for(uint32_t i=0;usable && i<count;++i) {
                float rects[2][4];
                const auto& m=s.moving[i];
                bool placed=true;
                for(uint32_t e=0;e<eyes && placed;++e) placed=HiZ::ScreenRect(views[e],m.data(),m[3],rects[e]);
                if(!placed) {
                    // Wholly within the near exemption or behind the eye (a Skeleton 4,077 units behind
                    // the camera widened it to 4,269): its depths are already ignored, so nothing is widened for it.
                    float farthest=0.f;
                    for(uint32_t e=0;e<eyes;++e) farthest=std::max(farthest,HiZ::FarthestW(views[e],m.data(),m[3]));
                    if(farthest<=s.settings.nearExempt) continue;
                    const float dx=m[0]-eye.x,dy=m[1]-eye.y,dz=m[2]-eye.z;
                    const float reach=std::sqrt(dx*dx+dy*dy+dz*dz)+m[3];
                    if(reach>s.settings.nearExempt) {
                        rectWidened=true;
                        if(record) NameEyeReference(s.movingForm[i],"bounds reach the camera plane",m[3],reach-m[3],reach);
                    }
                    exempt=std::max(exempt,reach);
                    continue;
                }
                for(uint32_t e=0;e<eyes && usable;++e) usable=stereo ? Reduce::ExcludeEye(params,e,rects[e]) : Reduce::Exclude(params,rects[e]);
            }
            if(!usable) return false;
            if(exempt>s.settings.nearExempt) {
                params.options[0]=s.depth.Ndc(exempt);
                if(record) {
                    s.eyeExempted.fetch_add(1,relaxed);
                    (rectWidened ? s.eyeRect : s.eyeAtEye).fetch_add(1,relaxed);
                    for(float prior=s.eyeWidest.load(relaxed);exempt>prior && !s.eyeWidest.compare_exchange_weak(prior,exempt,relaxed););
                }
            }
            if(record) {
                const auto excluded=static_cast<uint32_t>(params.options[2]);
                s.movingExcluded.store(stereo ? excluded%256+excluded/256 : excluded,relaxed);
            }
            return true;
        }

        void CaptureBody() {
            auto& s=S();
            if(!s.checkedLater) { s.checkedLater=true; LogLaterHooks(); }
            if(s.dumpPendingCapture) DumpCapture();
            const bool gated=s.settings.ab ? s.abState.load(relaxed)==uint8_t(AbState::Off) : !s.paced;
            if(!s.enabled.load(relaxed) || gated) { if(s.initOk) Age(false); return; }
            // Only depth whose convention CrossCheck has read is captured (see Convention); on VR the eyes' own
            // projections give it (StereoProjection).
            const bool stereo=s.eyes==2;
            if(!stereo && s.convention!=Convention::Standard && s.convention!=Convention::Reversed) { if(s.initOk) Age(false); return; }
            auto* renderer=RE::BSGraphics::Renderer::GetSingleton();
            if(!renderer || !InitGpu(renderer)) return;
            auto* camera=RE::Main::WorldRootCamera();
            HiZ::Mat4 clip,eyeClips[2];
            HiZ::Mat4d exact;
            if(!camera || (stereo ? !StereoProjection(camera,eyeClips) : !Projection(camera,clip,&exact))) {
                if(!stereo && Report(kNoProjection)) logger::warn("Hi-Z capture skipped: the main camera has no perspective frustum");
                Age(true);
                return;
            }
            auto* source=DepthView(renderer->GetDepthStencilData().depthStencils[RE::RENDER_TARGETS_DEPTHSTENCIL::kPOST_ZPREPASS_COPY]);
            Reduce::Area area;
            if(!source || !CurrentArea(area)) { Age(true); return; }
            // Where dynamic resolution puts each eye inside a smaller double-wide area is not modelled.
            if(stereo && area.partial) {
                if(Report(kStereoArea)) logger::warn("Hi-Z capture skipped on VR: dynamic resolution renders {}x{} of the {}x{} depth copy",
                    area.width,area.height,s.depthDesc.Width,s.depthDesc.Height);
                Age(true);
                return;
            }
            auto params=Reduce::MakeParams(area,s.gridW,s.gridH,s.depth.Ndc(s.settings.nearExempt),s.reversed,stereo);
            // On VR each moving reference is cut out of both eyes' halves, each in its own eye's NDC.
            if(s.settings.excludeMoving && !ExcludeMoving(params,camera,stereo ? eyeClips : &clip,stereo,true)) {
                s.movingSkipped.fetch_add(1,relaxed); Age(true); return;
            }
            auto* context=s.context;
            context->UpdateSubresource(s.params,0,nullptr,&params,0,0);
            // Leave the engine's compute bindings exactly as found.
            ID3D11ComputeShader* oldShader{}; ID3D11ShaderResourceView* oldView{}; ID3D11UnorderedAccessView* oldTarget{}; ID3D11Buffer* oldParams{};
            context->CSGetShader(&oldShader,nullptr,nullptr);
            context->CSGetShaderResources(0,1,&oldView);
            context->CSGetUnorderedAccessViews(0,1,&oldTarget);
            context->CSGetConstantBuffers(0,1,&oldParams);
            const UINT keep=~0u;
            context->CSSetShader(s.shader,nullptr,0);
            context->CSSetShaderResources(0,1,&source);
            context->CSSetUnorderedAccessViews(0,1,&s.gridView,&keep);
            context->CSSetConstantBuffers(0,1,&s.params);
            if(s.settings.diagnostics) s.reduceTimer.Begin(s.device,context);
            context->Dispatch((s.gridW+7)/8,(s.gridH+7)/8,1);
            if(s.settings.diagnostics) s.reduceTimer.End(context);
            context->CSSetShader(oldShader,nullptr,0);
            context->CSSetShaderResources(0,1,&oldView);
            context->CSSetUnorderedAccessViews(0,1,&oldTarget,&keep);
            context->CSSetConstantBuffers(0,1,&oldParams);
            if(oldShader) oldShader->Release();
            if(oldView) oldView->Release();
            if(oldTarget) oldTarget->Release();
            if(oldParams) oldParams->Release();
            const uint32_t index=static_cast<uint32_t>(s.captures%kSlots);
            context->CopyResource(s.staging[index],s.gridTexture);
            Slot slot{s.captures,clip,{camera->world.translate.x,camera->world.translate.y,camera->world.translate.z},true};
            if(stereo) { slot.eyeClip[0]=eyeClips[0]; slot.eyeClip[1]=eyeClips[1]; }
            else slot.exact=exact;
            s.slots[index]=slot;
            ++s.captures;
            if(!s.tracedCapture) {
                s.tracedCapture=true;
                logger::info("Hi-Z first capture: depth {}x{} format {}, area {}x{}, grid dispatched and copied to readback slot {}",s.depthDesc.Width,
                    s.depthDesc.Height,static_cast<uint32_t>(s.depthDesc.Format),area.width,area.height,index);
            }
        }

        void Capture() {
            auto& s=S();
            if(!s.settings.diagnostics) return CaptureBody();
            const uint64_t begin=__rdtsc();
            CaptureBody();
            s.captureTicks.fetch_add(__rdtsc()-begin,relaxed); s.captureCalls.fetch_add(1,relaxed);
        }

        void Helper();

        // Main draw, right after Main::Begin and before the culling jobs: newest finished readback,
        // reprojected into this frame's camera.
        // Hi-Z pays off only while frames are slow enough: uncapped interiors at about 300 fps (3.4 ms) lost 1.5-3% with it,
        // exteriors at 5.9-9.5 ms gained 6-16%. The smoothed frame time
        // switches it on at hiz_min_frame_ms and off below 90% of that; 0 keeps it always on.
        bool Paced() {
            auto& s=S();
            const auto now=Clock::now();
            if(s.lastPrepare.time_since_epoch().count()) {
                const double ms=std::chrono::duration<double,std::milli>(now-s.lastPrepare).count();
                if(ms>0.0 && ms<250.0) s.frameMs=s.frameMs>0.0 ? s.frameMs*0.9+ms*0.1 : ms;
            }
            s.lastPrepare=now;
            const float threshold=s.minFrameMs.load(relaxed);
            if(!(threshold>0.f)) { s.paced=true; return true; }
            if(s.paced && s.frameMs<threshold*0.9) s.paced=false;
            else if(!s.paced && s.frameMs>=threshold) s.paced=true;
            if(!s.paced) s.fastFrames.fetch_add(1,relaxed);
            return s.paced;
        }

        // Hi-Z pays off only when it hides enough: at one Whiterun view it hid 4 of 2,424 objects tested per
        // frame and the frame was 3.7% slower than precombines alone. After 120 frames hiding less than hiz_min_yield of what
        // they tested (64 tests or more), it tests only hiz_probe_burst frames out of every hiz_probe_interval; a probe frame
        // that hides enough brings it back to every frame. Captures continue meanwhile, so a probe finds a fresh grid.
        bool YieldGate() {
            auto& s=S();
            auto& gate=s.gate;
            gate.minYield=s.minYield.load(relaxed); gate.interval=s.settings.probeInterval; gate.burst=s.settings.probeBurst;
            const bool active=gate.Step(s.lastFadeTested.load(relaxed)+s.lastGeometryTested.load(relaxed),
                                        s.lastFadeCulled.load(relaxed)+s.lastGeometryCulled.load(relaxed));
            if(!active) s.yieldSkipped.fetch_add(1,relaxed);
            return active;
        }

        // "hiz_ab", from Prepare: the interval since the previous Prepare is the time of the frame that just ended, and so
        // are the Hi-Z counts just taken; both go to that frame's block unless it was among the block's first abSkip frames.
        // A block ends after abFrames frames; a round is one block per state, in an order shuffled per round.
        void AbStep(State& s) {
            const auto now=Clock::now();
            if(s.abLast.time_since_epoch().count() && s.abInBlock!=~0u && s.abInBlock>s.settings.abSkip) {
                const double ms=std::chrono::duration<double,std::milli>(now-s.abLast).count();
                if(ms<1000.0) {
                    auto& b=s.abCurrent;
                    ++b.frames; b.sumMs+=ms; b.maxMs=std::max(b.maxMs,ms);
                    b.tested+=s.lastFadeTested.load(relaxed)+s.lastGeometryTested.load(relaxed);
                    b.culled+=s.lastFadeCulled.load(relaxed)+s.lastGeometryCulled.load(relaxed);
                }
            }
            s.abLast=now;
            if(s.abInBlock==~0u || s.abInBlock>=s.settings.abFrames) {
                if(s.abCurrent.frames) {
                    if(const auto* camera=s.camera.load(relaxed)) {
                        const auto& at=camera->world.translate;
                        s.abCurrent.eye={at.x,at.y,at.z};
                    }
                    std::lock_guard lock(s.abMutex);
                    if(s.abDone.size()<100000) s.abDone.push_back(s.abCurrent);
                }
                if(++s.abSlot>=kAbStates) {
                    s.abSlot=0; ++s.abRound;
                    for(uint32_t i=kAbStates-1;i>0;--i) {
                        s.abSeed^=s.abSeed<<13; s.abSeed^=s.abSeed>>17; s.abSeed^=s.abSeed<<5;
                        std::swap(s.abOrder[i],s.abOrder[s.abSeed%(i+1)]);
                    }
                }
                const auto state=static_cast<AbState>(s.abOrder[s.abSlot]);
                s.abCurrent=AbBlock{}; s.abCurrent.round=s.abRound; s.abCurrent.state=state;
                s.abState.store(uint8_t(state),relaxed);
                s.abInBlock=0;
            }
            ++s.abInBlock;
        }
        void PrepareBody();
        void BoxPrepare(const RE::NiCamera* camera);
        void AuditRead();
        void Prepare() {
            auto& s=S();
            // VR diagnostics: one census line per interval comparing this culler with the game's (VRDepthCulling).
            if(s.eyes==2) VRDepthCulling::OnFrame();
            if(!s.settings.diagnostics) return PrepareBody();
            const uint64_t begin=__rdtsc();
            PrepareBody();
            s.prepareTicks.fetch_add(__rdtsc()-begin,relaxed); s.prepareCalls.fetch_add(1,relaxed);
        }
        void PrepareBody() {
            auto& s=S();
            if(s.settings.frameLog) {
                const auto now=Clock::now();
                if(s.lastFrame.time_since_epoch().count()) {
                    const float ms=std::chrono::duration<float,std::milli>(now-s.lastFrame).count();
                    std::lock_guard lock(s.frameMutex);
                    if(s.frameTimes.size()<100000) s.frameTimes.push_back(ms);
                }
                s.lastFrame=now;
            }
            s.lastFadeTested.store(Take(kFadeTested),relaxed);
            s.lastFadeCulled.store(Take(kFadeCulled),relaxed);
            s.lastGeometryTested.store(Take(kGeometryTested),relaxed);
            s.lastGeometryCulled.store(Take(kGeometryCulled),relaxed);
            s.lastPrevisCulled.store(Take(kPrevisCulled),relaxed);
            s.lastReferencesCulled.store(Take(kReferencesCulled),relaxed);
            s.lastUntested.store(Take(kUntested),relaxed);
            s.lastSkipped.store(Take(kSkipped),relaxed);
            s.lastFadeKept.store(Take(kFadeKept),relaxed);
            if(s.settings.ab && s.enabled.load(relaxed)) AbStep(s);
            else s.abState.store(uint8_t(AbState::New),relaxed);
            if(s.sampling.exchange(false,relaxed)) {
                const uint64_t start=s.buildStartTsc.exchange(0,relaxed),published=s.publishTsc.exchange(0,relaxed),first=s.firstTestTsc.exchange(0,relaxed);
                if(start && first) {
                    s.raceFrames.fetch_add(1,relaxed);
                    s.firstTestTicks.fetch_add(first>start ? first-start : 0,relaxed);
                    if(published>start) s.publishTicks.fetch_add(published-start,relaxed);
                    if(!published || published>first) s.raceLost.fetch_add(1,relaxed);
                }
            }
            s.movingCount.store(0,relaxed); s.movingAtEye.store(false,relaxed); s.eyeExempt.store(0.f,relaxed);
            // The main view's culling processes carry this camera; previs and Hi-Z cull for it only.
            auto* camera=RE::Main::WorldRootCamera();
            s.camera.store(camera,relaxed);
            BoxPrepare(camera);
            if(s.eyes==1 && s.convention==Convention::Unknown && camera && s.enabled.load(relaxed)) {
                HiZ::Mat4 cameraToClip;
                HiZ::DepthModel model;
                if(CameraToClip(camera->GetRuntimeData2().viewFrustum,cameraToClip,model)) CrossCheck(camera,cameraToClip);
            }
            if(s.dumpPendingRead) DumpRead();
            if(s.settings.dumpSeconds>0.f && s.eyes==1 && s.dumps<s.settings.dumpMax && !s.dumpPendingCapture && !s.dumpPendingRead && camera &&
               s.enabled.load(relaxed) && (s.convention==Convention::Standard || s.convention==Convention::Reversed)) {
                const auto now=Clock::now();
                if(now-s.lastDump>=std::chrono::duration<float>(s.settings.dumpSeconds)) {
                    s.lastDump=now;
                    if(s.dumpEntries.empty()) s.dumpEntries.resize(State::kMaxDumpEntries);
                    s.dumpCount.store(0,relaxed);
                    s.dumpPendingCapture=true;
                    s.dumpRecording.store(true,std::memory_order_release);
                }
            }
            {
                // The hidden references were computed for one view cell and stay right anywhere within one cell of it
                // (each of those cells' visible sets is part of the union); a teleport or a stale set hides nothing.
                const auto* set=s.references.load(std::memory_order_acquire);
                bool usable=false;
                if(set && camera && set->size>0.f && !set->sorted.empty()) {
                    const auto& at=camera->world.translate;
                    const int64_t now[3]={static_cast<int64_t>(std::floor(at.x/set->size)),static_cast<int64_t>(std::floor(at.y/set->size)),
                                          static_cast<int64_t>(std::floor(at.z/set->size))};
                    usable=std::abs(now[0]-set->cell[0])<=1 && std::abs(now[1]-set->cell[1])<=1 && std::abs(now[2]-set->cell[2])<=1;
                }
                s.referencesUsable.store(usable,relaxed);
            }
            {
                std::lock_guard lock(s.jobMutex);
                ++s.frame;
                s.cullFrame.store(static_cast<uint32_t>(s.frame),relaxed);
                s.active.store(nullptr,std::memory_order_release);
                if(s.probeArmed.exchange(false,relaxed)) {
                    if(++s.probeTries<=State::kProbeTries) { s.probeLines.store(0,relaxed); s.probeFrame.store(static_cast<uint32_t>(s.frame),relaxed); }
                    else logger::info("Hi-Z probe: no frame with a pyramid in {} frames (Hi-Z off or paused?)",State::kProbeTries);
                }
                if(s.settings.diagnostics && s.enabled.load(relaxed) && s.frame%State::kSampleEvery==0) {
                    s.buildStartTsc.store(0,relaxed); s.publishTsc.store(0,relaxed); s.firstTestTsc.store(0,relaxed);
                    s.sampledFrames.fetch_add(1,relaxed);
                    s.sampling.store(true,relaxed);
                }
            }
            if(s.settings.audit && s.eyes==1) {
                // Last audit frames' counts, then whether this frame's culls are kept for AuditCapture (Occlusion::Audit).
                AuditRead();
                const bool start=camera && s.frame%std::max(1u,s.settings.auditEvery)==0;
                if(start) {
                    if(!s.auditEntries) s.auditEntries=std::make_unique<State::AuditEntry[]>(Audit::kMaxSpheres);
                    s.auditCount.store(0,relaxed);
                    s.auditEye={camera->world.translate.x,camera->world.translate.y,camera->world.translate.z};
                }
                s.auditing.store(start,relaxed);
            }
            if(s.settings.ab) {
                if(s.abState.load(relaxed)==uint8_t(AbState::Off) || !s.enabled.load(relaxed) || !s.initOk || !s.calibrated) return;
            } else if(!Paced() || !s.enabled.load(relaxed) || !s.initOk || !s.calibrated || !YieldGate()) return;
            // The helper still builds from the grid: leave it alone and cull nothing this frame.
            if(s.busy.load(std::memory_order_acquire)) { s.skippedFrames.fetch_add(1,relaxed); return; }
            if(!s.helperStarted) { s.helperStarted=true; std::thread(Helper).detach(); }
            const auto begin=Clock::now();
            for(uint32_t k=1;k<=kSlots && s.captures>=k;++k) {
                const uint32_t index=static_cast<uint32_t>((s.captures-k)%kSlots);
                auto& slot=s.slots[index];
                if(!slot.pending) continue;
                if(s.captures-slot.capture>kMaxAge) { slot.pending=false; continue; }
                D3D11_MAPPED_SUBRESOURCE mapped{};
                const HRESULT hr=s.context->Map(s.staging[index],0,D3D11_MAP_READ,D3D11_MAP_FLAG_DO_NOT_WAIT,&mapped);
                if(hr==DXGI_ERROR_WAS_STILL_DRAWING) continue;
                if(FAILED(hr)) { slot.pending=false; continue; }
                // On VR the texture's left and right halves are the two eyes' grids.
                const uint32_t width=s.gridW/s.eyes;
                for(uint32_t e=0;e<s.eyes;++e) {
                    auto& grid=s.grids[e];
                    grid.width=width; grid.height=s.gridH; grid.depth.resize(size_t(width)*s.gridH);
                    for(uint32_t y=0;y<s.gridH;++y)
                        std::memcpy(grid.depth.data()+size_t(y)*width,static_cast<const std::uint8_t*>(mapped.pData)+size_t(y)*mapped.RowPitch+size_t(e)*width*sizeof(float),
                                    width*sizeof(float));
                    grid.worldToClip=s.eyes==2 ? slot.eyeClip[e] : slot.clip;
                    grid.hasExact=s.eyes!=2;
                    if(grid.hasExact) grid.exact=slot.exact;
                }
                s.context->Unmap(s.staging[index],0);
                s.gridCapture=slot.capture; s.gridEye=slot.eye; s.haveGrid=true;
                if(!s.tracedReadback) { s.tracedReadback=true; logger::info("Hi-Z first readback: capture {} mapped at capture count {}",slot.capture,s.captures); }
                for(auto& other:s.slots) if(other.capture<=slot.capture) other.pending=false;
                break;
            }
            const uint64_t age=s.haveGrid ? s.captures-s.gridCapture : ~0ull;
            const auto& eye=camera ? camera->world.translate : RE::NiPoint3{};
            const float dx=eye.x-s.gridEye[0],dy=eye.y-s.gridEye[1],dz=eye.z-s.gridEye[2];
            HiZ::Mat4d inverse;
            // No grid, a stale one, a jump (teleport, load) or a changed projection (zoom): cull nothing. Each reason is
            // counted and logged at its 1st, 100th and 10,000th frame (with one mod set Hi-Z built nothing).
            // The grid keeps its capture's projection and this frame's pyramid gets the camera's current one, so a
            // projection that differs between the two (a mod widens the far plane from 353,840 to 371,648
            // around the depth prepass, every frame; or a zoom) is reprojected exactly instead of skipped.
            // On VR each eye keeps the projection of the newest capture in the camera's local space (StereoProjection).
            HiZ::Mat4 currentToClip[2];
            HiZ::DepthModel currentDepth[2];
            const bool stereo=s.eyes==2;
            if(stereo) for(int e=0;e<2;++e) { currentToClip[e]=s.eyeCameraToClip[e]; currentDepth[e]=s.eyeDepth[e]; }
            const int reason=!camera ? 0 : !s.haveGrid ? 1 : age>kMaxAge ? 2 : dx*dx+dy*dy+dz*dz>2048.f*2048.f ? 3 :
                             !stereo && !CameraToClip(camera->GetRuntimeData2().viewFrustum,currentToClip[0],currentDepth[0]) ? 4 :
                             !HiZ::Invert(CameraWorldExact(camera),inverse) ? 5 : -1;
            if(reason>=0) {
                static std::atomic<uint64_t> counts[6]{};
                static constexpr const char* names[6]={"no camera","no grid","stale grid","camera jump","degenerate projection","camera not invertible"};
                const auto n=counts[reason].fetch_add(1,relaxed)+1;
                if(n==1 || n==100 || n==10000) logger::info("Hi-Z skipped a frame ({}; {} frames)",names[reason],n);
                return;
            }
            if(!stereo && !SameFrustum(camera->GetRuntimeData2().viewFrustum,s.frustum) && !s.loggedProjection) {
                s.loggedProjection=true;
                const auto& a=camera->GetRuntimeData2().viewFrustum; const auto& b=s.frustum;
                logger::info("Hi-Z: the camera projection differs between culling and the depth capture (now near {} far {}, at capture near {} far {}); "
                             "each is used for its own frame",a.fNear,a.fFar,b.fNear,b.fFar);
            }
            {
                std::lock_guard lock(s.jobMutex);
                State::Job job{{},{eye.x,eye.y,eye.z},s.frame,age,&s.views[s.next],{},s.settings.nearExempt,s.eyes};
                // In double (see HiZ::Mat4d): from the frustum itself on SE and AE, from each eye's camera-space projection on VR.
                HiZ::Mat4d cameraToClip;
                for(uint32_t e=0;e<s.eyes;++e) {
                    if(stereo || !CameraToClip(camera->GetRuntimeData2().viewFrustum,cameraToClip)) cameraToClip=HiZ::ToDouble(currentToClip[e]);
                    job.exact[e]=HiZ::Multiply(cameraToClip,inverse);
                    job.clip[e]=HiZ::ToFloat(job.exact[e]);
                    job.depth[e]=currentDepth[e];
                }
                s.job=job;
                s.jobPending=true;
                s.waitDeadline.store((Clock::now()+std::chrono::microseconds(s.settings.waitMicros)).time_since_epoch().count(),relaxed);
                s.busy.store(true,std::memory_order_release);
                if(s.sampling.load(relaxed)) s.buildStartTsc.store(__rdtsc(),relaxed);
            }
            s.jobReady.notify_one();
            s.next^=1;
            s.gridAge.store(static_cast<uint32_t>(age),relaxed);
            s.prepareMicros.store(static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-begin).count()),relaxed);
        }

        void SplitWorker() {
            auto& s=S();
            while(true) {
                const std::function<void()>* task;
                {
                    std::unique_lock lock(s.splitMutex);
                    s.splitReady.wait(lock,[&]{ return s.splitTask!=nullptr; });
                    task=s.splitTask;
                }
                (*task)();
                {
                    std::lock_guard lock(s.splitMutex);
                    s.splitTask=nullptr; s.splitFinished=true;
                }
                s.splitDone.notify_one();
            }
        }
        // Pyramid::Both for the helper: the second half runs on the split worker while the helper builds the first.
        void SplitBoth(const std::function<void()>& first,const std::function<void()>& second) {
            auto& s=S();
            if(!s.splitStarted) { s.splitStarted=true; std::thread(SplitWorker).detach(); }
            {
                std::lock_guard lock(s.splitMutex);
                s.splitTask=&second; s.splitFinished=false;
            }
            s.splitReady.notify_one();
            first();
            std::unique_lock lock(s.splitMutex);
            s.splitDone.wait(lock,[&]{ return s.splitFinished; });
        }

        void Helper() {
            auto& s=S();
            while(true) {
                State::Job job;
                {
                    std::unique_lock lock(s.jobMutex);
                    s.jobReady.wait(lock,[&]{ return s.jobPending; });
                    job=s.job; s.jobPending=false;
                }
                const auto building=Clock::now();
                uint32_t landed=0;
                job.target->count=job.count;
                for(uint32_t e=0;e<job.count;++e)
                    landed+=s.settings.splitBuild ?
                        job.target->eye[e].Build(s.grids[e],job.clip[e],job.depth[e],job.eye,job.nearExempt,SplitBoth,&job.exact[e]) :
                        job.target->eye[e].Build(s.grids[e],job.clip[e],job.depth[e],job.eye,job.nearExempt,&job.exact[e]);
                const auto micros=static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-building).count());
                bool published=false;
                {
                    std::lock_guard lock(s.jobMutex);
                    if(job.frame==s.frame && s.enabled.load(relaxed) && job.target->Ready()) {
                        s.active.store(job.target,std::memory_order_release);
                        published=true;
                        if(s.sampling.load(relaxed)) s.publishTsc.store(__rdtsc(),relaxed);
                    }
                }
                s.busy.store(false,std::memory_order_release);
                s.buildMicros.store(micros,relaxed);
                s.buildMicrosSum.fetch_add(micros,relaxed); s.buildCalls.fetch_add(1,relaxed);
                s.landed.store(landed,relaxed);
                if(published) {
                    s.builds.fetch_add(1,relaxed);
                    s.publishedCaptures.store(s.captures,relaxed);
                } else s.lateBuilds.fetch_add(1,relaxed);
                if(published && !s.tracedBuild) {
                    s.tracedBuild=true;
                    logger::info("Hi-Z first pyramid: {} texels landed, grid age {}, built in {} us on the helper thread; culling starts",
                        landed,job.age,micros);
                }
            }
        }

        // ---- The GPU box test ("box_occlusion"; Occlusion::Boxes) ----
        // Each frame the main view registers the boxes of its static geometry (RegisterBox, culling threads); after
        // Main::RenderDepth they are drawn against that frame's depth, eroded for box_occlusion_max_shift of eye motion
        // (BoxCapture, Boxes::kErodeShader); a frame or two later Prepare reads the result back and publishes the hidden ones
        // (BoxPrepare), which the main view leaves out (BoxHidden) while the eye stays within that shift of where the result
        // was drawn. Objects new to the view have no result and are drawn; an object partly off screen or at the near plane is
        // never tested.
        constexpr uint64_t kMaxBoxAge=4, kBoxLogFrames=1800;
        constexpr int64_t kBoxVertexBudget=262144;  // mesh vertices read per frame for new boxes (about 0.3 ms at most)

        // A mesh's box in its own space from its vertices, cached by renderer data with what identifies its contents:
        // renderer data freed and reallocated for another mesh differs in its vertex buffer, its layout, its vertex count or
        // the model bound the box was checked against (bound: centre and radius; a hit on pointers alone could keep a box that
        // does not fit the geometry now drawn). The vertices are the CPU copy
        // (BSGraphics::TriShape::rawVertexData) where the engine keeps one; most static meshes have none (about 50 of
        // 860 main-view geometries in Blackreach), so their vertex buffer is copied once from the GPU (MeshReads) and the box
        // follows a frame or two later. Vertices outside the geometry's own model bound refuse it.
        struct ModelBox { const void* vertexBuffer=nullptr; uint64_t desc=0; uint32_t vertices=0; float bound[4]{}; float lo[3]{},hi[3]{}; bool ok=false, pending=false; };
        struct BoxShard { std::mutex mutex; std::unordered_map<const void*,ModelBox> boxes; };
        std::array<BoxShard,64>& BoxShards() { static auto* shards=new std::array<BoxShard,64>; return *shards; }
        BoxShard& ShardOf(const void* shape) { return BoxShards()[(reinterpret_cast<std::uintptr_t>(shape)>>6)%64]; }
        // Why a geometry has no box (MeshBox).
        enum class BoxSource : uint8_t { Have, Type, Waiting, Invalid };
        // A vertex buffer to copy (culling threads queue it, holding a reference; the render thread copies and releases).
        struct MeshRead { const void* shape=nullptr; ID3D11Buffer* buffer=nullptr; uint64_t desc=0; uint32_t vertices=0, bytes=0, offset=0;
                          float center[3]{}, radius=0.f; };
        struct MeshSlot { ID3D11Buffer* staging=nullptr; std::vector<MeshRead> reads; bool pending=false; };
        struct MeshReads {
            std::mutex mutex;
            std::vector<MeshRead> requests;  // guarded by mutex
            MeshSlot slots[kSlots];          // render thread
            bool initTried=false, initOk=false;
            std::atomic<uint64_t> boxes{0}, invalid{0}, cpuBoxes{0}, dropped{0};
            std::array<std::atomic<uint64_t>,16> types{};  // geometries without a box by BSGeometry::Type, since the start
        };
        MeshReads& Reads() { static auto* reads=new MeshReads; return *reads; }
        constexpr uint32_t kMeshStagingBytes=8u<<20, kMeshReadsPerFrame=64, kMaxMeshRequests=4096;

        float HalfToFloat(uint16_t h) {
            const uint32_t sign=uint32_t(h&0x8000u)<<16, exponent=(h>>10)&0x1Fu, mantissa=h&0x3FFu;
            if(exponent==0) {
                const float value=std::ldexp(float(mantissa),-24);
                return sign ? -value : value;
            }
            const uint32_t bits=exponent==31 ? sign|0x7F800000u|(mantissa<<13) : sign|((exponent+112)<<23)|(mantissa<<13);
            float out;
            std::memcpy(&out,&bits,4);
            return out;
        }

        // Interleaved vertices of a layout `desc` (BSGraphics::VertexDesc: the stride in dwords in the low four bits, the
        // position's offset, full precision or half): their box, refused when a vertex is not finite or lies outside the
        // model bound (centre, radius) by more than 1% and a unit.
        // Whether a layout holds full-precision positions (float x, y, z and the bitangent's x) or half: by the room the
        // position takes up to the next attribute (or the vertex's end). The flag is not enough: the engine keeps loaded
        // meshes in full precision without it (desc 0003B00007650408, stride 32, the UV at 16 bytes, positions
        // plain floats; read as half they were garbage, and 757 of 798 boxes were refused).
        bool FullPrecision(uint64_t desc) {
            const uint64_t flags=desc>>44;
            const uint32_t stride=uint32_t(desc&0xF)*4,position=uint32_t(desc>>2)&0x3C;
            uint32_t next=stride;
            for(uint32_t attribute=1;attribute<RE::BSGraphics::Vertex::VA_COUNT;++attribute) {
                if(!(flags&(1ull<<attribute))) continue;
                const uint32_t at=uint32_t(desc>>(4*attribute+2))&0x3C;
                if(at>position) next=std::min(next,at);
            }
            return (flags&RE::BSGraphics::Vertex::VF_FULLPREC)!=0 || next>=position+12;
        }

        bool BoxFromVertices(const std::uint8_t* data,uint32_t vertices,uint64_t desc,const float center[3],float radius,float lo[3],float hi[3]) {
            const uint32_t stride=uint32_t(desc&0xF)*4;
            const uint32_t offset=uint32_t(desc>>2)&0x3C;  // VertexDesc::GetAttributeOffset(VA_POSITION)
            const uint64_t flags=desc>>44;
            const bool full=FullPrecision(desc);
            if(!data || !vertices || !(flags&RE::BSGraphics::Vertex::VF_VERTEX) || stride<offset+(full ? 12u : 6u)) return false;
            const float limit=radius*1.01f+1.f;
            for(int k=0;k<3;++k) { lo[k]=std::numeric_limits<float>::max(); hi[k]=-std::numeric_limits<float>::max(); }
            for(uint32_t v=0;v<vertices;++v) {
                const auto* at=data+size_t(v)*stride+offset;
                float p[3];
                if(full) std::memcpy(p,at,sizeof(p));
                else for(int k=0;k<3;++k) { uint16_t h; std::memcpy(&h,at+2*k,2); p[k]=HalfToFloat(h); }
                const float dx=p[0]-center[0],dy=p[1]-center[1],dz=p[2]-center[2];
                if(!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]) || dx*dx+dy*dy+dz*dz>limit*limit) return false;
                for(int k=0;k<3;++k) { lo[k]=std::min(lo[k],p[k]); hi[k]=std::max(hi[k],p[k]); }
            }
            return true;
        }

        void StoreBox(const void* shape,const ModelBox& box) {
            auto& shard=ShardOf(shape);
            std::lock_guard lock(shard.mutex);
            if(shard.boxes.size()>=8192) shard.boxes.clear();  // freed meshes' entries; the live ones come back
            shard.boxes[shape]=box;
        }

        // The box of a static triangle shape in its own space, or why there is none: another geometry type or no renderer
        // data (Type); its vertices still on their way (Waiting: a GPU copy queued or in flight, the queue full, or this
        // frame's CPU budget spent); vertices that do not fit its model bound (Invalid).
        BoxSource MeshBox(const RE::BSGeometry& geometry,float lo[3],float hi[3]) {
            auto& s=S();
            auto& reads=Reads();
            const auto type=geometry.GetType().get();
            if(type!=RE::BSGeometry::Type::kTriShape && type!=RE::BSGeometry::Type::kMeshLODTriShape) {
                reads.types[static_cast<uint8_t>(type)&15].fetch_add(1,relaxed);
                return BoxSource::Type;
            }
            const auto* shape=geometry.GetGeometryRuntimeData().rendererData;
            if(!shape || !shape->vertexBuffer) { reads.types[static_cast<uint8_t>(type)&15].fetch_add(1,relaxed); return BoxSource::Type; }
            uint64_t desc;
            std::memcpy(&desc,&shape->vertexDesc,sizeof(desc));
            const uint32_t vertices=static_cast<const RE::BSTriShape&>(geometry).GetTrishapeRuntimeData().vertexCount;
            const auto& bound=geometry.GetModelData().modelBound;
            const float center[3]={bound.center.x,bound.center.y,bound.center.z};
            // A bound that is not a number would let every vertex through BoxFromVertices and never match a cached box.
            if(!std::isfinite(center[0]) || !std::isfinite(center[1]) || !std::isfinite(center[2]) || !std::isfinite(bound.radius)) return BoxSource::Invalid;
            const float key[4]={center[0],center[1],center[2],bound.radius};
            auto& shard=ShardOf(shape);
            bool request=false;
            {
                std::lock_guard lock(shard.mutex);
                const auto it=shard.boxes.find(shape);
                if(it!=shard.boxes.end() && it->second.vertexBuffer==shape->vertexBuffer && it->second.desc==desc && it->second.vertices==vertices &&
                   std::equal(key,key+4,it->second.bound)) {
                    const auto& cached=it->second;
                    if(cached.pending) return BoxSource::Waiting;
                    if(!cached.ok) return BoxSource::Invalid;
                    std::memcpy(lo,cached.lo,sizeof(cached.lo)); std::memcpy(hi,cached.hi,sizeof(cached.hi));
                    return BoxSource::Have;
                }
                // No CPU copy: one GPU copy of the vertex buffer, requested once (the pending entry stops the others).
                if(!shape->rawVertexData) {
                    if(shard.boxes.size()>=8192) shard.boxes.clear();
                    shard.boxes[shape]=ModelBox{shape->vertexBuffer,desc,vertices,{key[0],key[1],key[2],key[3]},{},{},false,true};
                    request=true;
                }
            }
            if(request) {
                MeshRead read{shape,reinterpret_cast<ID3D11Buffer*>(shape->vertexBuffer),desc,vertices};
                std::memcpy(read.center,center,sizeof(center)); read.radius=bound.radius;
                read.buffer->AddRef();  // alive here (the geometry is being drawn); released once the copy is queued
                bool queued=false;
                {
                    std::lock_guard lock(reads.mutex);
                    if(reads.requests.size()<kMaxMeshRequests) { reads.requests.push_back(read); queued=true; }
                }
                if(!queued) {
                    read.buffer->Release();
                    std::lock_guard lock(shard.mutex);
                    if(const auto it=shard.boxes.find(shape);it!=shard.boxes.end() && it->second.pending) shard.boxes.erase(it);
                }
                return BoxSource::Waiting;
            }
            if(s.boxVertexBudget.fetch_sub(vertices,relaxed)<int64_t(vertices)) return BoxSource::Waiting;
            ModelBox box{shape->vertexBuffer,desc,vertices,{key[0],key[1],key[2],key[3]}};
            box.ok=BoxFromVertices(shape->rawVertexData,vertices,desc,center,bound.radius,box.lo,box.hi);
            // Diagnostics (757 of 798 Blackreach geometries refused): the first refused and the first accepted
            // meshes in the log, with both layouts the engine keeps (the renderer data's and the geometry's).
            static std::atomic<uint32_t> described{0};
            if(const uint32_t n=described.load(relaxed);(box.ok ? n<4 : n<24) && described.fetch_add(1,relaxed)<28) {
                uint64_t own;
                std::memcpy(&own,&geometry.GetGeometryRuntimeData().vertexDesc,sizeof(own));
                std::string first;
                const uint32_t stride=uint32_t(desc&0xF)*4,offset=uint32_t(desc>>2)&0x3C;
                const bool full=FullPrecision(desc);
                float worst=0.f; uint32_t worstAt=0;
                for(uint32_t v=0;v<vertices && stride;++v) {
                    const auto* at=shape->rawVertexData+size_t(v)*stride+offset;
                    float p[3];
                    if(full) std::memcpy(p,at,sizeof(p));
                    else for(int k=0;k<3;++k) { uint16_t h; std::memcpy(&h,at+2*k,2); p[k]=HalfToFloat(h); }
                    if(v<3) first+=std::format("({:.1f},{:.1f},{:.1f}) ",p[0],p[1],p[2]);
                    const float d=std::sqrt((p[0]-center[0])*(p[0]-center[0])+(p[1]-center[1])*(p[1]-center[1])+(p[2]-center[2])*(p[2]-center[2]));
                    if(d>worst) { worst=d; worstAt=v; }
                }
                std::string bytes;
                for(uint32_t i=0;i<std::min(48u,stride*2);++i) bytes+=std::format("{:02x}{}",shape->rawVertexData[i],(i+1)%4 ? "" : " ");
                const auto* name=geometry.name.c_str();
                logger::info("Box mesh {} \"{}\": type {}, {} vertices, desc {:016X} (geometry's {:016X}), stride {}, position at {}, {} precision; "
                             "model bound ({:.1f},{:.1f},{:.1f}) r {:.1f}; first vertices {}; farthest {:.1f} at vertex {}; box {}; raw {}",
                    box.ok ? "accepted" : "refused",name ? name : "",static_cast<uint32_t>(type),vertices,desc,own,stride,offset,full ? "full" : "half",
                    center[0],center[1],center[2],bound.radius,first,worst,worstAt,
                    box.ok ? std::format("({:.1f},{:.1f},{:.1f})-({:.1f},{:.1f},{:.1f})",box.lo[0],box.lo[1],box.lo[2],box.hi[0],box.hi[1],box.hi[2]) : std::string("none"),
                    bytes);
            }
            // A box far smaller than its bound sphere is no mesh's: refused (the sphere holds the box's half diagonal).
            if(box.ok) {
                const float dx=box.hi[0]-box.lo[0],dy=box.hi[1]-box.lo[1],dz=box.hi[2]-box.lo[2];
                if(0.5f*std::sqrt(dx*dx+dy*dy+dz*dz)<0.2f*bound.radius-1.f) box.ok=false;
            }
            StoreBox(shape,box);
            if(box.ok) reads.cpuBoxes.fetch_add(1,relaxed);
            if(!box.ok) return BoxSource::Invalid;
            std::memcpy(lo,box.lo,sizeof(box.lo)); std::memcpy(hi,box.hi,sizeof(box.hi));
            return BoxSource::Have;
        }

        // Render thread, every frame of the box test: finished vertex-buffer copies become boxes; queued requests are
        // copied into a free staging buffer (at most kMeshReadsPerFrame and kMeshStagingBytes a frame). With the box test
        // off, the queue is emptied (its references released, its meshes asked again later).
        void MeshReadsStep(bool on) {
            auto& s=S();
            auto& reads=Reads();
            if(!reads.initTried && on && s.device) {
                reads.initTried=true;
                D3D11_BUFFER_DESC buffer{};
                buffer.ByteWidth=kMeshStagingBytes; buffer.Usage=D3D11_USAGE_STAGING; buffer.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
                reads.initOk=true;
                for(auto& slot:reads.slots) if(FAILED(s.device->CreateBuffer(&buffer,nullptr,&slot.staging))) reads.initOk=false;
                if(!reads.initOk) logger::warn("Box occlusion: no staging buffers for vertex reads; only meshes with a CPU copy get boxes");
            }
            for(auto& slot:reads.slots) {
                if(!slot.pending) continue;
                D3D11_MAPPED_SUBRESOURCE mapped{};
                const HRESULT hr=s.context->Map(slot.staging,0,D3D11_MAP_READ,D3D11_MAP_FLAG_DO_NOT_WAIT,&mapped);
                if(hr==DXGI_ERROR_WAS_STILL_DRAWING) continue;
                for(const auto& read:slot.reads) {
                    ModelBox box{read.buffer,read.desc,read.vertices,{read.center[0],read.center[1],read.center[2],read.radius}};
                    // Only what was copied is read (the copy stops at the buffer's end); a buffer holding fewer vertices than the
                    // geometry draws gives no box.
                    const uint32_t stride=uint32_t(read.desc&0xF)*4,vertices=stride ? std::min(read.vertices,read.bytes/stride) : 0;
                    box.ok=SUCCEEDED(hr) && vertices==read.vertices && BoxFromVertices(static_cast<const std::uint8_t*>(mapped.pData)+read.offset,vertices,
                                                                                       read.desc,read.center,read.radius,box.lo,box.hi);
                    StoreBox(read.shape,box);
                    (box.ok ? reads.boxes : reads.invalid).fetch_add(1,relaxed);
                }
                if(SUCCEEDED(hr)) s.context->Unmap(slot.staging,0);
                slot.reads.clear();
                slot.pending=false;
            }
            std::vector<MeshRead> taken;
            {
                std::lock_guard lock(reads.mutex);
                if(!on || !reads.initOk) taken.swap(reads.requests);
                else {
                    const size_t n=std::min<size_t>(reads.requests.size(),kMeshReadsPerFrame);
                    taken.assign(reads.requests.begin(),reads.requests.begin()+n);
                    reads.requests.erase(reads.requests.begin(),reads.requests.begin()+n);
                }
            }
            if(taken.empty()) return;
            MeshSlot* free=nullptr;
            for(auto& slot:reads.slots) if(!slot.pending) { free=&slot; break; }
            uint32_t offset=0;
            std::vector<MeshRead> back;
            for(auto& read:taken) {
                D3D11_BUFFER_DESC desc{};
                read.buffer->GetDesc(&desc);
                read.bytes=std::min<uint32_t>(desc.ByteWidth,read.vertices*uint32_t(read.desc&0xF)*4);
                if(on && reads.initOk && free && read.bytes && offset+read.bytes<=kMeshStagingBytes) {
                    read.offset=offset;
                    const D3D11_BOX range{0,0,0,read.bytes,1,1};
                    s.context->CopySubresourceRegion(free->staging,0,offset,0,0,read.buffer,0,&range);
                    offset=(offset+read.bytes+15)&~15u;
                    free->reads.push_back(read);
                } else if(on && reads.initOk && read.bytes && read.bytes<=kMeshStagingBytes) {
                    back.push_back(read);  // no room this frame: keeps its reference and waits
                    continue;
                } else {
                    // Dropped (box test off, no staging, empty buffer): its mesh is asked again when seen.
                    auto& shard=ShardOf(read.shape);
                    std::lock_guard lock(shard.mutex);
                    if(const auto it=shard.boxes.find(read.shape);it!=shard.boxes.end() && it->second.pending) shard.boxes.erase(it);
                    reads.dropped.fetch_add(1,relaxed);
                }
                read.buffer->Release();  // the runtime keeps a copy's source alive until the GPU has run it
            }
            if(free && !free->reads.empty()) free->pending=true;
            if(!back.empty()) {
                std::lock_guard lock(reads.mutex);
                reads.requests.insert(reads.requests.begin(),back.begin(),back.end());
            }
        }

        uint32_t BoxHash(const void* object) { return static_cast<uint32_t>(((reinterpret_cast<std::uintptr_t>(object)>>4)*0x9E3779B97F4A7C15ull)>>32); }

        // A culling thread's slots in this frame's arrays, taken kBoxBatch at a time (one atomic step per batch; the
        // slots a batch leaves unused stay empty).
        struct BoxBatch { uint32_t epoch=~0u, next=0, end=0; };
        constexpr uint32_t kBoxBatch=16;

        // Main view, static geometry: its box relative to this frame's eye for this frame's test, grown by 2 units and 0.2% of
        // its distance from the eye (depth precision, the downsampled target), and every half extent at least one texel of the
        // depth target at that distance (Boxes::TexelSpan). Not tested: no box, a corner at the near plane or within the
        // near exemption, or a corner within 2% of the screen edge (its unseen part could turn into view before the
        // result is used).
        void RegisterBox(const RE::BSGeometry& geometry) {
            auto& s=S();
            if(!(s.boxTexel>0.f)) return;
            float lo[3],hi[3];
            if(const auto source=MeshBox(geometry,lo,hi);source!=BoxSource::Have) {
                Bump(kBoxNoModel);
                Bump(source==BoxSource::Type ? kBoxType : source==BoxSource::Waiting ? kBoxWaiting : kBoxInvalid);
                return;
            }
            const auto& world=geometry.world;
            const auto& r=world.rotate.entry;
            const float scale=world.scale;
            const float local[3]={0.5f*(lo[0]+hi[0]),0.5f*(lo[1]+hi[1]),0.5f*(lo[2]+hi[2])};
            const float origin[3]={world.translate.x,world.translate.y,world.translate.z};
            // The centre relative to the eye, the difference taken in double: in world space the box pass's float projection was
            // off by about 0.15% in depth far from the origin (Fort Amol, about (98000,-18500)), close to the growth below;
            // relative to the eye it is exact to a few thousandths of a unit.
            float center[3];
            for(int i=0;i<3;++i)
                center[i]=static_cast<float>(double(origin[i])-double(s.boxEye[i])+double(scale*(r[i][0]*local[0]+r[i][1]*local[1]+r[i][2]*local[2])));
            const float distance=std::sqrt(center[0]*center[0]+center[1]*center[1]+center[2]*center[2]);
            const float grow=2.f+0.002f*distance,least=distance*s.boxTexel;
            // Column k of the rotation is the mesh's axis k in the world.
            float axis[3][3];
            for(int k=0;k<3;++k) {
                const float half=std::max(0.5f*(hi[k]-lo[k])*scale+grow,least);
                for(int i=0;i<3;++i) axis[k][i]=r[i][k]*half;
            }
            const auto& m=s.boxClip.m;
            float c[4],a[3][4];
            for(int row=0;row<4;++row) {
                c[row]=m[row][0]*center[0]+m[row][1]*center[1]+m[row][2]*center[2]+m[row][3];
                for(int k=0;k<3;++k) a[k][row]=m[row][0]*axis[k][0]+m[row][1]*axis[k][1]+m[row][2]*axis[k][2];
            }
            const float nearest=std::max(s.boxNear,s.settings.nearExempt);
            for(uint32_t corner=0;corner<8;++corner) {
                float p[4];
                for(int row=0;row<4;++row)
                    p[row]=c[row]+(corner&1 ? a[0][row] : -a[0][row])+(corner&2 ? a[1][row] : -a[1][row])+(corner&4 ? a[2][row] : -a[2][row]);
                if(!(p[3]>nearest) || !(std::abs(p[0])<=0.98f*p[3]) || !(std::abs(p[1])<=0.98f*p[3])) { Bump(kBoxEdge); return; }
            }
            thread_local BoxBatch batch;
            const uint32_t epoch=s.boxEpoch.load(relaxed);
            if(batch.epoch!=epoch || batch.next>=batch.end) {
                batch.epoch=epoch;
                const uint32_t first=s.boxCount.fetch_add(kBoxBatch,relaxed);
                if(first>=Boxes::kMaxBoxes) { batch.next=batch.end=0; return; }
                batch.next=first; batch.end=std::min(first+kBoxBatch,Boxes::kMaxBoxes);
                for(uint32_t i=batch.next;i<batch.end;++i) { s.boxes[i]=Boxes::Box{}; s.boxKeys[i]=State::BoxKey{}; }
            }
            const uint32_t slot=batch.next++;
            auto& box=s.boxes[slot];
            for(int i=0;i<3;++i) { box.center[i]=center[i]; for(int k=0;k<3;++k) box.axis[k][i]=axis[k][i]; }
            const auto& bound=geometry.worldBound;
            auto& key=s.boxKeys[slot];
            key.object=&geometry; key.center[0]=bound.center.x; key.center[1]=bound.center.y; key.center[2]=bound.center.z; key.radius=bound.radius;
            // The frame last, once the box and the key are written: BoxCaptureBody takes a slot only when it reads this frame there.
            std::atomic_ref(key.epoch).store(epoch,std::memory_order_release);
            Bump(kBoxRegistered);
        }

        // Hidden in the published result, and unchanged since: the same world bound (bit for bit, recomputed each frame
        // from the same transforms; a moved, swaying or rescaled object differs and is drawn).
        bool BoxHidden(const RE::BSGeometry& geometry) {
            auto& s=S();
            const auto* table=s.boxActive.load(std::memory_order_acquire);
            if(!table || !table->hidden) return false;
            const auto* object=static_cast<const RE::NiAVObject*>(&geometry);
            for(uint32_t i=BoxHash(object)&table->mask;;i=(i+1)&table->mask) {
                const auto& entry=table->entries[i];
                if(!entry.object) return false;
                if(entry.object!=object) continue;
                const auto& bound=geometry.worldBound;
                if(entry.center[0]!=bound.center.x || entry.center[1]!=bound.center.y || entry.center[2]!=bound.center.z || entry.radius!=bound.radius) return false;
                Bump(kBoxCulled);
                return true;
            }
        }

        // The engine's graphics pipeline state that the box pass changes, put back exactly as found.
        struct PipelineGuard {
            ID3D11DeviceContext* c;
            ID3D11InputLayout* layout{}; D3D11_PRIMITIVE_TOPOLOGY topology{};
            ID3D11Buffer* vertexBuffer{}; UINT stride=0, offset=0; ID3D11Buffer* indexBuffer{}; DXGI_FORMAT indexFormat{}; UINT indexOffset=0;
            ID3D11VertexShader* vs{}; ID3D11ShaderResourceView* vsView{}; ID3D11Buffer* vsParams{};
            ID3D11HullShader* hs{}; ID3D11DomainShader* ds{}; ID3D11GeometryShader* gs{};
            ID3D11PixelShader* ps{}; ID3D11ShaderResourceView* psView{}; ID3D11Buffer* psParams{};
            ID3D11RasterizerState* raster{};
            D3D11_VIEWPORT viewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE]{}; UINT viewportCount=D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
            D3D11_RECT scissors[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE]{}; UINT scissorCount=D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
            ID3D11RenderTargetView* targets[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT]{}; ID3D11DepthStencilView* depth{};
            ID3D11UnorderedAccessView* uavs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT]{};
            ID3D11DepthStencilState* depthState{}; UINT stencilRef=0;
            ID3D11BlendState* blend{}; float blendFactor[4]{}; UINT sampleMask=0;
            explicit PipelineGuard(ID3D11DeviceContext* context):c(context) {
                c->IAGetInputLayout(&layout); c->IAGetPrimitiveTopology(&topology);
                c->IAGetVertexBuffers(0,1,&vertexBuffer,&stride,&offset); c->IAGetIndexBuffer(&indexBuffer,&indexFormat,&indexOffset);
                c->VSGetShader(&vs,nullptr,nullptr); c->VSGetShaderResources(0,1,&vsView); c->VSGetConstantBuffers(0,1,&vsParams);
                c->HSGetShader(&hs,nullptr,nullptr); c->DSGetShader(&ds,nullptr,nullptr); c->GSGetShader(&gs,nullptr,nullptr);
                c->PSGetShader(&ps,nullptr,nullptr); c->PSGetShaderResources(0,1,&psView); c->PSGetConstantBuffers(0,1,&psParams);
                c->RSGetState(&raster); c->RSGetViewports(&viewportCount,viewports); c->RSGetScissorRects(&scissorCount,scissors);
                c->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT,targets,&depth);
                c->OMGetRenderTargetsAndUnorderedAccessViews(0,nullptr,nullptr,0,D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT,uavs);
                c->OMGetDepthStencilState(&depthState,&stencilRef); c->OMGetBlendState(&blend,blendFactor,&sampleMask);
            }
            ~PipelineGuard() {
                c->IASetInputLayout(layout); c->IASetPrimitiveTopology(topology);
                c->IASetVertexBuffers(0,1,&vertexBuffer,&stride,&offset); c->IASetIndexBuffer(indexBuffer,indexFormat,indexOffset);
                c->VSSetShader(vs,nullptr,0); c->VSSetShaderResources(0,1,&vsView); c->VSSetConstantBuffers(0,1,&vsParams);
                c->HSSetShader(hs,nullptr,0); c->DSSetShader(ds,nullptr,0); c->GSSetShader(gs,nullptr,0);
                c->PSSetShader(ps,nullptr,0); c->PSSetShaderResources(0,1,&psView); c->PSSetConstantBuffers(0,1,&psParams);
                c->RSSetState(raster); c->RSSetViewports(viewportCount,viewports); c->RSSetScissorRects(scissorCount,scissors);
                // Render targets below their highest bound slot, unordered access views from there (the engine binds none).
                UINT used=D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT;
                while(used && !targets[used-1]) --used;
                const UINT keep[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT]={~0u,~0u,~0u,~0u,~0u,~0u,~0u,~0u};
                if(used==D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT) c->OMSetRenderTargets(used,targets,depth);
                else c->OMSetRenderTargetsAndUnorderedAccessViews(used,targets,depth,used,D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT-used,uavs+used,keep);
                c->OMSetDepthStencilState(depthState,stencilRef); c->OMSetBlendState(blend,blendFactor,sampleMask);
                for(IUnknown* held:std::initializer_list<IUnknown*>{layout,vertexBuffer,indexBuffer,vs,vsView,vsParams,hs,ds,gs,ps,psView,psParams,raster,depth,depthState,blend})
                    if(held) held->Release();
                for(auto* view:targets) if(view) view->Release();
                for(auto* view:uavs) if(view) view->Release();
            }
        };

        bool InitBoxGpu() {
            auto& s=S();
            if(s.boxInitTried) return s.boxInitOk;
            s.boxInitTried=true;
            const auto fail=[](const char* what,HRESULT hr=S_OK) { logger::error("Box occlusion disabled: {} (0x{:08X})",what,static_cast<uint32_t>(hr)); return false; };
            const HMODULE compiler=LoadLibraryW(L"d3dcompiler_47.dll");
            const auto compile=compiler ? reinterpret_cast<pD3DCompile>(GetProcAddress(compiler,"D3DCompile")) : nullptr;
            if(!compile) return fail("d3dcompiler_47.dll unavailable");
            const auto build=[&](const char* source,size_t size,const char* entry,const char* target,ID3DBlob*& code) {
                ID3DBlob* errors{};
                const HRESULT hr=compile(source,size,"PrecombinesBoxes",nullptr,nullptr,entry,target,D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&code,&errors);
                if(errors) {
                    if(FAILED(hr)) logger::error("Box occlusion shader {}: {}",entry,static_cast<const char*>(errors->GetBufferPointer()));
                    errors->Release();
                }
                return SUCCEEDED(hr);
            };
            ID3DBlob* code{};
            HRESULT hr=S_OK;
            if(!build(Boxes::kFillShader,sizeof(Boxes::kFillShader)-1,"vs","vs_5_0",code)) return fail("fill vertex shader");
            hr=s.device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&s.fillVertex); code->Release();
            if(FAILED(hr)) return fail("CreateVertexShader (fill)",hr);
            if(!build(Boxes::kFillShader,sizeof(Boxes::kFillShader)-1,"ps","ps_5_0",code)) return fail("fill pixel shader");
            hr=s.device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&s.fillPixel); code->Release();
            if(FAILED(hr)) return fail("CreatePixelShader (fill)",hr);
            if(!build(Boxes::kBoxShader,sizeof(Boxes::kBoxShader)-1,"vs","vs_5_0",code)) return fail("box vertex shader");
            hr=s.device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&s.boxVertex); code->Release();
            if(FAILED(hr)) return fail("CreateVertexShader (boxes)",hr);
            if(!build(Boxes::kBoxShader,sizeof(Boxes::kBoxShader)-1,"ps","ps_5_0",code)) return fail("box pixel shader");
            hr=s.device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&s.boxPixel); code->Release();
            if(FAILED(hr)) return fail("CreatePixelShader (boxes)",hr);
            // Motion erosion (Boxes::kErodeShader): without it no result could be used while the eye moves.
            for(const auto& [entry,shader]:{std::pair{"edges",&s.erodeEdges},std::pair{"down",&s.erodeDown},std::pair{"expose",&s.erodeExpose}}) {
                if(!build(Boxes::kErodeShader,sizeof(Boxes::kErodeShader)-1,entry,"cs_5_0",code)) return fail("erosion compute shader");
                hr=s.device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,shader); code->Release();
                if(FAILED(hr)) return fail("CreateComputeShader (erosion)",hr);
            }
            if(!build(Boxes::kErodeShader,sizeof(Boxes::kErodeShader)-1,"apply","ps_5_0",code)) return fail("erosion pixel shader");
            hr=s.device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&s.erodeApply); code->Release();
            if(FAILED(hr)) return fail("CreatePixelShader (erosion)",hr);
            D3D11_BUFFER_DESC buffer{};
            buffer.ByteWidth=sizeof(Reduce::Params); buffer.Usage=D3D11_USAGE_DEFAULT; buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            if(FAILED(hr=s.device->CreateBuffer(&buffer,nullptr,&s.fillParams))) return fail("CreateBuffer (fill parameters)",hr);
            buffer.ByteWidth=sizeof(Boxes::BoxParams);
            if(FAILED(hr=s.device->CreateBuffer(&buffer,nullptr,&s.boxParams))) return fail("CreateBuffer (box parameters)",hr);
            buffer.ByteWidth=sizeof(Boxes::ErodeParams);
            if(FAILED(hr=s.device->CreateBuffer(&buffer,nullptr,&s.erodeParams))) return fail("CreateBuffer (erosion parameters)",hr);
            buffer=D3D11_BUFFER_DESC{};
            buffer.ByteWidth=sizeof(Boxes::Box)*Boxes::kMaxBoxes; buffer.Usage=D3D11_USAGE_DYNAMIC; buffer.BindFlags=D3D11_BIND_SHADER_RESOURCE;
            buffer.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE; buffer.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; buffer.StructureByteStride=sizeof(Boxes::Box);
            if(FAILED(hr=s.device->CreateBuffer(&buffer,nullptr,&s.boxBuffer))) return fail("CreateBuffer (boxes)",hr);
            D3D11_SHADER_RESOURCE_VIEW_DESC view{};
            view.Format=DXGI_FORMAT_UNKNOWN; view.ViewDimension=D3D11_SRV_DIMENSION_BUFFER; view.Buffer.FirstElement=0; view.Buffer.NumElements=Boxes::kMaxBoxes;
            if(FAILED(hr=s.device->CreateShaderResourceView(s.boxBuffer,&view,&s.boxView))) return fail("CreateShaderResourceView (boxes)",hr);
            buffer=D3D11_BUFFER_DESC{};
            buffer.ByteWidth=4*Boxes::kMaxBoxes; buffer.Usage=D3D11_USAGE_DEFAULT; buffer.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
            buffer.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
            if(FAILED(hr=s.device->CreateBuffer(&buffer,nullptr,&s.boxResults))) return fail("CreateBuffer (results)",hr);
            D3D11_UNORDERED_ACCESS_VIEW_DESC access{};
            access.Format=DXGI_FORMAT_R32_TYPELESS; access.ViewDimension=D3D11_UAV_DIMENSION_BUFFER; access.Buffer.NumElements=Boxes::kMaxBoxes;
            access.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;
            if(FAILED(hr=s.device->CreateUnorderedAccessView(s.boxResults,&access,&s.boxResultsView))) return fail("CreateUnorderedAccessView (results)",hr);
            buffer=D3D11_BUFFER_DESC{};
            buffer.ByteWidth=4*Boxes::kMaxBoxes; buffer.Usage=D3D11_USAGE_STAGING; buffer.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
            for(auto& slot:s.boxSlots) if(FAILED(hr=s.device->CreateBuffer(&buffer,nullptr,&slot.staging))) return fail("CreateBuffer (readback)",hr);
            D3D11_DEPTH_STENCIL_DESC depth{};
            depth.DepthEnable=TRUE; depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL; depth.DepthFunc=D3D11_COMPARISON_ALWAYS;
            if(FAILED(hr=s.device->CreateDepthStencilState(&depth,&s.fillState))) return fail("CreateDepthStencilState (fill)",hr);
            depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO; depth.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
            if(FAILED(hr=s.device->CreateDepthStencilState(&depth,&s.boxState))) return fail("CreateDepthStencilState (boxes)",hr);
            D3D11_RASTERIZER_DESC raster{};
            raster.FillMode=D3D11_FILL_SOLID; raster.CullMode=D3D11_CULL_NONE; raster.DepthClipEnable=TRUE;
            if(FAILED(hr=s.device->CreateRasterizerState(&raster,&s.boxRaster))) return fail("CreateRasterizerState",hr);
            s.boxInitOk=true;
            logger::info("Box occlusion ready: up to {} boxes a frame, depth target 1/{} of the rendered area, {} readback slots",
                Boxes::kMaxBoxes,std::clamp(s.settings.boxDownsample,1u,8u),kSlots);
            return true;
        }

        template<class T> void Drop(T*& object) { if(object) { object->Release(); object=nullptr; } }
        void DropBoxTarget() {
            auto& s=S();
            Drop(s.boxDepthRead); Drop(s.boxDepthView); Drop(s.boxDepth);
            Drop(s.exposedRead); Drop(s.exposedWrite); Drop(s.exposed); Drop(s.cliffChain);
            for(auto*& view:s.cliffRead) Drop(view);
            for(auto*& view:s.cliffWrite) Drop(view);
            s.cliffRead.clear(); s.cliffWrite.clear();
            Drop(s.cliffs);
            s.boxDepthWidth=s.boxDepthHeight=0;
        }

        // The box pass's own depth target, (re)created at the size the rendered area asks for, with what the motion erosion
        // needs at that size: a shader view of the target, the cliff map with its whole mip chain (a view of each level to
        // write and to read it, one of the whole chain) and the exposed texels of level 1.
        bool BoxTarget(uint32_t width,uint32_t height) {
            auto& s=S();
            if(s.boxDepth && s.boxDepthWidth==width && s.boxDepthHeight==height) return true;
            DropBoxTarget();
            const uint32_t levels=Boxes::Levels(width,height);
            D3D11_TEXTURE2D_DESC texture{};
            texture.Width=width; texture.Height=height; texture.MipLevels=1; texture.ArraySize=1; texture.Format=DXGI_FORMAT_R32_TYPELESS;
            texture.SampleDesc.Count=1; texture.Usage=D3D11_USAGE_DEFAULT; texture.BindFlags=D3D11_BIND_DEPTH_STENCIL|D3D11_BIND_SHADER_RESOURCE;
            D3D11_DEPTH_STENCIL_VIEW_DESC view{};
            view.Format=DXGI_FORMAT_D32_FLOAT; view.ViewDimension=D3D11_DSV_DIMENSION_TEXTURE2D;
            D3D11_SHADER_RESOURCE_VIEW_DESC read{};
            read.Format=DXGI_FORMAT_R32_FLOAT; read.ViewDimension=D3D11_SRV_DIMENSION_TEXTURE2D; read.Texture2D.MostDetailedMip=0; read.Texture2D.MipLevels=1;
            bool ok=SUCCEEDED(s.device->CreateTexture2D(&texture,nullptr,&s.boxDepth)) && SUCCEEDED(s.device->CreateDepthStencilView(s.boxDepth,&view,&s.boxDepthView)) &&
                SUCCEEDED(s.device->CreateShaderResourceView(s.boxDepth,&read,&s.boxDepthRead));
            texture.MipLevels=levels; texture.Format=DXGI_FORMAT_R32_FLOAT; texture.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
            ok=ok && SUCCEEDED(s.device->CreateTexture2D(&texture,nullptr,&s.cliffs));
            read.Texture2D.MipLevels=levels;
            ok=ok && SUCCEEDED(s.device->CreateShaderResourceView(s.cliffs,&read,&s.cliffChain));
            s.cliffRead.assign(levels,nullptr); s.cliffWrite.assign(levels,nullptr);
            D3D11_UNORDERED_ACCESS_VIEW_DESC write{};
            write.Format=DXGI_FORMAT_R32_FLOAT; write.ViewDimension=D3D11_UAV_DIMENSION_TEXTURE2D;
            read.Texture2D.MipLevels=1;
            for(uint32_t level=0;ok && level<levels;++level) {
                read.Texture2D.MostDetailedMip=level; write.Texture2D.MipSlice=level;
                ok=SUCCEEDED(s.device->CreateShaderResourceView(s.cliffs,&read,&s.cliffRead[level])) &&
                   SUCCEEDED(s.device->CreateUnorderedAccessView(s.cliffs,&write,&s.cliffWrite[level]));
            }
            texture.Width=std::max(1u,width/2); texture.Height=std::max(1u,height/2); texture.MipLevels=1;
            read.Texture2D.MostDetailedMip=0; write.Texture2D.MipSlice=0;
            ok=ok && SUCCEEDED(s.device->CreateTexture2D(&texture,nullptr,&s.exposed)) && SUCCEEDED(s.device->CreateShaderResourceView(s.exposed,&read,&s.exposedRead)) &&
               SUCCEEDED(s.device->CreateUnorderedAccessView(s.exposed,&write,&s.exposedWrite));
            if(!ok) {
                DropBoxTarget();
                logger::warn("Box occlusion: no {}x{} depth target",width,height);
                return false;
            }
            s.boxDepthWidth=width; s.boxDepthHeight=height;
            return true;
        }

        // Motion erosion's compute passes (Boxes::kErodeShader edges, down and expose) on the filled target, with the engine's
        // compute bindings put back as found.
        void ErodeCliffs(ID3D11DeviceContext* context,const Boxes::ErodeParams& erode) {
            auto& s=S();
            ID3D11ComputeShader* oldShader{}; ID3D11ShaderResourceView* oldView{}; ID3D11UnorderedAccessView* oldTarget{}; ID3D11Buffer* oldParams{};
            context->CSGetShader(&oldShader,nullptr,nullptr);
            context->CSGetShaderResources(0,1,&oldView);
            context->CSGetUnorderedAccessViews(0,1,&oldTarget);
            context->CSGetConstantBuffers(0,1,&oldParams);
            const UINT keep=~0u;
            ID3D11ShaderResourceView* const noView=nullptr;
            ID3D11UnorderedAccessView* const noTarget=nullptr;
            const auto pass=[&](ID3D11ComputeShader* shader,ID3D11ShaderResourceView* input,ID3D11UnorderedAccessView* output,uint32_t width,uint32_t height) {
                context->CSSetShader(shader,nullptr,0);
                context->CSSetShaderResources(0,1,&input);
                context->CSSetUnorderedAccessViews(0,1,&output,&keep);
                context->Dispatch((width+7)/8,(height+7)/8,1);
                context->CSSetUnorderedAccessViews(0,1,&noTarget,&keep);
                context->CSSetShaderResources(0,1,&noView);
            };
            context->CSSetConstantBuffers(0,1,&s.erodeParams);
            const uint32_t width=erode.sizes[0],height=erode.sizes[1],levels=erode.sizes[2];
            pass(s.erodeEdges,s.boxDepthRead,s.cliffWrite[0],width,height);
            for(uint32_t level=1;level<levels;++level)
                pass(s.erodeDown,s.cliffRead[level-1],s.cliffWrite[level],std::max(1u,width>>level),std::max(1u,height>>level));
            pass(s.erodeExpose,s.cliffChain,s.exposedWrite,std::max(1u,width/2),std::max(1u,height/2));
            context->CSSetShader(oldShader,nullptr,0);
            context->CSSetShaderResources(0,1,&oldView);
            context->CSSetUnorderedAccessViews(0,1,&oldTarget,&keep);
            context->CSSetConstantBuffers(0,1,&oldParams);
            if(oldShader) oldShader->Release();
            if(oldView) oldView->Release();
            if(oldTarget) oldTarget->Release();
            if(oldParams) oldParams->Release();
        }

        // Main draw, right after Main::RenderDepth (after the Hi-Z capture): this frame's boxes against this frame's
        // prepass depth eroded for the eye's motion (Boxes::kErodeShader), with the main camera as it culled. Moving references
        // and depths within the near exemption do not occlude (as in the Hi-Z grid). The results go to the readback slot of
        // this frame, with the shift they may be used for.
        void BoxCaptureBody();
        void BoxCapture() {
            auto& s=S();
            if(!s.settings.diagnostics) {
                BoxCaptureBody();
                return MeshReadsStep(s.settings.boxes && s.enabled.load(relaxed) && s.eyes==1);
            }
            const auto begin=Clock::now();
            const uint64_t passes=s.boxPasses.load(relaxed);
            BoxCaptureBody();
            MeshReadsStep(s.settings.boxes && s.enabled.load(relaxed) && s.eyes==1);
            if(s.boxPasses.load(relaxed)!=passes) {
                s.boxCpuMicros+=std::chrono::duration<double,std::micro>(Clock::now()-begin).count();
                ++s.boxCpuPasses;
            }
        }
        void BoxCaptureBody() {
            auto& s=S();
            const bool registered=s.boxRegistering.exchange(false,relaxed);
            const uint32_t count=std::min(s.boxCount.load(relaxed),Boxes::kMaxBoxes);
            if(!registered || !count || !s.settings.boxes || !s.enabled.load(relaxed) || s.eyes!=1) return;
            if(s.convention!=Convention::Standard && s.convention!=Convention::Reversed) return;
            auto* renderer=RE::BSGraphics::Renderer::GetSingleton();
            if(!renderer || !InitGpu(renderer) || !InitBoxGpu()) return;
            auto* camera=RE::Main::WorldRootCamera();
            HiZ::Mat4 clip;
            HiZ::Mat4d exact;
            if(!camera || !Projection(camera,clip,&exact)) return;
            auto* source=DepthView(renderer->GetDepthStencilData().depthStencils[RE::RENDER_TARGETS_DEPTHSTENCIL::kPOST_ZPREPASS_COPY]);
            Reduce::Area area;
            if(!source || !CurrentArea(area)) return;
            const uint32_t factor=std::clamp(s.settings.boxDownsample,1u,8u);
            const uint32_t width=Boxes::TargetSide(area.width,factor),height=Boxes::TargetSide(area.height,factor);
            if(!BoxTarget(width,height)) return;
            auto fill=Reduce::MakeParams(area,width,height,s.depth.Ndc(s.settings.nearExempt),s.reversed);
            if(s.settings.excludeMoving && !ExcludeMoving(fill,camera,&clip,false,false)) return;
            // The shift BoxPrepare trusts the result with is the one the target is eroded for; the nearest depth a box reaches
            // is RegisterBox's (the near plane or the near exemption). A target too small for a level 1 gives no result.
            const auto& frustum=camera->GetRuntimeData2().viewFrustum;
            const auto erode=Boxes::MakeErodeParams(width,height,s.depth.a,s.depth.b,frustum.fLeft,frustum.fRight,frustum.fTop,frustum.fBottom,
                s.settings.boxMaxShift,std::max(frustum.fNear,s.settings.nearExempt));
            const bool eroding=erode.reach[1]>0.f;
            if(eroding && erode.sizes[2]<2) return;
            auto* context=s.context;
            const uint32_t index=static_cast<uint32_t>(s.boxPasses.load(relaxed)%kSlots);
            auto& slot=s.boxSlots[index];
            D3D11_MAPPED_SUBRESOURCE mapped{};
            if(FAILED(context->Map(s.boxBuffer,0,D3D11_MAP_WRITE_DISCARD,0,&mapped))) return;
            // Keys and boxes in one pass, each slot's frame read first (RegisterBox writes it last): a slot without this frame's
            // stamp (taken but not yet written, or left from an earlier frame) or with a zero axis is dropped and its box uploaded
            // empty, so registrations still running cannot pair one geometry's key with another's box or with an unwritten one.
            // From here on nothing returns before the slot is pending again.
            slot.pending=false;
            const uint32_t epoch=s.boxEpoch.load(relaxed);
            auto* upload=static_cast<Boxes::Box*>(mapped.pData);
            slot.keys.resize(count);
            for(uint32_t i=0;i<count;++i) {
                const bool current=std::atomic_ref(s.boxKeys[i].epoch).load(std::memory_order_acquire)==epoch;
                auto key=s.boxKeys[i];
                auto box=s.boxes[i];
                if(!current || !key.object || !Boxes::Solid(box)) { key=State::BoxKey{}; box=Boxes::Box{}; }
                upload[i]=box; slot.keys[i]=key;
            }
            context->Unmap(s.boxBuffer,0);
            // The boxes are relative to the culling camera's eye (RegisterBox), and so is the matrix they are drawn with.
            Boxes::BoxParams params{};
            const auto toClip=HiZ::Relative(exact,s.boxEye);
            std::memcpy(params.toClip,toClip.m,sizeof(params.toClip));
            params.count[0]=count;
            context->UpdateSubresource(s.fillParams,0,nullptr,&fill,0,0);
            context->UpdateSubresource(s.boxParams,0,nullptr,&params,0,0);
            if(eroding) context->UpdateSubresource(s.erodeParams,0,nullptr,&erode,0,0);
            if(s.settings.diagnostics) s.boxTimer.Begin(s.device,context);
            {
                PipelineGuard guard(context);
                const D3D11_VIEWPORT viewport{0.f,0.f,float(width),float(height),0.f,1.f};
                ID3D11ShaderResourceView* const noView=nullptr;
                ID3D11UnorderedAccessView* const noAccess=nullptr;
                context->IASetInputLayout(nullptr); context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                context->HSSetShader(nullptr,nullptr,0); context->DSSetShader(nullptr,nullptr,0); context->GSSetShader(nullptr,nullptr,0);
                context->RSSetState(s.boxRaster); context->RSSetViewports(1,&viewport);
                context->OMSetBlendState(nullptr,nullptr,0xFFFFFFFFu);
                // The depth target from the depth copy.
                context->OMSetRenderTargetsAndUnorderedAccessViews(0,nullptr,s.boxDepthView,0,0,nullptr,nullptr);
                context->OMSetDepthStencilState(s.fillState,0);
                context->VSSetShader(s.fillVertex,nullptr,0);
                context->PSSetShader(s.fillPixel,nullptr,0); context->PSSetShaderResources(0,1,&source); context->PSSetConstantBuffers(0,1,&s.fillParams);
                context->Draw(3,0);
                context->PSSetShaderResources(0,1,&noView);
                if(eroding) {
                    // Its cliffs on the compute pipeline (the target unbound to be read), then the exposed texels written far.
                    context->OMSetRenderTargetsAndUnorderedAccessViews(0,nullptr,nullptr,0,0,nullptr,nullptr);
                    ErodeCliffs(context,erode);
                    context->OMSetRenderTargetsAndUnorderedAccessViews(0,nullptr,s.boxDepthView,0,0,nullptr,nullptr);
                    context->PSSetShader(s.erodeApply,nullptr,0); context->PSSetShaderResources(0,1,&s.exposedRead);
                    context->Draw(3,0);
                    context->PSSetShaderResources(0,1,&noView);
                }
                // The boxes against it.
                const UINT zeros[4]{};
                context->ClearUnorderedAccessViewUint(s.boxResultsView,zeros);
                context->OMSetRenderTargetsAndUnorderedAccessViews(0,nullptr,s.boxDepthView,0,1,&s.boxResultsView,nullptr);
                context->OMSetDepthStencilState(s.boxState,0);
                context->VSSetShader(s.boxVertex,nullptr,0); context->VSSetShaderResources(0,1,&s.boxView); context->VSSetConstantBuffers(0,1,&s.boxParams);
                context->PSSetShader(s.boxPixel,nullptr,0);
                context->DrawInstanced(36,count,0,0);
                context->VSSetShaderResources(0,1,&noView);
                context->OMSetRenderTargetsAndUnorderedAccessViews(0,nullptr,nullptr,0,1,&noAccess,nullptr);
            }
            if(s.settings.diagnostics) s.boxTimer.End(context);
            const D3D11_BOX range{0,0,0,count*4u,1,1};
            context->CopySubresourceRegion(slot.staging,0,0,0,0,s.boxResults,0,&range);
            slot.frame=s.boxFrame;
            slot.eye={camera->world.translate.x,camera->world.translate.y,camera->world.translate.z};
            slot.frustum=frustum;
            slot.shift=eroding ? erode.reach[1] : 0.f;
            slot.pending=true;
            s.boxPasses.fetch_add(1,relaxed);
        }

        // From Prepare, before this frame's culling: last frame's counts, this frame's culling camera, the newest finished
        // readback turned into a table of hidden geometry, and that table published while it is usable for this camera.
        void BoxPrepare(const RE::NiCamera* camera) {
            auto& s=S();
            ++s.boxFrame;
            s.lastBoxRegistered.store(Take(kBoxRegistered),relaxed); s.lastBoxCulled.store(Take(kBoxCulled),relaxed);
            s.lastBoxNoModel.store(Take(kBoxNoModel),relaxed); s.lastBoxEdge.store(Take(kBoxEdge),relaxed);
            s.lastBoxType.store(Take(kBoxType),relaxed); s.lastBoxWaiting.store(Take(kBoxWaiting),relaxed); s.lastBoxInvalid.store(Take(kBoxInvalid),relaxed);
            s.boxRegistering.store(false,relaxed);
            s.boxCount.store(0,relaxed);
            s.boxEpoch.fetch_add(1,relaxed);
            s.boxVertexBudget.store(kBoxVertexBudget,relaxed);
            if(s.settings.diagnostics && s.boxCpuPasses && s.boxFrame-s.boxLogFrame>=kBoxLogFrames) {
                double ms=0.0; uint32_t spans=0;
                s.boxTimer.Take(ms,spans);
                logger::info("Box occlusion cost: render thread {:.0f} us per pass ({} passes), GPU {} per pass",s.boxCpuMicros/s.boxCpuPasses,s.boxCpuPasses,
                    spans ? std::format("{:.0f} us ({} timed)",ms*1000.0/spans,spans) : std::string("not timed"));
                s.boxCpuMicros=0.0; s.boxCpuPasses=0; s.boxLogFrame=s.boxFrame;
            }
            const auto unpublish=[&] { s.boxActive.store(nullptr,std::memory_order_release); s.boxTableHidden.store(0,relaxed); };
            if(!s.settings.boxes || !s.enabled.load(relaxed) || s.eyes!=1 || !camera || s.boxes.size()!=Boxes::kMaxBoxes) return unpublish();
            const auto& frustum=camera->GetRuntimeData2().viewFrustum;
            HiZ::Mat4d cameraToClip,inverse;
            if(!CameraToClip(frustum,cameraToClip) || !HiZ::Invert(CameraWorldExact(camera),inverse)) return unpublish();
            const auto& eye=camera->world.translate;
            s.boxEye={eye.x,eye.y,eye.z};
            // Built in double, for points relative to the eye (see RegisterBox).
            s.boxClip=HiZ::Relative(HiZ::Multiply(cameraToClip,inverse),s.boxEye);
            s.boxNear=frustum.fNear;
            // The depth target's texel: from the last pass's target, else from the screen and the dynamic-resolution
            // ratios (the rendered area; never larger than it, so texels are never underestimated).
            uint32_t targetWidth=s.boxDepthWidth,targetHeight=s.boxDepthHeight;
            if(!targetWidth || !targetHeight) {
                const auto* state=RE::BSGraphics::State::GetSingleton();
                const uint32_t factor=std::clamp(s.settings.boxDownsample,1u,8u);
                if(state && state->screenWidth && state->screenHeight) {
                    const auto& data=state->GetRuntimeData();
                    const auto side=[&](uint32_t screen,float ratio) {
                        return Boxes::TargetSide(static_cast<uint32_t>(float(screen)*(std::isfinite(ratio) ? std::clamp(ratio,0.125f,1.f) : 1.f)),factor);
                    };
                    targetWidth=side(state->screenWidth,data.dynamicResolutionWidthRatio);
                    targetHeight=side(state->screenHeight,data.dynamicResolutionHeightRatio);
                }
            }
            s.boxTexel=Boxes::TexelSpan(frustum.fRight-frustum.fLeft,frustum.fTop-frustum.fBottom,targetWidth,targetHeight);
            // The newest finished readback; older ones still pending are dropped with it.
            int newest=-1;
            for(uint32_t k=0;k<kSlots;++k) {
                auto& slot=s.boxSlots[k];
                if(slot.pending && s.boxFrame-slot.frame>kMaxBoxAge) slot.pending=false;
            }
            while(true) {
                int pick=-1;
                for(uint32_t k=0;k<kSlots;++k)
                    if(s.boxSlots[k].pending && (pick<0 || s.boxSlots[k].frame>s.boxSlots[pick].frame) && (newest<0 || s.boxSlots[k].frame<s.boxSlots[newest].frame))
                        pick=int(k);
                if(pick<0) break;
                auto& slot=s.boxSlots[pick];
                D3D11_MAPPED_SUBRESOURCE mapped{};
                const HRESULT hr=s.context->Map(slot.staging,0,D3D11_MAP_READ,D3D11_MAP_FLAG_DO_NOT_WAIT,&mapped);
                if(hr==DXGI_ERROR_WAS_STILL_DRAWING) { newest=pick; continue; }  // try an older one
                slot.pending=false;
                if(FAILED(hr)) { newest=pick; continue; }
                const auto* visible=static_cast<const uint32_t*>(mapped.pData);
                const int next=s.boxCurrent<0 ? 0 : s.boxCurrent^1;
                auto& table=s.boxTables[next];
                uint32_t hidden=0;
                for(size_t i=0;i<slot.keys.size();++i) if(!visible[i] && slot.keys[i].object) ++hidden;
                uint32_t size=64;
                while(size<hidden*2) size<<=1;
                table.entries.assign(size,State::BoxKey{});
                table.mask=size-1;
                for(size_t i=0;i<slot.keys.size();++i) {
                    if(visible[i] || !slot.keys[i].object) continue;
                    uint32_t at=BoxHash(slot.keys[i].object)&table.mask;
                    while(table.entries[at].object) at=(at+1)&table.mask;
                    table.entries[at]=slot.keys[i];
                }
                s.context->Unmap(slot.staging,0);
                table.hidden=hidden; table.frame=slot.frame; table.eye=slot.eye; table.frustum=slot.frustum; table.shift=slot.shift; table.valid=true;
                s.boxCurrent=next;
                s.boxReadbacks.fetch_add(1,relaxed);
                for(auto& other:s.boxSlots) if(other.pending && other.frame<slot.frame) other.pending=false;
                break;
            }
            const State::BoxTable* use=nullptr;
            if(s.boxCurrent>=0) {
                const auto& table=s.boxTables[s.boxCurrent];
                const float dx=eye.x-table.eye[0],dy=eye.y-table.eye[1],dz=eye.z-table.eye[2];
                // The shift its depth target was eroded for (Boxes::kErodeShader): beyond it, a moved eye may see past an edge.
                const float shift=table.shift;
                if(!table.valid || s.boxFrame-table.frame>kMaxBoxAge || !SameFrustum(frustum,table.frustum)) s.boxStale.fetch_add(1,relaxed);
                else if(dx*dx+dy*dy+dz*dz>shift*shift) s.boxShifted.fetch_add(1,relaxed);
                else use=&table;
            }
            s.boxActive.store(use,std::memory_order_release);
            s.boxTableHidden.store(use ? use->hidden : 0,relaxed);
            s.boxRegistering.store(true,relaxed);
        }

        bool InitAuditGpu() {
            auto& s=S();
            if(s.auditInitTried) return s.auditInitOk;
            s.auditInitTried=true;
            const auto fail=[](const char* what,HRESULT hr=S_OK) { logger::error("Hi-Z audit disabled: {} (0x{:08X})",what,static_cast<uint32_t>(hr)); return false; };
            const HMODULE compiler=LoadLibraryW(L"d3dcompiler_47.dll");
            const auto compile=compiler ? reinterpret_cast<pD3DCompile>(GetProcAddress(compiler,"D3DCompile")) : nullptr;
            if(!compile) return fail("d3dcompiler_47.dll unavailable");
            const auto build=[&](const char* entry,const char* target,ID3DBlob*& code) {
                ID3DBlob* errors{};
                const HRESULT hr=compile(Audit::kShader,sizeof(Audit::kShader)-1,"PrecombinesAudit",nullptr,nullptr,entry,target,D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&code,&errors);
                if(errors) {
                    if(FAILED(hr)) logger::error("Hi-Z audit shader {}: {}",entry,static_cast<const char*>(errors->GetBufferPointer()));
                    errors->Release();
                }
                return SUCCEEDED(hr);
            };
            ID3DBlob* code{};
            HRESULT hr=S_OK;
            if(!build("vs","vs_5_0",code)) return fail("vertex shader");
            hr=s.device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&s.auditVertex); code->Release();
            if(FAILED(hr)) return fail("CreateVertexShader",hr);
            if(!build("ps","ps_5_0",code)) return fail("pixel shader");
            hr=s.device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&s.auditPixel); code->Release();
            if(FAILED(hr)) return fail("CreatePixelShader",hr);
            D3D11_BUFFER_DESC buffer{};
            buffer.ByteWidth=sizeof(Audit::Params); buffer.Usage=D3D11_USAGE_DEFAULT; buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            if(FAILED(hr=s.device->CreateBuffer(&buffer,nullptr,&s.auditParams))) return fail("CreateBuffer (parameters)",hr);
            buffer=D3D11_BUFFER_DESC{};
            buffer.ByteWidth=16*Audit::kMaxSpheres; buffer.Usage=D3D11_USAGE_DYNAMIC; buffer.BindFlags=D3D11_BIND_SHADER_RESOURCE;
            buffer.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE; buffer.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; buffer.StructureByteStride=16;
            if(FAILED(hr=s.device->CreateBuffer(&buffer,nullptr,&s.auditSpheres))) return fail("CreateBuffer (spheres)",hr);
            D3D11_SHADER_RESOURCE_VIEW_DESC view{};
            view.Format=DXGI_FORMAT_UNKNOWN; view.ViewDimension=D3D11_SRV_DIMENSION_BUFFER; view.Buffer.NumElements=Audit::kMaxSpheres;
            if(FAILED(hr=s.device->CreateShaderResourceView(s.auditSpheres,&view,&s.auditSphereView))) return fail("CreateShaderResourceView (spheres)",hr);
            buffer=D3D11_BUFFER_DESC{};
            buffer.ByteWidth=4*Audit::kMaxSpheres; buffer.Usage=D3D11_USAGE_DEFAULT; buffer.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
            buffer.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
            if(FAILED(hr=s.device->CreateBuffer(&buffer,nullptr,&s.auditCounts))) return fail("CreateBuffer (counts)",hr);
            D3D11_UNORDERED_ACCESS_VIEW_DESC access{};
            access.Format=DXGI_FORMAT_R32_TYPELESS; access.ViewDimension=D3D11_UAV_DIMENSION_BUFFER; access.Buffer.NumElements=Audit::kMaxSpheres;
            access.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;
            if(FAILED(hr=s.device->CreateUnorderedAccessView(s.auditCounts,&access,&s.auditCountView))) return fail("CreateUnorderedAccessView (counts)",hr);
            buffer=D3D11_BUFFER_DESC{};
            buffer.ByteWidth=4*Audit::kMaxSpheres; buffer.Usage=D3D11_USAGE_STAGING; buffer.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
            for(auto& slot:s.auditSlots) if(FAILED(hr=s.device->CreateBuffer(&buffer,nullptr,&slot.staging))) return fail("CreateBuffer (readback)",hr);
            D3D11_RASTERIZER_DESC raster{};
            raster.FillMode=D3D11_FILL_SOLID; raster.CullMode=D3D11_CULL_NONE; raster.DepthClipEnable=TRUE;
            if(FAILED(hr=s.device->CreateRasterizerState(&raster,&s.auditRaster))) return fail("CreateRasterizerState",hr);
            s.auditInitOk=true;
            logger::info("Hi-Z audit ready: every {}th frame up to {} culls checked against that frame's depth ({} pixels in front by {}% and a unit make a false cull)",
                std::max(1u,s.settings.auditEvery),Audit::kMaxSpheres,Audit::kMinPixels,Audit::kMargin*100.f);
            return true;
        }

        // After RenderDepth on an audit frame: the frame's culls drawn against its own depth copy (Occlusion::Audit); their
        // counts are read back a frame or two later (AuditRead). A full ring skips the frame.
        void AuditCapture() {
            auto& s=S();
            if(!s.auditing.exchange(false,relaxed)) return;
            const uint32_t count=std::min(s.auditCount.load(relaxed),Audit::kMaxSpheres);
            if(!count || !s.auditEntries || s.eyes!=1) return;
            if(s.convention!=Convention::Standard && s.convention!=Convention::Reversed) return;
            auto* renderer=RE::BSGraphics::Renderer::GetSingleton();
            if(!renderer || !InitGpu(renderer) || !InitAuditGpu()) return;
            auto& slot=s.auditSlots[s.auditNext%kSlots];
            if(slot.pending) return;
            auto* camera=RE::Main::WorldRootCamera();
            HiZ::Mat4 clip;
            HiZ::Mat4d exact;
            if(!camera || !Projection(camera,clip,&exact)) return;
            auto* source=DepthView(renderer->GetDepthStencilData().depthStencils[RE::RENDER_TARGETS_DEPTHSTENCIL::kPOST_ZPREPASS_COPY]);
            Reduce::Area area;
            if(!source || !CurrentArea(area)) return;
            // Relative to the culling camera's eye, as the spheres are (a world-space float projection is off far from the origin).
            const auto toClip=HiZ::Relative(exact,s.auditEye);
            HiZ::Mat4 fromClip;
            if(!HiZ::Invert(toClip,fromClip)) return;
            Audit::Params params{};
            std::memcpy(params.toClip,toClip.m,sizeof(params.toClip)); std::memcpy(params.fromClip,fromClip.m,sizeof(params.fromClip));
            params.area[0]=float(area.width); params.area[1]=float(area.height); params.area[2]=1.f/float(area.width); params.area[3]=1.f/float(area.height);
            params.model[0]=s.depth.a; params.model[1]=s.depth.b; params.model[2]=s.reversed ? 1.f : 0.f; params.model[3]=Audit::kMargin;
            auto* context=s.context;
            D3D11_MAPPED_SUBRESOURCE mapped{};
            if(FAILED(context->Map(s.auditSpheres,0,D3D11_MAP_WRITE_DISCARD,0,&mapped))) return;
            slot.entries.assign(s.auditEntries.get(),s.auditEntries.get()+count);
            slot.eye=s.auditEye;
            auto* spheres=static_cast<float*>(mapped.pData);
            for(uint32_t i=0;i<count;++i) {
                const auto& entry=slot.entries[i];
                for(int k=0;k<3;++k) spheres[4*i+k]=entry.center[k]-s.auditEye[k];
                spheres[4*i+3]=entry.radius;
            }
            context->Unmap(s.auditSpheres,0);
            context->UpdateSubresource(s.auditParams,0,nullptr,&params,0,0);
            {
                PipelineGuard guard(context);
                const D3D11_VIEWPORT viewport{0.f,0.f,float(area.width),float(area.height),0.f,1.f};
                ID3D11ShaderResourceView* const noView=nullptr;
                ID3D11UnorderedAccessView* const noAccess=nullptr;
                const UINT zeros[4]{};
                context->IASetInputLayout(nullptr); context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                context->HSSetShader(nullptr,nullptr,0); context->DSSetShader(nullptr,nullptr,0); context->GSSetShader(nullptr,nullptr,0);
                context->RSSetState(s.auditRaster); context->RSSetViewports(1,&viewport);
                context->OMSetBlendState(nullptr,nullptr,0xFFFFFFFFu); context->OMSetDepthStencilState(nullptr,0);
                context->ClearUnorderedAccessViewUint(s.auditCountView,zeros);
                // No render target and no depth target: the pixel shader only counts.
                context->OMSetRenderTargetsAndUnorderedAccessViews(0,nullptr,nullptr,0,1,&s.auditCountView,nullptr);
                context->VSSetShader(s.auditVertex,nullptr,0); context->VSSetShaderResources(0,1,&s.auditSphereView); context->VSSetConstantBuffers(0,1,&s.auditParams);
                context->PSSetShader(s.auditPixel,nullptr,0); context->PSSetShaderResources(0,1,&source); context->PSSetConstantBuffers(0,1,&s.auditParams);
                context->DrawInstanced(36,count,0,0);
                context->VSSetShaderResources(0,1,&noView); context->PSSetShaderResources(0,1,&noView);
                context->OMSetRenderTargetsAndUnorderedAccessViews(0,nullptr,nullptr,0,1,&noAccess,nullptr);
            }
            const D3D11_BOX range{0,0,0,count*4u,1,1};
            context->CopySubresourceRegion(slot.staging,0,0,0,0,s.auditCounts,0,&range);
            slot.pending=true;
            ++s.auditNext; ++s.auditFrames;
        }

        // From Prepare: the finished audit frames, oldest first. Every 30 s one line sums the false culls up by size ("Hi-Z
        // audit:") and the next names the eight largest since the last summary ("Hi-Z audit: largest").
        void AuditRead() {
            auto& s=S();
            static constexpr const char* kNames[Audit::kCullers]={"Hi-Z (geometry)","Hi-Z (reference)","the box test"};
            for(uint32_t k=0;k<kSlots;++k) {
                auto& slot=s.auditSlots[(s.auditNext+k)%kSlots];
                if(!slot.pending) continue;
                D3D11_MAPPED_SUBRESOURCE mapped{};
                const HRESULT hr=s.context->Map(slot.staging,0,D3D11_MAP_READ,D3D11_MAP_FLAG_DO_NOT_WAIT,&mapped);
                if(hr==DXGI_ERROR_WAS_STILL_DRAWING) break;
                slot.pending=false;
                if(FAILED(hr)) continue;
                const auto* counts=static_cast<const uint32_t*>(mapped.pData);
                for(size_t i=0;i<slot.entries.size();++i) {
                    const auto& entry=slot.entries[i];
                    const uint32_t by=std::min<uint32_t>(entry.culler,Audit::kCullers-1);
                    ++s.auditCulled[by]; ++s.auditChecked;
                    if(counts[i]<Audit::kMinPixels) continue;
                    ++s.auditFalseBy[by]; ++s.auditFalse;
                    ++s.auditBuckets[counts[i]<4 ? 0 : counts[i]<16 ? 1 : counts[i]<64 ? 2 : 3];
                    const float dx=entry.center[0]-slot.eye[0],dy=entry.center[1]-slot.eye[1],dz=entry.center[2]-slot.eye[2];
                    auto& worst=s.auditWorst;
                    if(worst.size()<8 || counts[i]>worst.back().pixels) {
                        if(worst.size()==8) worst.pop_back();
                        const auto at=std::find_if(worst.begin(),worst.end(),[&](const auto& w) { return w.pixels<counts[i]; });
                        worst.insert(at,State::AuditWorst{counts[i],std::sqrt(dx*dx+dy*dy+dz*dz),entry});
                    }
                }
                s.context->Unmap(slot.staging,0);
            }
            const auto now=Clock::now();
            if(s.auditFrames && now-s.auditLogged>=std::chrono::seconds(30)) {
                s.auditLogged=now;
                logger::info("Hi-Z audit: {} frames checked, {} culls (Hi-Z geometry {}, references {}, box test {}), {} false (Hi-Z geometry {}, references {}, box test {}); "
                             "by visible pixels 2-3: {}, 4-15: {}, 16-63: {}, 64+: {}",
                    s.auditFrames,s.auditChecked,s.auditCulled[0],s.auditCulled[1],s.auditCulled[2],s.auditFalse,s.auditFalseBy[0],s.auditFalseBy[1],s.auditFalseBy[2],
                    s.auditBuckets[0],s.auditBuckets[1],s.auditBuckets[2],s.auditBuckets[3]);
                std::string largest;
                for(const auto& w:s.auditWorst)
                    largest+=std::format("{}\"{}\" {:08X} {} px {:.0f} units r {:.0f} at ({:.0f},{:.0f},{:.0f}) by {}",largest.empty() ? "" : "; ",w.entry.name,w.entry.form,w.pixels,
                        w.distance,w.entry.radius,w.entry.center[0],w.entry.center[1],w.entry.center[2],kNames[std::min<uint32_t>(w.entry.culler,Audit::kCullers-1)]);
                if(!largest.empty()) logger::info("Hi-Z audit: largest false culls since the last line: {}",largest);
                s.auditWorst.clear();
            }
        }

        // No parent walk (NiAVObject::GetUserData climbs to the root for every geometry): skinned geometry is excluded by
        // the caller, and actors at their fade-node roots, whose own userData names the reference.
        bool Eligible(const RE::NiAVObject& object,bool root) {
            if(object.GetFlags().all(RE::NiAVObject::Flag::kAlwaysDraw) || !(object.worldBound.radius>0.f)) return false;
            if(!root) return true;
            const auto* reference=REL::RelocateMember<RE::TESObjectREFR*>(&object,0x0F8,0x110);
            return !reference || reference->GetFormType()!=RE::FormType::ActorCharacter;
        }

        // This frame's pyramid, waiting for the helper until the frame's deadline when it is still being built: at a few
        // hundred frames per second the engine finished culling before the pyramid was published and Hi-Z tested nothing
        // (Ustengrav02 brazier, 131,584 pyramids built, none used).
        const Views* FramePyramid() {
            auto& s=S();
            const auto* pyramid=s.active.load(std::memory_order_acquire);
            if(pyramid || !s.busy.load(std::memory_order_acquire)) return pyramid;
            const int64_t deadline=s.waitDeadline.load(relaxed);
            if(Clock::now().time_since_epoch().count()>=deadline) return nullptr;
            s.waits.fetch_add(1,relaxed);
            for(uint32_t spin=0;;++spin) {
                if((pyramid=s.active.load(std::memory_order_acquire)) || !s.busy.load(std::memory_order_acquire)) return pyramid;
                if((spin&63)==0 && Clock::now().time_since_epoch().count()>=deadline) { s.waitTimeouts.fetch_add(1,relaxed); return nullptr; }
                _mm_pause();
            }
        }

        // A Hi-Z test's outcome. Untested: another camera, no pyramid for the frame, skipped by the visible-object table, not
        // eligible. The flicker log (NoteFlicker) needs the difference between that and Visible.
        enum class Verdict : uint8_t { Untested, Visible, Hidden };
        // The visible-object table is read before the eligibility checks: it only holds objects that were eligible and
        // found visible, and a skip only ever draws, so its lookup (a slot in 128 KiB, in the L2 cache) spares the skipped
        // objects (about two thirds at interval 8) the eligibility read of their reference's form type, a likely cache miss.
        Verdict HiddenTest(const RE::NiAVObject& object,const RE::NiCullingProcess& process,Count tested,Count culled,bool root) {
            auto& s=S();
            if(process.camera!=s.camera.load(relaxed)) return Verdict::Untested;
            const auto* pyramid=FramePyramid();
            if(!pyramid) return Verdict::Untested;
            const uint32_t retest=s.settings.visibleRetest;
            const uint32_t frame=s.cullFrame.load(relaxed);
            if(retest>1 && s.visible.Skip(&object,frame)) { Bump(kSkipped); return Verdict::Untested; }
            if(!Eligible(object,root)) return Verdict::Untested;
            Bump(tested);
            const auto& bound=object.worldBound;
            const float center[3]={bound.center.x,bound.center.y,bound.center.z};
            if(!pyramid->Occluded(center,bound.radius)) { s.visible.Visible(&object,frame,retest); return Verdict::Visible; }
            Bump(culled);
            return Verdict::Hidden;
        }
        // "hiz_ab" state EligibleFirst: the same test with the eligibility checks before the visible-object table, as in
        // runtime 5dd7002c and before, to measure what the order above saves.
        Verdict HiddenTestEligibleFirst(const RE::NiAVObject& object,const RE::NiCullingProcess& process,Count tested,Count culled,bool root) {
            auto& s=S();
            if(process.camera!=s.camera.load(relaxed) || !Eligible(object,root)) return Verdict::Untested;
            const auto* pyramid=FramePyramid();
            if(!pyramid) return Verdict::Untested;
            const uint32_t retest=s.settings.visibleRetest;
            const uint32_t frame=s.cullFrame.load(relaxed);
            if(retest>1 && s.visible.Skip(&object,frame)) { Bump(kSkipped); return Verdict::Untested; }
            Bump(tested);
            const auto& bound=object.worldBound;
            const float center[3]={bound.center.x,bound.center.y,bound.center.z};
            if(!pyramid->Occluded(center,bound.radius)) { s.visible.Visible(&object,frame,retest); return Verdict::Visible; }
            Bump(culled);
            return Verdict::Hidden;
        }
        // Diagnostics (sampled frames, hiz_diagnostics): the same test timed in stages; the counters are added after the
        // last timestamp, so they are not part of any stage.
        Verdict HiddenTestTimed(const RE::NiAVObject& object,const RE::NiCullingProcess& process,Count tested,Count culled,bool root) {
            auto& s=S();
            uint64_t stage[4]{};
            bool hidden=false,candidate=false,test=false;
            Verdict verdict=Verdict::Untested;
            const uint64_t t0=__rdtsc();
            if(process.camera==s.camera.load(relaxed)) {
                const uint64_t t1=__rdtsc(); stage[0]=t1-t0; candidate=true;
                if(!s.firstTestTsc.load(relaxed)) { uint64_t none=0; s.firstTestTsc.compare_exchange_strong(none,t1,relaxed); }
                const bool building=!s.active.load(std::memory_order_acquire) && s.busy.load(std::memory_order_acquire);
                const auto* pyramid=FramePyramid();
                const uint64_t t2=__rdtsc(); stage[1]=t2-t1;
                if(building) Bump(kUntested);
                if(pyramid) {
                    const uint32_t retest=s.settings.visibleRetest;
                    const uint32_t frame=s.cullFrame.load(relaxed);
                    const bool skip=retest>1 && s.visible.Skip(&object,frame);
                    const uint64_t t3=__rdtsc(); stage[2]=t3-t2;
                    if(skip) Bump(kSkipped);
                    else {
                        test=true;
                        if(Eligible(object,root)) {
                            const auto& bound=object.worldBound;
                            const float center[3]={bound.center.x,bound.center.y,bound.center.z};
                            hidden=pyramid->Occluded(center,bound.radius);
                            if(!hidden) s.visible.Visible(&object,frame,retest);
                            Bump(tested); if(hidden) Bump(culled);
                            verdict=hidden ? Verdict::Hidden : Verdict::Visible;
                        }
                        stage[3]=__rdtsc()-t3;
                    }
                }
            } else stage[0]=__rdtsc()-t0;
            for(int i=0;i<4;++i) if(stage[i]) s.stageTicks[i].fetch_add(stage[i],relaxed);
            s.hizCalls.fetch_add(1,relaxed);
            if(candidate) s.candidateCalls.fetch_add(1,relaxed);
            if(test) s.testCalls.fetch_add(1,relaxed);
            return verdict;
        }
        // "hiz_ab", the state "as before": the original occlusion test on every object, counted into shared counters.
        Verdict HiddenTestBefore(const RE::NiAVObject& object,const RE::NiCullingProcess& process,Count tested,Count culled,bool root) {
            auto& s=S();
            if(process.camera!=s.camera.load(relaxed) || !Eligible(object,root)) return Verdict::Untested;
            const auto* pyramid=FramePyramid();
            if(!pyramid) return Verdict::Untested;
            s.sharedCounts[tested].fetch_add(1,relaxed);
            const auto& bound=object.worldBound;
            const float center[3]={bound.center.x,bound.center.y,bound.center.z};
            if(!pyramid->OccludedReference(center,bound.radius)) return Verdict::Visible;
            s.sharedCounts[culled].fetch_add(1,relaxed);
            return Verdict::Hidden;
        }
        // "hiz_diagnostics": objects Hi-Z keeps hiding and showing again (hidden, then tested and found visible on a later
        // frame, then hidden again, six times within 300 frames), each named once in the log with its reference, base, distance
        // and bound, and with what the frame's pyramid decided for it (Pyramid::Explain: the sphere's nearest view depth and the
        // occluder's). An object whose bound is smaller than what it draws can hide behind its own depth: the next capture
        // lacks it, so it shows, is captured, and hides again (distant scenery flickered with Hi-Z on). Frames
        // on which Hi-Z did not test the object (Prepare's early returns, no pyramid yet with hiz_wait_us 0, the visible-object
        // table, its root hidden whole) are no flicker: counted as shown, they made a steady hide read as flicker;
        // the hides after such frames are counted apart. Tested-visible frames go to a lock-free table (HiZ::ShownFrames),
        // not under the mutex: there are thousands per frame.
        struct Flicker { uint32_t lastHidden=0, windowStart=0, flips=0, gaps=0; bool logged=false; };
        HiZ::ShownFrames& Shown() { static auto* shown=new HiZ::ShownFrames; return *shown; }
        void NoteFlicker(const RE::NiAVObject& object,bool root,bool hidden) {
            auto& s=S();
            const uint32_t frame=s.cullFrame.load(relaxed);
            if(!hidden) return Shown().Shown(&object,frame);
            static std::mutex mutex;
            static std::unordered_map<const void*,Flicker> seen;
            static std::atomic<uint32_t> named{0};
            bool report=false;
            uint32_t flips=0,gaps=0;
            {
                std::lock_guard lock(mutex);
                if(seen.size()>65536) seen.clear();
                auto& entry=seen[&object];
                if(!entry.windowStart || frame-entry.windowStart>300) { entry.windowStart=frame; entry.flips=0; entry.gaps=0; }
                if(entry.lastHidden && frame>entry.lastHidden+1) {
                    const uint32_t shown=Shown().Last(&object);
                    if(shown>entry.lastHidden && shown<frame) ++entry.flips;
                    else ++entry.gaps;
                }
                entry.lastHidden=frame;
                if(entry.flips>=6 && !entry.logged && named.load(relaxed)<64) {
                    entry.logged=true; report=true; flips=entry.flips; gaps=entry.gaps; named.fetch_add(1,relaxed);
                }
            }
            if(!report) return;
            const RE::NiAVObject* node=&object;
            while(node && !const_cast<RE::NiAVObject*>(node)->AsFadeNode()) node=node->parent;
            const auto* reference=node ? REL::RelocateMember<RE::TESObjectREFR*>(node,0x0F8,0x110) : nullptr;
            const auto* base=reference ? reference->GetBaseObject() : nullptr;
            const auto* camera=s.camera.load(relaxed);
            const auto& bound=object.worldBound;
            float distance=0.f;
            if(camera) {
                const auto& eye=camera->world.translate;
                const float dx=bound.center.x-eye.x,dy=bound.center.y-eye.y,dz=bound.center.z-eye.z;
                distance=std::sqrt(dx*dx+dy*dy+dz*dz);
            }
            // What the pyramid that hid it this frame rests on (one per eye on VR).
            std::string explained;
            if(const auto* views=s.active.load(std::memory_order_acquire)) {
                const float center[3]={bound.center.x,bound.center.y,bound.center.z};
                for(uint32_t e=0;e<views->count;++e) {
                    const auto x=views->eye[e].Explain(center,bound.radius);
                    explained+=std::format("{}{}: nearest view depth {:.1f}, occluder view depth {:.1f} (depth {:.7f}, level {}, texels {}-{} x {}-{})",
                        e ? "; " : "",x.why,x.wmin,x.occluderView,x.occluder,x.level,x.tx0,x.tx1,x.ty0,x.ty1);
                }
            }
            const char* name=object.name.c_str();
            const char* nodeName=node ? node->name.c_str() : nullptr;
            const char* model=nullptr;
            if(base) if(const auto* modelForm=base->As<RE::TESModel>()) model=modelForm->GetModel();
            logger::info("Hi-Z flicker: {} \"{}\" (fade node \"{}\") hidden, found visible and hidden again {} times in under 300 frames (and {} times "
                         "hidden again after frames it was not tested); reference {:08X} base {:08X} \"{}\" model \"{}\"; {:.0f} units away, bound "
                         "radius {:.0f}, centre ({:.0f},{:.0f},{:.0f}); {}",
                root ? "reference" : "geometry",name ? name : "",nodeName ? nodeName : "",flips,gaps,reference ? reference->GetFormID() : 0u,
                base ? base->GetFormID() : 0u,base ? base->GetName() : "",model ? model : "",distance,bound.radius,bound.center.x,bound.center.y,bound.center.z,
                explained.empty() ? std::string("no pyramid published") : explained);
        }
        // A dump frame's candidate (see State::DumpEntry): recorded, never hidden.
        uint64_t ParentRoot(const RE::NiAVObject* node) {
            for(;node;node=node->parent) if(const_cast<RE::NiAVObject*>(node)->AsFadeNode()) return reinterpret_cast<uint64_t>(node);
            return 0;
        }
        void Record(const RE::NiAVObject& object,const RE::NiCullingProcess& process,bool root) {
            auto& s=S();
            if(process.camera!=s.camera.load(relaxed) || !Eligible(object,root)) return;
            const uint32_t index=s.dumpCount.fetch_add(1,relaxed);
            if(index>=s.dumpEntries.size()) return;
            auto& entry=s.dumpEntries[index];
            const auto& bound=object.worldBound;
            entry.center[0]=bound.center.x; entry.center[1]=bound.center.y; entry.center[2]=bound.center.z; entry.radius=bound.radius;
            entry.root=root ? 1u : 0u;
            entry.self=reinterpret_cast<uint64_t>(&object);
            entry.parentRoot=ParentRoot(object.parent);
        }
        // An audit frame's cull (Occlusion::Audit), kept with its bound, its reference and its name for AuditCapture. Culling
        // threads; at most Audit::kMaxSpheres a frame.
        void AuditNote(const RE::NiAVObject& object,Audit::Culler culler) {
            auto& s=S();
            const uint32_t index=s.auditCount.fetch_add(1,relaxed);
            if(index>=Audit::kMaxSpheres || !s.auditEntries) return;
            auto& entry=s.auditEntries[index];
            const auto& bound=object.worldBound;
            entry.center[0]=bound.center.x; entry.center[1]=bound.center.y; entry.center[2]=bound.center.z; entry.radius=bound.radius;
            entry.culler=culler;
            const RE::NiAVObject* node=&object;
            while(node && !const_cast<RE::NiAVObject*>(node)->AsFadeNode()) node=node->parent;
            const auto* reference=node ? REL::RelocateMember<RE::TESObjectREFR*>(node,0x0F8,0x110) : nullptr;
            entry.form=reference ? reference->GetFormID() : 0;
            const char* name=object.name.c_str();
            strncpy_s(entry.name,name ? name : "",_TRUNCATE);
        }
        bool Hidden(const RE::NiAVObject& object,const RE::NiCullingProcess& process,Count tested,Count culled,bool root) {
            auto& s=S();
            if(s.dumpRecording.load(relaxed)) { Record(object,process,root); return false; }
            Verdict verdict;
            if(s.sampling.load(relaxed)) verdict=HiddenTestTimed(object,process,tested,culled,root);
            else if(s.settings.ab && s.abState.load(relaxed)==uint8_t(AbState::Old)) verdict=HiddenTestBefore(object,process,tested,culled,root);
            else if(s.settings.ab && s.abState.load(relaxed)==uint8_t(AbState::EligibleFirst)) verdict=HiddenTestEligibleFirst(object,process,tested,culled,root);
            else verdict=HiddenTest(object,process,tested,culled,root);
            if(s.settings.diagnostics && verdict!=Verdict::Untested) NoteFlicker(object,root,verdict==Verdict::Hidden);
            if(verdict==Verdict::Hidden && s.auditing.load(relaxed)) AuditNote(object,root ? Audit::kHiZReference : Audit::kHiZGeometry);
            return verdict==Verdict::Hidden;
        }

        // Both thunks pass the original's return value through: the main draw keeps Main::Begin's RAX
        // (`mov r15, rax` right after the call at +0xD8) and dereferences it later in the frame. An earlier build crashed
        // on its first world frame with a void thunk that left Prepare's RAX there. Prepare runs before Main::Begin:
        // the cullers' frustums are already set from this frame's camera, and the helper builds while Begin runs.
        // `func` is whatever the call site called before this plugin: the engine function or another plugin's hook.
        struct BeginHook {
            static std::uint64_t thunk(std::uint64_t a1,std::uint64_t a2,std::uint64_t a3,std::uint64_t a4) {
                Prepare();
                return func(a1,a2,a3,a4);
            }
            static inline REL::Relocation<decltype(thunk)> func;
        };
        // When another plugin hooks this call too (CS Terrain Blending, Dynamic Terrain Deformation), either order works:
        // RenderDepth's own last step copies the prepass depth into the copy texture, which Capture reads directly.
        struct RenderDepthHook {
            static std::uint64_t thunk(std::uint64_t a1,std::uint64_t a2,std::uint64_t a3,std::uint64_t a4) {
                const auto result=func(a1,a2,a3,a4);
                Capture();
                BoxCapture();
                AuditCapture();
                return result;
            }
            static inline REL::Relocation<decltype(thunk)> func;
        };
        // Reference previs: the node's own userData names its reference (a reference's root fade or tree node).
        bool ReferencePrevisHidden(const RE::NiAVObject& node,const RE::NiCullingProcess& process) {
            auto& s=S();
            if(!s.referencesUsable.load(relaxed) || process.camera!=s.camera.load(relaxed)) return false;
            const auto* set=s.references.load(std::memory_order_acquire);
            const auto* reference=REL::RelocateMember<RE::TESObjectREFR*>(&node,0x0F8,0x110);
            if(!set || !reference || !std::binary_search(set->sorted.begin(),set->sorted.end(),reference->GetFormID())) return false;
            Bump(kReferencesCulled);
            return true;
        }
        // Whether skipping this node's OnVisible leaves its fade exactly as the engine would have it. BSFadeNode::OnVisible
        // (SE RVA 128FE30) does no fade work only for a node with kIgnoreFade whose fade amount (+0x100) and current fade
        // (+0x130) are both 1; for every other node it steps the fade by distance and time, and after two or more frames
        // without a visit it sets the current fade straight to 1. Skipping such a node froze or reset its fade: distant
        // rocks vanilla fades out showed again with Hi-Z on (near Fort Amol, precombines on or off).
        // "hiz_fade_rule": 1 (default) only nodes on that no-fade path are skipped whole; 2 also fully faded-in nodes
        // (both fades 1, kIgnoreFade or not); 0 any node (the original rule). A node not skipped keeps its fade bookkeeping and
        // its geometry is still tested one by one (AppendVirtual), so it can still be culled draw by draw.
        bool FadeSafe(RE::NiAVObject& node,const RE::NiCullingProcess& process) {
            auto& s=S();
            const uint32_t rule=s.settings.fadeRule;
            if(rule==0 || process.camera!=s.camera.load(relaxed)) return true;
            auto* fade=node.AsFadeNode();
            if(!fade) return true;
            const float amount=REL::RelocateMember<float>(&node,0x100,0x118);
            const float current=fade->GetRuntimeData().currentFade;
            const bool safe=amount==1.f && current==1.f && (rule==2 || node.GetFlags().all(RE::NiAVObject::Flag::kIgnoreFade));
            if(!safe) Bump(kFadeKept);
            return safe;
        }
        __forceinline bool SkipNode(RE::NiAVObject& node,RE::NiCullingProcess& process) {
            auto& s=S();
            if(s.sampling.load(relaxed)) {
                const uint64_t begin=__rdtsc();
                const bool previs=ReferencePrevisHidden(node,process);
                s.previsTicks.fetch_add(__rdtsc()-begin,relaxed); s.previsCalls.fetch_add(1,relaxed);
                if(previs) return true;
            } else if(ReferencePrevisHidden(node,process)) return true;
            return s.settings.fadeNodes && FadeSafe(node,process) && Hidden(node,process,kFadeTested,kFadeCulled,true);
        }
        // BSCullingProcess::AppendVirtual queues one geometry for rendering. Skinned geometry (actors) is left
        // alone; merged chunks have no fade node of their own, so this is where they are tested.
        bool PrevisHidden(const RE::BSGeometry& geometry,const RE::NiCullingProcess& process) {
            auto& s=S();
            const auto* set=s.previs.load(std::memory_order_acquire);
            if(!set || set->sorted.empty() || !geometry.parent || process.camera!=s.camera.load(relaxed)) return false;
            const RE::NiAVObject* parent=geometry.parent;
            if(!std::binary_search(set->sorted.begin(),set->sorted.end(),parent)) return false;
            Bump(kPrevisCulled);
            return true;
        }
        // The farthest of a geometry's vertices from its world bound centre (the CPU vertex copy, or the corners of its cached
        // mesh box), or -1 without either; `source` says which.
        float ProbeReach(const RE::BSGeometry& geometry,const char*& source) {
            source="no vertex data";
            const auto type=geometry.GetType().get();
            if(type!=RE::BSGeometry::Type::kTriShape && type!=RE::BSGeometry::Type::kMeshLODTriShape) return -1.f;
            const auto* shape=geometry.GetGeometryRuntimeData().rendererData;
            if(!shape) return -1.f;
            const auto& world=geometry.world;
            const auto& c=geometry.worldBound.center;
            const auto reach=[&](const float p[3]) {
                const auto& r=world.rotate.entry;
                float q[3];
                for(int i=0;i<3;++i) q[i]=(r[i][0]*p[0]+r[i][1]*p[1]+r[i][2]*p[2])*world.scale;
                const float dx=q[0]+world.translate.x-c.x,dy=q[1]+world.translate.y-c.y,dz=q[2]+world.translate.z-c.z;
                return std::sqrt(dx*dx+dy*dy+dz*dz);
            };
            float farthest=-1.f;
            if(shape->rawVertexData) {
                uint64_t desc;
                std::memcpy(&desc,&shape->vertexDesc,sizeof(desc));
                const uint32_t stride=uint32_t(desc&0xF)*4,offset=uint32_t(desc>>2)&0x3C;
                const bool full=FullPrecision(desc);
                const uint32_t vertices=static_cast<const RE::BSTriShape&>(geometry).GetTrishapeRuntimeData().vertexCount;
                if(!stride || stride<offset+(full ? 12u : 6u)) return -1.f;
                for(uint32_t v=0;v<vertices;++v) {
                    const auto* at=shape->rawVertexData+size_t(v)*stride+offset;
                    float p[3];
                    if(full) std::memcpy(p,at,sizeof(p));
                    else for(int k=0;k<3;++k) { uint16_t h; std::memcpy(&h,at+2*k,2); p[k]=HalfToFloat(h); }
                    farthest=std::max(farthest,reach(p));
                }
                source="CPU vertex copy";
                return farthest;
            }
            auto& shard=ShardOf(shape);
            std::lock_guard lock(shard.mutex);
            const auto it=shard.boxes.find(shape);
            if(it==shard.boxes.end() || !it->second.ok) return -1.f;
            for(int corner=0;corner<8;++corner) {
                const float p[3]={corner&1 ? it->second.hi[0] : it->second.lo[0],corner&2 ? it->second.hi[1] : it->second.lo[1],
                                  corner&4 ? it->second.hi[2] : it->second.lo[2]};
                farthest=std::max(farthest,reach(p));
            }
            source="mesh box corners";
            return farthest;
        }
        // Where the crosshair ray (from the eye along the camera's forward axis) first meets a geometry's triangles (CPU
        // vertex and index copies), in units along the ray; -1 when it misses, -2 without a CPU copy.
        float ProbeRay(const RE::BSGeometry& geometry,const RE::NiPoint3& eye,const float dir[3]) {
            const auto type=geometry.GetType().get();
            if(type!=RE::BSGeometry::Type::kTriShape && type!=RE::BSGeometry::Type::kMeshLODTriShape) return -2.f;
            const auto* shape=geometry.GetGeometryRuntimeData().rendererData;
            if(!shape || !shape->rawVertexData || !shape->rawIndexData) return -2.f;
            uint64_t desc;
            std::memcpy(&desc,&shape->vertexDesc,sizeof(desc));
            const uint32_t stride=uint32_t(desc&0xF)*4,offset=uint32_t(desc>>2)&0x3C;
            const bool full=FullPrecision(desc);
            const auto& data=static_cast<const RE::BSTriShape&>(geometry).GetTrishapeRuntimeData();
            if(!stride || stride<offset+(full ? 12u : 6u)) return -2.f;
            const auto& world=geometry.world;
            const auto vertex=[&](uint32_t v,float out[3]) {
                const auto* at=shape->rawVertexData+size_t(v)*stride+offset;
                float p[3];
                if(full) std::memcpy(p,at,sizeof(p));
                else for(int k=0;k<3;++k) { uint16_t h; std::memcpy(&h,at+2*k,2); p[k]=HalfToFloat(h); }
                const auto& r=world.rotate.entry;
                for(int i=0;i<3;++i) out[i]=(r[i][0]*p[0]+r[i][1]*p[1]+r[i][2]*p[2])*world.scale;
                out[0]+=world.translate.x-eye.x; out[1]+=world.translate.y-eye.y; out[2]+=world.translate.z-eye.z;
            };
            float nearest=-1.f;
            for(uint32_t i=0;i<data.triangleCount;++i) {
                const uint16_t* index=shape->rawIndexData+size_t(i)*3;
                if(index[0]>=data.vertexCount || index[1]>=data.vertexCount || index[2]>=data.vertexCount) continue;
                float a[3],b[3],c[3];
                vertex(index[0],a); vertex(index[1],b); vertex(index[2],c);
                // Moller-Trumbore, both faces.
                const float e1[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]},e2[3]={c[0]-a[0],c[1]-a[1],c[2]-a[2]};
                const float pv[3]={dir[1]*e2[2]-dir[2]*e2[1],dir[2]*e2[0]-dir[0]*e2[2],dir[0]*e2[1]-dir[1]*e2[0]};
                const float det=e1[0]*pv[0]+e1[1]*pv[1]+e1[2]*pv[2];
                if(std::abs(det)<1e-9f) continue;
                const float inv=1.f/det,tv[3]={-a[0],-a[1],-a[2]};
                const float u=(tv[0]*pv[0]+tv[1]*pv[1]+tv[2]*pv[2])*inv;
                if(u<0.f || u>1.f) continue;
                const float qv[3]={tv[1]*e1[2]-tv[2]*e1[1],tv[2]*e1[0]-tv[0]*e1[2],tv[0]*e1[1]-tv[1]*e1[0]};
                const float w=(dir[0]*qv[0]+dir[1]*qv[1]+dir[2]*qv[2])*inv;
                if(w<0.f || u+w>1.f) continue;
                const float hit=(e2[0]*qv[0]+e2[1]*qv[1]+e2[2]*qv[2])*inv;
                if(hit>0.f && (nearest<0.f || hit<nearest)) nearest=hit;
            }
            return nearest;
        }
        // Developer probe (Probe): one main-view geometry of the probe frame, named in the log when its screen rectangle
        // covers the centre texel (the crosshair).
        void ProbeGeometry(const RE::BSGeometry& geometry,const RE::NiCullingProcess& process) {
            auto& s=S();
            if(process.camera!=s.camera.load(relaxed) || s.eyes!=1) return;
            const auto* views=FramePyramid();
            if(!views || !views->eye[0].Ready()) { s.probeArmed.store(true,relaxed); return; }
            const auto& pyramid=views->eye[0];
            const auto& bound=geometry.worldBound;
            const float center[3]={bound.center.x,bound.center.y,bound.center.z};
            const auto e=pyramid.Explain(center,bound.radius);
            const uint32_t cx=pyramid.Width()/2,cy=pyramid.Height()/2;
            if(!e.occluder && !e.tx1 && !e.ty1) return;  // decided before its rectangle (near the eye, off screen)
            if(e.tx1<cx || e.tx0>cx || e.ty1<cy || e.ty0>cy) return;
            const uint32_t index=s.probeLines.fetch_add(1,relaxed);
            if(index>=State::kProbeLines) return;
            if(index==0) {
                const auto* camera=s.camera.load(relaxed);
                const float z=pyramid.At(cx,cy);
                const auto& f=camera->GetRuntimeData2().viewFrustum;
                logger::info("Hi-Z probe, frame {}: eye ({:.0f},{:.0f},{:.0f}), frustum near {:.2f} far {:.0f}; centre texel ({},{}) of {}x{}: depth {:.7f} = view depth {:.1f}",
                    s.cullFrame.load(relaxed),camera->world.translate.x,camera->world.translate.y,camera->world.translate.z,f.fNear,f.fFar,cx,cy,
                    pyramid.Width(),pyramid.Height(),z,z<1.f ? pyramid.ViewDepth(z) : 0.f);
            }
            const RE::NiAVObject* node=&geometry;
            std::string chain;
            for(int up=0;node && up<6;++up,node=node->parent) {
                if(up) chain+=" < ";
                chain+=node->name.c_str() ? node->name.c_str() : "";
                if(const_cast<RE::NiAVObject*>(node)->AsFadeNode()) break;
            }
            const auto* reference=node && const_cast<RE::NiAVObject*>(node)->AsFadeNode() ? REL::RelocateMember<RE::TESObjectREFR*>(node,0x0F8,0x110) : nullptr;
            const auto* base=reference ? reference->GetBaseObject() : nullptr;
            const char* model=nullptr;
            if(base) if(const auto* modelForm=base->As<RE::TESModel>()) model=modelForm->GetModel();
            const char* source="";
            const float reach=ProbeReach(geometry,source);
            const auto* camera=s.camera.load(relaxed);
            const auto& cr=camera->world.rotate.entry;
            const float forward[3]={cr[0][0],cr[1][0],cr[2][0]};
            const float ray=ProbeRay(geometry,camera->world.translate,forward);
            const auto flags=geometry.GetFlags();
            const auto* property=geometry.GetGeometryRuntimeData().shaderProperty.get();
            const bool alpha=geometry.GetGeometryRuntimeData().alphaProperty.get()!=nullptr;
            const auto& local=geometry.GetModelData().modelBound;
            std::string texels;
            for(uint32_t y=e.ty0;y<=e.ty1 && y<e.ty0+4;++y) for(uint32_t x=e.tx0;x<=e.tx1 && x<e.tx0+6;++x) {
                const float z=pyramid.At(x,y);
                texels+=z<1.f ? std::format(" {:.0f}",pyramid.ViewDepth(z)) : std::string(" far");
            }
            const bool skinned=geometry.GetGeometryRuntimeData().skinInstance!=nullptr;
            logger::info("Hi-Z probe {}: \"{}\" type {}{}, ref {:08X} base {:08X} model \"{}\"; {} ({}); {:.0f} units away, bound r {:.1f} at ({:.0f},{:.0f},{:.0f}), "
                         "model bound r {:.1f} scale {:.3f}, vertices reach {:.1f} ({}){}; sphere nearest view depth {:.1f}, rectangle texels {}-{} x {}-{} "
                         "(level {}), farthest there {:.7f} = view depth {:.1f}; first view depths{}{}; crosshair ray {}; flags {:08X}, shader {}{}",
                index,chain,static_cast<uint32_t>(geometry.GetType().get()),skinned ? " skinned" : "",reference ? reference->GetFormID() : 0u,
                base ? base->GetFormID() : 0u,model ? model : "",e.hidden ? "HIDDEN" : "drawn",e.why,e.distance,bound.radius,center[0],center[1],center[2],
                local.radius,geometry.world.scale,reach,source,reach>bound.radius*1.01f+1.f ? " BEYOND THE BOUND" : "",e.wmin,e.tx0,e.tx1,e.ty0,e.ty1,e.level,
                e.occluder,e.occluderView,texels,s.visible.Skip(&geometry,s.cullFrame.load(relaxed)) ? "; the visible table skips its test this frame" : "",
                ray>=0.f ? std::format("hits at {:.0f}",ray) : ray<-1.5f ? std::string("not testable") : std::string("misses"),
                static_cast<uint32_t>(flags.underlying()),property && property->GetRTTI() && property->GetRTTI()->name ? property->GetRTTI()->name : "none",alpha ? ", alpha property" : "");
        }
        __forceinline bool SkipGeometry(RE::NiCullingProcess& process,RE::BSGeometry& object) {
            auto& s=S();
            // Some culling processes append whole nodes; only real geometry is tested.
            if(auto* geometry=static_cast<RE::NiAVObject&>(object).AsGeometry()) {
                if(s.settings.diagnostics && s.probeFrame.load(relaxed)==s.cullFrame.load(relaxed) && s.probeFrame.load(relaxed)) ProbeGeometry(*geometry,process);
                if(s.sampling.load(relaxed)) {
                    const uint64_t begin=__rdtsc();
                    const bool previs=PrevisHidden(*geometry,process);
                    s.previsTicks.fetch_add(__rdtsc()-begin,relaxed); s.previsCalls.fetch_add(1,relaxed);
                    if(previs) return true;
                } else if(PrevisHidden(*geometry,process)) return true;
                // The box test registers every static main-view geometry, culled or not, so each keeps a fresh result.
                const bool boxes=s.boxRegistering.load(relaxed) && process.camera==s.camera.load(relaxed) &&
                                 !geometry->GetGeometryRuntimeData().skinInstance && Eligible(*geometry,false);
                if(boxes) RegisterBox(*geometry);
                if(s.settings.geometry && !geometry->GetGeometryRuntimeData().skinInstance &&
                   Hidden(*geometry,process,kGeometryTested,kGeometryCulled,false)) return true;
                if(boxes && BoxHidden(*geometry)) {
                    if(s.auditing.load(relaxed)) AuditNote(*geometry,Audit::kBoxTest);
                    return true;
                }
            }
            return false;
        }
        // One hook per vtable, each with its own original: whatever the slot held when this plugin installed (the
        // engine's function or another plugin's hook) is called for everything not skipped. BSFadeNode::OnVisible
        // computes the node's fade and then visits its children; skipping it skips the whole reference. A node skipped
        // for two or more frames returns at full fade (its fade timer resets). BSLeafAnimNode and BSTreeNode (animated
        // plants and trees) override OnVisible.
        enum : int { kFadeNode, kLeafAnimNode, kTreeNode };
        // The reference behind a widened near exemption, named once in the log (diagnostics; at most kEyeLogLimit).
        void NameEyeReference(uint32_t formID,const char* why,float radius,float distance,float reach) {
            auto& s=S();
            if(!s.settings.diagnostics) return;
            {
                std::lock_guard lock(s.eyeLogMutex);
                if(s.eyeLogged.size()>=State::kEyeLogLimit || !s.eyeLogged.insert(formID).second) return;
            }
            const auto* form=RE::TESForm::LookupByID(formID);
            const auto* reference=form ? form->As<RE::TESObjectREFR>() : nullptr;
            const auto* base=reference ? reference->GetBaseObject() : nullptr;
            const auto* actor=reference ? reference->As<RE::Actor>() : nullptr;
            logger::info("Hi-Z near exemption widened ({}): reference {:08X} base {:08X} \"{}\"{}, bound radius {:.0f} (widened), {:.0f} units from the eye; exemption {:.0f} units",
                why,formID,base ? base->GetFormID() : 0u,reference ? reference->GetName() : "",actor && actor->IsDead() ? ", dead" : "",radius,distance,reach);
        }
        void NoteMoving(const RE::NiAVObject& node,const RE::NiCullingProcess& process) {
            auto& s=S();
            if(!s.sampling.load(relaxed)) return NoteMovingBody(node,process);
            const uint64_t begin=__rdtsc();
            NoteMovingBody(node,process);
            s.movingTicks.fetch_add(__rdtsc()-begin,relaxed); s.movingCalls.fetch_add(1,relaxed);
        }
        // A visible root of an actor or a door in the main view: recorded for this frame's capture (see State::moving).
        // Activators and movable statics are not: many are large effect volumes (mist, dust) whose bounds hold the
        // camera and which write no depth (every capture in Ustengrav02 was dropped for one).
        // At the eye: the player is left out (first-person geometry is inside the near exemption) and so is a door
        // (walking through it); another actor there (a mount in first person, a creature in the player's face) widens
        // the capture's near exemption to the far side of its bounds (see Capture).
        void NoteMovingBody(const RE::NiAVObject& node,const RE::NiCullingProcess& process) {
            auto& s=S();
            if(!s.settings.excludeMoving || !s.enabled.load(relaxed) || process.camera!=s.camera.load(relaxed)) return;
            const auto* reference=REL::RelocateMember<RE::TESObjectREFR*>(&node,0x0F8,0x110);
            if(!reference) return;
            const bool actor=reference->GetFormType()==RE::FormType::ActorCharacter;
            if(!actor) {
                const auto* base=reference->GetBaseObject();
                if(!base || base->GetFormType()!=RE::FormType::Door) return;
            }
            const auto& bound=node.worldBound;
            if(!(bound.radius>0.f)) return;
            // Skinned and animated bounds can trail the pose by a frame: widen them.
            float radius=bound.radius*1.25f+16.f;
            RE::NiPoint3 center=bound.center;
            // An actor's root bound can be far larger than its body (a Skeleton's 4,605 units, Wisps' 11,254
            // with their effects), which put it at the eye from 1,800-3,600 units away and emptied the capture. Beyond 1.5
            // times the actor's own bounding box plus 64 units, that box around the reference stands in for the root bound.
            if(actor) {
                const auto* body=static_cast<const RE::Actor*>(reference);
                const auto lo=body->GetBoundMin(),hi=body->GetBoundMax();
                const float scale=reference->GetScale();
                const float bx=hi.x-lo.x,by=hi.y-lo.y,bz=hi.z-lo.z;
                const float box=0.5f*std::sqrt(bx*bx+by*by+bz*bz)*scale;
                if(box>0.f && radius>box*1.5f+64.f) {
                    const auto& at=reference->GetPosition();
                    center={at.x,at.y,at.z+0.5f*(lo.z+hi.z)*scale};
                    radius=box*1.5f+64.f;
                    s.boundsReplaced.fetch_add(1,relaxed);
                }
            }
            if(const auto* camera=s.camera.load(relaxed)) {
                const auto& eye=camera->world.translate;
                const float dx=center.x-eye.x,dy=center.y-eye.y,dz=center.z-eye.z;
                const float d2=dx*dx+dy*dy+dz*dz;
                if(d2<=(radius+16.f)*(radius+16.f)) {
                    if(actor && reference!=RE::PlayerCharacter::GetSingleton()) {
                        s.movingAtEye.store(true,relaxed);
                        const float reach=std::sqrt(d2)+radius+16.f;
                        NameEyeReference(reference->GetFormID(),"a reference at the eye",radius,std::sqrt(d2),reach);
                        for(float prior=s.eyeExempt.load(relaxed);reach>prior && !s.eyeExempt.compare_exchange_weak(prior,reach,relaxed););
                    }
                    return;
                }
            }
            const uint32_t index=s.movingCount.fetch_add(1,relaxed);
            if(index<State::kMaxMoving) { s.moving[index]={center.x,center.y,center.z,radius}; s.movingForm[index]=reference->GetFormID(); }
        }

        template<int Table> struct OnVisibleHook {
            static void thunk(RE::NiAVObject* node,RE::NiCullingProcess& process,std::int32_t alphaGroup) {
                if(SkipNode(*node,process)) return;
                NoteMoving(*node,process);
                func(node,process,alphaGroup);
            }
            static inline REL::Relocation<decltype(thunk)> func;
        };
        // AppendVirtual is slot 0x18 of BSCullingProcess and of BSGeometryListCullingProcess, the main view's culler.
        enum : int { kCullingProcess, kGeometryListCullingProcess };
        template<int Table> struct AppendVirtualHook {
            static void thunk(RE::NiCullingProcess* process,RE::BSGeometry& object,std::int32_t alphaGroup) {
                if(SkipGeometry(*process,object)) return;
                func(process,object,alphaGroup);
            }
            static inline REL::Relocation<decltype(thunk)> func;
        };

        // ---- Coexistence with other plugins' hooks ----

        bool Readable(std::uintptr_t address,std::size_t size) {
            MEMORY_BASIC_INFORMATION info{};
            if(!address || !VirtualQuery(reinterpret_cast<LPCVOID>(address),&info,sizeof(info)) || info.State!=MEM_COMMIT) return false;
            constexpr DWORD readable=PAGE_READONLY|PAGE_READWRITE|PAGE_WRITECOPY|PAGE_EXECUTE_READ|PAGE_EXECUTE_READWRITE|PAGE_EXECUTE_WRITECOPY;
            if((info.Protect&(PAGE_GUARD|PAGE_NOACCESS)) || !(info.Protect&readable)) return false;
            return address+size<=reinterpret_cast<std::uintptr_t>(info.BaseAddress)+info.RegionSize;
        }
        bool IsCode(std::uintptr_t address) {
            MEMORY_BASIC_INFORMATION info{};
            if(!address || !VirtualQuery(reinterpret_cast<LPCVOID>(address),&info,sizeof(info)) || info.State!=MEM_COMMIT) return false;
            constexpr DWORD executable=PAGE_EXECUTE|PAGE_EXECUTE_READ|PAGE_EXECUTE_READWRITE|PAGE_EXECUTE_WRITECOPY;
            return !(info.Protect&(PAGE_GUARD|PAGE_NOACCESS)) && (info.Protect&executable);
        }
        // The destination of one branch stub (an SKSE/CommonLib trampoline `jmp [rip]`, a `jmp rel32`, `jmp rel8`, or
        // `mov rax|r11, imm64; jmp rax|r11`), or 0.
        std::uintptr_t NextHop(std::uintptr_t at) {
            if(!Readable(at,2)) return 0;
            const auto* b=reinterpret_cast<const std::uint8_t*>(at);
            const auto read32=[](std::uintptr_t p) { std::int32_t v; std::memcpy(&v,reinterpret_cast<const void*>(p),4); return static_cast<std::intptr_t>(v); };
            const auto read64=[](std::uintptr_t p) { std::uintptr_t v; std::memcpy(&v,reinterpret_cast<const void*>(p),8); return v; };
            if(b[0]==0xFF && b[1]==0x25 && Readable(at,6)) {
                const auto slot=at+6+read32(at+2);
                return Readable(slot,8) ? read64(slot) : 0;
            }
            if(b[0]==0xE9 && Readable(at,5)) return at+5+read32(at+1);
            if(b[0]==0xEB) return at+2+static_cast<std::int8_t>(b[1]);
            if(b[0]==0x48 && b[1]==0xB8 && Readable(at,12) && b[10]==0xFF && b[11]==0xE0) return read64(at+2);
            if(b[0]==0x49 && b[1]==0xBB && Readable(at,13) && b[10]==0x41 && b[11]==0xFF && b[12]==0xE3) return read64(at+2);
            return 0;
        }
        std::string ModuleName(std::uintptr_t address,std::uintptr_t& base) {
            HMODULE module{};
            if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   reinterpret_cast<LPCWSTR>(address),&module) || !module) return {};
            wchar_t path[MAX_PATH]{};
            const DWORD length=GetModuleFileNameW(module,path,MAX_PATH);
            if(!length) return {};
            std::wstring_view file(path,length);
            if(const auto slash=file.find_last_of(L"\\/");slash!=std::wstring_view::npos) file.remove_prefix(slash+1);
            std::string name;
            for(const wchar_t c:file) name.push_back(c>0 && c<0x80 ? static_cast<char>(c):'?');
            base=reinterpret_cast<std::uintptr_t>(module);
            return name;
        }

        // Who answers a call to `address`: the module that owns it, after following up to four branch stubs out of
        // memory that belongs to no module (SKSE trampolines), plus any in-place detour of an engine function's entry.
        struct Owner {
            bool code=false;    // committed executable memory
            bool module=false;  // it, or the end of its stubs, lies in a loaded module
            bool engine=false;  // that module is the game itself
            std::uintptr_t resolved=0;  // the address in that module
            std::string text;
        };
        Owner Inspect(std::uintptr_t address) {
            Owner out;
            if(!(out.code=IsCode(address))) { out.text=std::format("{:X} (not executable)",address); return out; }
            std::uintptr_t at=address,base=0;
            int hops=0;
            auto name=ModuleName(at,base);
            while(name.empty() && hops<4) {
                const auto next=NextHop(at);
                if(!next || !IsCode(next)) break;
                at=next; ++hops;
                name=ModuleName(at,base);
            }
            if(name.empty()) { out.text=std::format("{:X} (no module; unrecognized stub)",address); return out; }
            out.module=true;
            out.engine=base==REL::Module::get().base();
            out.resolved=at;
            out.text=std::format("{}+0x{:X}",name,at-base);
            if(hops) out.text+=std::format(" via {} stub{} at {:X}",hops,hops>1 ? "s":"",address);
            if(out.engine && !hops) {
                std::uintptr_t to=at,toBase=0;
                std::string toName;
                int jumps=0;
                while(jumps<4) {
                    const auto next=NextHop(to);
                    if(!next || !IsCode(next)) break;
                    to=next; ++jumps;
                    toName=ModuleName(to,toBase);
                    if(!toName.empty() && toBase!=base) break;
                }
                if(jumps && (toName.empty() || toBase!=base))
                    out.text+=" (entry detoured to "+(toName.empty() ? std::format("{:X}",to):std::format("{}+0x{:X}",toName,to-toBase))+")";
            }
            return out;
        }
        // A hook can be chained when the previous entry is executable code that resolves into a loaded module.
        bool Chainable(const Owner& owner) { return owner.code && owner.module; }
        // `expected` is the engine function this game version has there, or 0 when only the module is known.
        void Note(std::string& summary,std::string_view point,const Owner& owner,bool hooked,std::uintptr_t expected=0) {
            const char* who=!owner.engine ? "already hooked by " : expected && owner.resolved!=expected ? "unexpected engine function " : "engine ";
            summary+=std::format("{}{}: {}{}{}",summary.empty() ? "":"; ",point,who,owner.text,hooked ? "":" [not chained]");
        }

        bool ReadCall(std::uintptr_t site,std::uintptr_t& target) {
            if(!Readable(site,5) || *reinterpret_cast<const std::uint8_t*>(site)!=0xE8) return false;
            std::int32_t displacement;
            std::memcpy(&displacement,reinterpret_cast<const void*>(site+1),4);
            target=site+5+static_cast<std::intptr_t>(displacement);
            return true;
        }
        std::uintptr_t& VtableEntry(std::uintptr_t table,std::size_t index) { return reinterpret_cast<std::uintptr_t*>(table)[index]; }

        template<class Hook>
        bool HookSlot(const char* name,const REL::VariantID& id,std::size_t slot,std::uintptr_t expected,std::string& summary) {
            REL::Relocation<std::uintptr_t> table{id};
            const auto owner=Inspect(VtableEntry(table.address(),slot));
            const bool chain=Chainable(owner);
            Note(summary,name,owner,chain,expected);
            if(!chain) {
                logger::error("Hi-Z: {} holds {}, which is not code in a loaded module; that table is not hooked",name,owner.text);
                return false;
            }
            Hook::func=table.write_vfunc(slot,Hook::thunk);
            S().written.push_back({name,table.address()+slot*sizeof(std::uintptr_t),false,VtableEntry(table.address(),slot)});
            return true;
        }

        // Once, at the first capture (after every plugin's load-time and post-load hooks): which hook points another
        // plugin has hooked since, so it now runs before this plugin's hook and calls it.
        void LogLaterHooks() {
            std::string later;
            for(const auto& w:S().written) {
                std::uintptr_t now=0;
                if(w.call) { if(!ReadCall(w.at,now)) now=0; }
                else now=*reinterpret_cast<const std::uintptr_t*>(w.at);
                if(now==w.ours) continue;
                later+=std::format("{}{}: {}",later.empty() ? "":"; ",w.name,now ? Inspect(now).text : std::string("no longer a 5-byte call"));
            }
            if(later.empty()) logger::info("Hi-Z hook coexistence at the first frame: no plugin hooked these points after this one");
            else logger::info("Hi-Z hook coexistence at the first frame: hooked after this plugin (they call it): {}",later);
        }

        // A boolean as SimpleIni reads it (SPS's INI library): the last `key` in `section`, true for t/y/1/on and false
        // for f/n/0/off by the first letters; nothing when the file, the key or a boolean value is missing.
        std::optional<bool> IniBool(const char* path,std::string_view section,std::string_view key) {
            std::ifstream in(path);
            if(!in) return std::nullopt;
            const auto trim=[](std::string_view t) {
                while(!t.empty() && std::isspace(static_cast<unsigned char>(t.front()))) t.remove_prefix(1);
                while(!t.empty() && std::isspace(static_cast<unsigned char>(t.back()))) t.remove_suffix(1);
                return t;
            };
            const auto same=[](std::string_view a,std::string_view b) {
                return a.size()==b.size() && std::equal(a.begin(),a.end(),b.begin(),[](char x,char y) { return std::tolower(static_cast<unsigned char>(x))==std::tolower(static_cast<unsigned char>(y)); });
            };
            std::optional<bool> out;
            std::string line;
            bool inSection=false;
            while(std::getline(in,line)) {
                const auto t=trim(line);
                if(t.empty() || t.front()==';' || t.front()=='#') continue;
                if(t.front()=='[') { inSection=t.back()==']' && same(trim(t.substr(1,t.size()-2)),section); continue; }
                const auto equals=t.find('=');
                if(!inSection || equals==std::string_view::npos || !same(trim(t.substr(0,equals)),key)) continue;
                const auto value=trim(t.substr(equals+1));
                const char c0=value.empty() ? '\0' : static_cast<char>(std::tolower(static_cast<unsigned char>(value[0])));
                const char c1=value.size()<2 ? '\0' : static_cast<char>(std::tolower(static_cast<unsigned char>(value[1])));
                if(c0=='t' || c0=='y' || c0=='1' || (c0=='o' && c1=='n')) out=true;
                else if(c0=='f' || c0=='n' || c0=='0' || (c0=='o' && c1=='f')) out=false;
            }
            return out;
        }

        // Skyrim Performance Solution's "Hide occluded objects" (BETA, [Occlusion] bCull, off by default): SPS.ini holds
        // the defaults and SPS_User.ini what its menu changed (SPS 1.0.63, Config.cpp).
        bool SpsCulls() {
            if(!GetModuleHandleW(L"SPS.dll")) return false;
            auto value=IniBool("Data/SKSE/Plugins/SPS.ini","Occlusion","bCull");
            if(const auto user=IniBool("Data/SKSE/Plugins/SPS_User.ini","Occlusion","bCull")) value=user;
            return value.value_or(false);
        }

        // Bottled Shaders (a Community Shaders fork; the same CommunityShaders.dll) and its "HiZ Occlusion Culling"
        // feature (as of its commit 700607fb): installed when Shaders/Features/HiZOcclusion.ini exists, loaded unless "Disable at
        // Boot" lists it, culling unless "enableHiZCulling" is false (default true); SettingsUser.json over
        // SettingsDefault.json. Its culling hook (the accumulator's registration call in AE ID 76558) is installed only
        // on AE before 1.7.99: elsewhere it builds its pyramid and culls nothing.
        struct Bottled { bool installed=false,enabled=false,culls=false; };
        Bottled BottledHiZ() {
            Bottled out;
            if(!GetModuleHandleW(L"CommunityShaders.dll") || !std::filesystem::exists("Data/Shaders/Features/HiZOcclusion.ini")) return out;
            out.installed=true;
            bool disabled=false,culling=true;
            for(const char* path:{"Data/SKSE/Plugins/CommunityShaders/SettingsDefault.json","Data/SKSE/Plugins/CommunityShaders/SettingsUser.json"}) {
                std::ifstream in(path);
                if(!in) continue;
                const auto settings=nlohmann::json::parse(in,nullptr,false);
                if(!settings.is_object()) continue;
                if(const auto boot=settings.find("Disable at Boot");boot!=settings.end() && boot->is_object())
                    if(const auto entry=boot->find("HiZOcclusion");entry!=boot->end() && entry->is_boolean()) disabled=entry->get<bool>();
                if(const auto feature=settings.find("HiZ Occlusion Culling");feature!=settings.end() && feature->is_object())
                    if(const auto entry=feature->find("enableHiZCulling");entry!=feature->end() && entry->is_boolean()) culling=entry->get<bool>();
            }
            out.enabled=!disabled && culling;
            out.culls=out.enabled && REL::Module::IsAE() && REL::Module::get().version()<REL::Version(1,7,99,0);
            return out;
        }
    }

    void Install() {
        {
            auto& s=S();
            s.clockStart=Clock::now(); s.tscStart=__rdtsc();
            uint64_t best=~0ull;
            for(int i=0;i<1000;++i) { const uint64_t a=__rdtsc(); const uint64_t b=__rdtsc(); best=std::min(best,b-a); }
            s.tscCost=best;
        }
        if(!SupportedRuntime()) { logger::info("Hi-Z occlusion: unsupported game version; hooks not installed"); return; }
        // VR renders both eyes in one main draw, side by side (stereo instancing), and has its own GPU test of geometry
        // against the depth prepass (bDepthBufferCulling, switched on by VRDepthCulling). Hi-Z on VR tests both eyes and
        // is installed only with hiz_vr=true (SkyrimPrecombines.ini), read here because the hooks go in at plugin load.
        const bool vr=REL::Module::IsVR();
        if(vr) {
            const auto& config=Config::Get();
            const auto key=config.find("hiz_vr");
            if(key==config.end() || !key->is_boolean() || !key->get<bool>()) {
                logger::info("Hi-Z occlusion on Skyrim VR is off (hiz_vr=true installs it); hooks not installed");
                return;
            }
            S().eyes=2;
        }
        // Main world draw (SE ID 35560, RVA 5B1860; AE ID 36559; VR 5B9330): call Main::Begin at +0xD3 in every version
        // (before the culling jobs are queued) and call Main::RenderDepth (the depth prepass and its copy) at +0x395 on SE
        // and AE 1.6 (1.6.1170 Steam, 1.6.1179 GOG), +0x3B3 on AE 1.7.104, +0x2EE on VR. Each call is chained to whatever
        // it calls now: the engine (SE IDs 35562, 100421; AE 36561, 107139) or a plugin that hooked the call first.
        const REL::Relocation<std::uintptr_t> draw{REL::RelocationID(35560,36559)};
        const std::uintptr_t depthOffset=vr ? 0x2EE : REL::Module::get().version()>=REL::Version(1,7,0,0) ? 0x3B3:0x395;
        const std::uintptr_t beginSite=draw.address()+0xD3,depthSite=draw.address()+depthOffset;
        std::uintptr_t beginTarget=0,depthTarget=0;
        if(!ReadCall(beginSite,beginTarget) || !ReadCall(depthSite,depthTarget)) {
            logger::error("Hi-Z occlusion: a main draw call site is no longer a 5-byte call (another plugin rewrote it); hooks not installed");
            return;
        }
        const auto beginOwner=Inspect(beginTarget),depthOwner=Inspect(depthTarget);
        std::string summary;
        Note(summary,"Main::Begin call",beginOwner,Chainable(beginOwner),REL::RelocationID(35562,36561).address());
        Note(summary,"Main::RenderDepth call",depthOwner,Chainable(depthOwner),REL::RelocationID(100421,107139).address());
        if(!Chainable(beginOwner) || !Chainable(depthOwner)) {
            logger::error("Hi-Z occlusion: a main draw call site targets something that is not code in a loaded module; hooks not installed ({})",summary);
            return;
        }
        SKSE::AllocTrampoline(64);
        auto& trampoline=SKSE::GetTrampoline();
        // CommonLib 13 takes trampoline memory only from SKSE's trampoline interface and, unlike 4.39.3, does not fall
        // back to its own allocation when that interface is missing.
        if(trampoline.free_size()<64) trampoline.create(64);
        BeginHook::func=trampoline.write_call<5>(beginSite,BeginHook::thunk);
        RenderDepthHook::func=trampoline.write_call<5>(depthSite,RenderDepthHook::thunk);
        for(const auto& [name,site]:{std::pair{"Main::Begin call",beginSite},std::pair{"Main::RenderDepth call",depthSite}}) {
            std::uintptr_t ours=0;
            ReadCall(site,ours);
            S().written.push_back({name,site,true,ours});
        }
        // OnVisible is slot 0x34 on SE and AE (BSFadeNode's: SE ID 98848, AE 105485; BSLeafAnimNode and BSTreeNode have
        // their own) and 0x35 on VR (one extra VR virtual before it); AppendVirtual is slot 0x18 (both culler tables:
        // BSCullingProcess::AppendVirtual, SE ID 74807, AE 76556, VR D99F80). The VR address library has neither ID, so
        // there the log names only the module.
        const std::uintptr_t fadeVisible=vr ? 0 : REL::RelocationID(98848,105485).address(),append=vr ? 0 : REL::RelocationID(74807,76556).address();
        const std::size_t visibleSlot=vr ? 0x35 : 0x34;
        size_t visibleTables=0;
        visibleTables+=HookSlot<OnVisibleHook<kFadeNode>>("BSFadeNode::OnVisible",RE::VTABLE_BSFadeNode[0],visibleSlot,fadeVisible,summary);
        visibleTables+=HookSlot<OnVisibleHook<kLeafAnimNode>>("BSLeafAnimNode::OnVisible",RE::VTABLE_BSLeafAnimNode[0],visibleSlot,0,summary);
        visibleTables+=HookSlot<OnVisibleHook<kTreeNode>>("BSTreeNode::OnVisible",RE::VTABLE_BSTreeNode[0],visibleSlot,0,summary);
        const bool cullingHooked=HookSlot<AppendVirtualHook<kCullingProcess>>("BSCullingProcess::AppendVirtual",RE::VTABLE_BSCullingProcess[0],0x18,
            append,summary);
        const bool listHooked=HookSlot<AppendVirtualHook<kGeometryListCullingProcess>>("BSGeometryListCullingProcess::AppendVirtual",
            RE::VTABLE_BSGeometryListCullingProcess[0],0x18,append,summary);
        S().installed=listHooked;
        logger::info("Hi-Z occlusion hooks installed: main draw begin/depth, OnVisible on {} of 3 node tables, AppendVirtual on {} of 2 culler tables",
            visibleTables,int(cullingHooked)+int(listHooked));
        logger::info("Hi-Z hook coexistence: {}",summary);
    }

    bool Installed() { return S().installed; }

    void SetPrevisHidden(std::vector<RE::NiPointer<RE::NiAVObject>> roots) {
        auto& s=S();
        auto set=std::make_shared<PrevisSet>();
        set->sorted.reserve(roots.size());
        for(const auto& root:roots) if(root) set->sorted.push_back(root.get());
        std::sort(set->sorted.begin(),set->sorted.end());
        set->held=std::move(roots);
        std::lock_guard lock(s.previsMutex);
        // Skip republishing an identical set (the common case on every task).
        if(const auto* current=s.previs.load(relaxed);current && current->sorted==set->sorted) return;
        s.previsRetained.push_back(set);
        s.previs.store(set.get(),std::memory_order_release);
        // A culling pass lasts one frame; snapshots replaced eight publications ago are no longer read.
        if(s.previsRetained.size()>8) s.previsRetained.erase(s.previsRetained.begin());
    }

    void SetReferencePrevisHidden(std::vector<uint32_t> forms,const std::array<int64_t,3>& cell,float size) {
        auto& s=S();
        auto set=std::make_shared<ReferenceSet>();
        std::sort(forms.begin(),forms.end());
        set->sorted=std::move(forms); set->cell=cell; set->size=size;
        std::lock_guard lock(s.previsMutex);
        if(const auto* current=s.references.load(relaxed);current && current->sorted==set->sorted && current->cell==set->cell && current->size==set->size) return;
        s.referencesRetained.push_back(set);
        s.references.store(set.get(),std::memory_order_release);
        // A culling pass lasts one frame; snapshots replaced eight publications ago are no longer read.
        if(s.referencesRetained.size()>8) s.referencesRetained.erase(s.referencesRetained.begin());
    }

    void Configure(const Settings& settings) {
        auto& s=S();
        s.settings=settings;
        if(!s.initTried) {
            s.gridW=std::clamp(settings.gridWidth,32u,1024u)&~1u;
            s.gridH=s.gridW/2;
        }
        s.minFrameMs.store(settings.minFrameMs,relaxed);
        s.minYield.store(settings.minYield,relaxed);
        {
            std::lock_guard lock(s.switchMutex);
            s.configured.store(settings.enabled,relaxed);
            s.enabled.store(settings.enabled && !s.yielded.load(relaxed),relaxed);
        }
        logger::info("Hi-Z occlusion {} (fade nodes {}, geometry {}, {}x{} grid, near exemption {:.0f}, moving references excluded {}, wait {} us, split build {})",
            settings.enabled ? "ON":"OFF",settings.fadeNodes,settings.geometry,s.gridW,s.gridH,settings.nearExempt,settings.excludeMoving,settings.waitMicros,
            settings.splitBuild);
        logger::info("Hi-Z works while the smoothed frame time is at least {:.1f} ms (0: always)",settings.minFrameMs);
        logger::info("Hi-Z tests every frame while it hides at least {:.0f}% of what it tests, otherwise {} of every {} frames (0%: always)",
            settings.minYield*100.f,settings.probeBurst,settings.probeInterval);
        // The registration arrays exist from the start, so the developer toggle can switch the box test on in play.
        if(s.boxes.size()!=Boxes::kMaxBoxes) { s.boxes.resize(Boxes::kMaxBoxes); s.boxKeys.resize(Boxes::kMaxBoxes); }
        if(settings.boxes)
            // The shift each pass uses (Boxes::MotionShift in BoxCaptureBody) also depends on the camera's near plane, unknown
            // here, so the line names both bounds rather than one number.
            logger::info("Box occlusion ON (developer): geometry boxes against the prepass depth eroded for box_occlusion_max_shift {:.0f} units of eye "
                         "motion, or half the larger of the near plane and the near exemption ({:.0f}) when that is less, and used while the eye moved at "
                         "most that far since the result; depth target 1/{} of the rendered area",settings.boxMaxShift,settings.nearExempt,
                         std::clamp(settings.boxDownsample,1u,8u));
    }

    bool ToggleGeometry() {
        auto& s=S();
        s.settings.geometry=!s.settings.geometry;
        return s.settings.geometry;
    }

    bool ToggleBoxes() {
        auto& s=S();
        s.settings.boxes=!s.settings.boxes;
        if(!s.settings.boxes) { s.boxActive.store(nullptr,std::memory_order_release); s.boxRegistering.store(false,relaxed); }
        return s.settings.boxes;
    }

    bool Probe() {
        auto& s=S();
        if(!s.settings.diagnostics) return false;
        s.probeTries=0;
        s.probeArmed.store(true,relaxed);
        return true;
    }

    void SetEnabled(bool on) {
        auto& s=S();
        bool paused=false;
        {
            std::lock_guard lock(s.switchMutex);
            s.configured.store(on,relaxed);
            paused=s.yielded.load(relaxed);
            const bool effective=on && !paused;
            s.enabled.store(effective,relaxed);
            // The box test's published result goes at once too, not at the next Prepare.
            if(!effective) { s.active.store(nullptr,std::memory_order_release); s.boxActive.store(nullptr,std::memory_order_release); s.boxRegistering.store(false,relaxed); }
        }
        logger::info("Hi-Z occlusion {}{}",on ? "ON":"OFF",paused && on ? " (paused while another plugin's occlusion culling is on)":"");
    }

    // No log: an SKSE Menu Framework slider sets these every frame while it is dragged.
    void SetMinFrameMs(float ms) { if(std::isfinite(ms)) S().minFrameMs.store(std::clamp(ms,0.f,100.f),relaxed); }
    void SetMinYield(float share) { if(std::isfinite(share)) S().minYield.store(std::clamp(share,0.f,1.f),relaxed); }

    bool Toggle() {
        auto& s=S();
        SetEnabled(!s.configured.load(relaxed));
        return s.enabled.load(relaxed);
    }

    void CheckOtherCullers() {
        auto& s=S();
        // While Bottled Shaders' HiZ Occlusion Culling culls or SPS's "Hide occluded objects" is on, Hi-Z pauses (two
        // occlusion cullers would test every object twice); when neither does, it returns to the configured state. Said in
        // the log only. Bottled culls only where its hook installs (BottledHiZ): with its option on and
        // nothing culled, Hi-Z stays on.
        const auto bottled=BottledHiZ();
        const bool sps=SpsCulls();
        const std::string other=bottled.culls && sps ? "Bottled Shaders' HiZ Occlusion Culling and SPS \"Hide occluded objects\"" :
                                bottled.culls ? "Bottled Shaders' HiZ Occlusion Culling" : sps ? "SPS \"Hide occluded objects\"" : "";
        std::lock_guard lock(s.switchMutex);  // against SetEnabled from the menu or the Hi-Z key
        if(!other.empty() && !s.yielded.load(relaxed)) {
            s.yielded.store(true,relaxed);
            s.enabled.store(false,relaxed);
            s.active.store(nullptr,std::memory_order_release);
            s.boxActive.store(nullptr,std::memory_order_release); s.boxRegistering.store(false,relaxed);
            logger::info("Hi-Z occlusion OFF: {} is on and hides occluded objects already; precombines stay on",other);
        } else if(other.empty() && s.yielded.load(relaxed)) {
            const bool on=s.configured.load(relaxed);
            s.yielded.store(false,relaxed);
            s.enabled.store(on,relaxed);
            logger::info("Hi-Z occlusion {}: no other plugin's occlusion culling is on now",on ? "ON":"OFF");
        }
        if(bottled.enabled && !bottled.culls)
            logger::info("Bottled Shaders' HiZ Occlusion Culling is on but culls nothing on this game version; Hi-Z occlusion stays {}",
                s.enabled.load(relaxed) ? "ON":"OFF");
    }

    void TakeAbBlocks(std::vector<AbBlock>& out) {
        auto& s=S();
        out.clear();
        std::lock_guard lock(s.abMutex);
        out.swap(s.abDone);
    }

    void TakeFrameTimes(std::vector<float>& out) {
        auto& s=S();
        out.clear();
        std::lock_guard lock(s.frameMutex);
        out.swap(s.frameTimes);
    }

    Cost ReadCost() {
        auto& s=S();
        Cost out;
        out.eyeWidest=s.eyeWidest.exchange(0.f,relaxed);
        if(!s.tscStart) return out;
        const double micros=std::chrono::duration<double,std::micro>(Clock::now()-s.clockStart).count();
        const double perMicro=micros>0.0 ? double(__rdtsc()-s.tscStart)/micros : 0.0;
        if(!(perMicro>0.0)) return out;
        const auto net=[&](uint64_t ticks,uint64_t calls){ return std::max(0.0,double(ticks)-double(calls)*double(s.tscCost))/perMicro; };
        const uint64_t frames=s.sampledFrames.exchange(0,relaxed);
        const uint64_t hizCalls=s.hizCalls.exchange(0,relaxed),movingCalls=s.movingCalls.exchange(0,relaxed),previsCalls=s.previsCalls.exchange(0,relaxed);
        const uint64_t movingTicks=s.movingTicks.exchange(0,relaxed),previsTicks=s.previsTicks.exchange(0,relaxed);
        const uint64_t candidates=s.candidateCalls.exchange(0,relaxed),tests=s.testCalls.exchange(0,relaxed);
        uint64_t stageTicks[4];
        for(int i=0;i<4;++i) stageTicks[i]=s.stageTicks[i].exchange(0,relaxed);
        out.sampledFrames=frames;
        if(frames) {
            const uint64_t stageCalls[4]={hizCalls,candidates,candidates,tests};
            out.hizMicros=0.0;
            for(int i=0;i<4;++i) { out.stageMicros[i]=net(stageTicks[i],stageCalls[i])/frames; out.hizMicros+=out.stageMicros[i]; }
            out.candidatesPerFrame=double(candidates)/frames; out.testsPerFrame=double(tests)/frames;
            out.movingMicros=net(movingTicks,movingCalls)/frames;
            out.previsMicros=net(previsTicks,previsCalls)/frames; out.hizCallsPerFrame=double(hizCalls)/frames;
        }
        const uint64_t raceFrames=s.raceFrames.exchange(0,relaxed);
        const uint64_t publishTicks=s.publishTicks.exchange(0,relaxed),firstTestTicks=s.firstTestTicks.exchange(0,relaxed);
        out.raceFrames=raceFrames; out.raceLost=s.raceLost.exchange(0,relaxed);
        if(raceFrames) { out.publishMicros=double(publishTicks)/perMicro/raceFrames; out.firstTestMicros=double(firstTestTicks)/perMicro/raceFrames; }
        return out;
    }
    CaptureCost TakeCaptureCost() {
        auto& s=S();
        CaptureCost out;
        const double micros=std::chrono::duration<double,std::micro>(Clock::now()-s.clockStart).count();
        const double perMicro=s.tscStart && micros>0.0 ? double(__rdtsc()-s.tscStart)/micros : 0.0;
        out.captures=s.captureCalls.exchange(0,relaxed); out.prepares=s.prepareCalls.exchange(0,relaxed); out.builds=s.buildCalls.exchange(0,relaxed);
        const uint64_t captureTicks=s.captureTicks.exchange(0,relaxed),prepareTicks=s.prepareTicks.exchange(0,relaxed),built=s.buildMicrosSum.exchange(0,relaxed);
        if(perMicro>0.0) {
            if(out.captures) out.captureMicros=double(captureTicks)/perMicro/double(out.captures);
            if(out.prepares) out.prepareMicros=double(prepareTicks)/perMicro/double(out.prepares);
        }
        if(out.builds) out.buildMicros=double(built)/double(out.builds);
        double ms=0.0; uint32_t spans=0;
        s.reduceTimer.Take(ms,spans);  // the render thread (VRDepthCulling's census runs there)
        out.gpuSpans=spans;
        if(spans) out.gpuMicros=ms*1000.0/spans;
        return out;
    }

    Stats Read() {
        auto& s=S();
        Stats out;
        out.enabled=s.enabled.load(relaxed); out.calibrated=s.calibrated;
        out.captures=s.publishedCaptures.load(relaxed); out.builds=s.builds.load(relaxed);
        out.fadeTested=s.lastFadeTested.load(relaxed); out.fadeCulled=s.lastFadeCulled.load(relaxed);
        out.geometryTested=s.lastGeometryTested.load(relaxed); out.geometryCulled=s.lastGeometryCulled.load(relaxed);
        out.prepareMicros=s.prepareMicros.load(relaxed); out.gridAge=s.gridAge.load(relaxed); out.landed=s.landed.load(relaxed);
        out.previsCulled=s.lastPrevisCulled.load(relaxed);
        out.referencePrevisCulled=s.lastReferencesCulled.load(relaxed);
        out.buildMicros=s.buildMicros.load(relaxed);
        out.lateBuilds=s.lateBuilds.load(relaxed); out.skippedFrames=s.skippedFrames.load(relaxed);
        out.skippedCaptures=s.skippedCaptures.load(relaxed);
        out.depthWidth=s.depthWidth.load(relaxed); out.depthHeight=s.depthHeight.load(relaxed);
        out.movingExcluded=s.movingExcluded.load(relaxed); out.movingSkipped=s.movingSkipped.load(relaxed);
        out.eyeExempted=s.eyeExempted.load(relaxed);
        out.eyeAtEye=s.eyeAtEye.load(relaxed); out.eyeRect=s.eyeRect.load(relaxed); out.boundsReplaced=s.boundsReplaced.load(relaxed);
        out.untested=s.lastUntested.load(relaxed); out.skipped=s.lastSkipped.load(relaxed); out.fadeKept=s.lastFadeKept.load(relaxed);
        out.yieldSkipped=s.yieldSkipped.load(relaxed);
        out.waits=s.waits.load(relaxed); out.waitTimeouts=s.waitTimeouts.load(relaxed);
        out.fastFrames=s.fastFrames.load(relaxed); out.frameMs=static_cast<float>(s.frameMs);
        out.boxes=s.settings.boxes;
        out.boxRegistered=s.lastBoxRegistered.load(relaxed); out.boxCulled=s.lastBoxCulled.load(relaxed);
        out.boxNoModel=s.lastBoxNoModel.load(relaxed); out.boxEdge=s.lastBoxEdge.load(relaxed); out.boxTableHidden=s.boxTableHidden.load(relaxed);
        out.boxPasses=s.boxPasses.load(relaxed); out.boxReadbacks=s.boxReadbacks.load(relaxed);
        out.boxType=s.lastBoxType.load(relaxed); out.boxWaiting=s.lastBoxWaiting.load(relaxed); out.boxInvalid=s.lastBoxInvalid.load(relaxed);
        {
            auto& reads=Reads();
            out.boxMeshReads=reads.boxes.load(relaxed); out.boxMeshInvalid=reads.invalid.load(relaxed);
            out.boxCpuBoxes=reads.cpuBoxes.load(relaxed); out.boxMeshDropped=reads.dropped.load(relaxed);
            {
                std::lock_guard lock(reads.mutex);
                out.boxQueued=static_cast<uint32_t>(reads.requests.size());
            }
            for(size_t i=0;i<out.boxTypes.size();++i) out.boxTypes[i]=reads.types[i].load(relaxed);
        }
        out.boxShifted=s.boxShifted.load(relaxed); out.boxStale=s.boxStale.load(relaxed);
        return out;
    }
}
