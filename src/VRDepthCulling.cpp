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

#include "VRDepthCulling.h"
#include "Config.h"
#include "GpuTimer.h"
#include "Occlusion.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <intrin.h>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

// Every address below was checked in a disassembly of SkyrimVR.exe 1.4.15. The bound fix and the recovery rule follow
// Community Shaders' VR feature (ParticleTroned's fork, src/Features/VR.cpp and VRDepthCullingTemporal*.{h,cpp};
// GPL-3.0-or-later; ParticleTroned, Treatid2, Tim Butler).
namespace VRDepthCulling {
    namespace {
        constexpr std::uintptr_t kEnabled=0x1EC6B88;     // bool bDepthBufferCulling:VRPerformance
        // float fMinOccludeeBoxExtent:VR (0.5): the queue (FUN_141356320) widens every box half-extent below it to it, so
        // a thin object is tested as a thicker box and is hidden only when that box is.
        constexpr std::uintptr_t kMinExtent=0x1ED64E8;
        // The queue's box source (an unnamed setting, default 1): the model bound, widened by the extent; 0: the bound
        // sphere, scaled, without it.
        constexpr std::uintptr_t kModelBoxes=0x1ED64D0;
        // BSGeometry::CopyTransformAndBounds (SE ID 69528): copies the source's bound sphere (+0x138..+0x144), then its
        // model-bound centre (+0x148..+0x150) from the sphere's offsets again, so a clone's box is centred on its sphere.
        constexpr std::uintptr_t kCopyBounds=0xCB6F60;
        constexpr std::array<std::uint32_t,3> kCopyLoads{0xD9,0xE5,0xF1};  // mov eax,[rsi+disp32]; its store follows
        // In the culling setup (FUN_141321d80, +0x30B): the manager's readback of the previous frame's results; the
        // engine calls it only while the culling is on.
        constexpr std::uintptr_t kReadbackCall=0x132208B;
        // Main::RenderDepth +0x3B1: the manager's GPU test of this frame's boxes against the depth prepass.
        constexpr std::uintptr_t kRenderCall=0x1323601;
        // Both sites load the manager (0x1436F1870) into rcx just before the call.
        constexpr std::array<std::uint8_t,7> kReadbackLoad{0x48,0x8B,0x0D,0xE5,0xF7,0x3C,0x02};
        constexpr std::array<std::uint8_t,7> kRenderLoad{0x48,0x8B,0x0D,0x6F,0xE2,0x3C,0x02};
        // The manager: queued boxes (count +0xB0; +0xB8: one 4x4 matrix per box, its columns the half-extent axes x, y, z
        // and the centre), which of the two result buffers is current (+0xC0) and those buffers (+0xD0, one uint32 per
        // box, 0: hidden). The readback copies the GPU results into the current buffer.
        constexpr std::size_t kCount=0xB0,kBoxes=0xB8,kSelector=0xC0,kResults=0xD0;
        constexpr std::uint32_t kMaxBoxes=0x1000;
        // +0xB4: geometries the main view's culler (BSGeometryListCullingProcess, VR vfunc DA1860) left out because their
        // box was hidden. At the readback both counts are still the previous frame's; the engine resets them after it.
        constexpr std::size_t kCulled=0xB4;
        // Diagnostics only ("vr_depth_culling_log"): the culler's call registering one geometry's box (DA1860 +0xF7 ->
        // FUN_1413560a0, culling threads) and the box upload (FUN_1413562d0, from the culling setup +0x37E and
        // RenderDepth +0x11A), timed for the census.
        constexpr std::uintptr_t kRegisterCall=0xDA1957,kRegister=0x13560A0,kUpload=0x13562D0;
        constexpr std::array<std::uintptr_t,2> kUploadCalls{0x13220FE,0x132336A};

        // flush (experiment, "vr_depth_culling_flush"): submit the queued GPU commands right after the box test, so the
        // next frame's readback does not wait for commands the driver still holds. flushAb alternates it per census line.
        struct Settings { bool enabled=true,recovery=true,log=false,flush=false,flushAb=false; float minExtent=10.f; std::uint32_t budget=64,logFrames=5400; };
        Settings settings;
        bool installed=false;
        std::optional<float> userExtent;

        // The view the latest GPU test was rendered from. Both hooks run on the render thread, readback before render.
        struct Pose { RE::NiMatrix3 rotate; RE::NiPoint3 translate; bool valid=false; };
        Pose tested;
        std::atomic<std::uint64_t> readbacks{0},recoveries{0},restored{0};

        // The census ("vr_depth_culling_log"): per frame, what the game's depth culling and this plugin's Hi-Z each test
        // and hide, and what each costs, averaged over logFrames frames into one log line. Render thread, except the
        // registration slots (culling threads, one cache line each).
        struct Census {
            std::uint64_t frames=0,nativeFrames=0,registered=0,culled=0;
            std::uint64_t readbackTicks=0,renderTicks=0,recoverTicks=0,uploadTicks=0,uploads=0;
            std::uint64_t fadeTested=0,fadeCulled=0,geometryTested=0,geometryCulled=0;
            bool readThisFrame=false;
            std::uint64_t pendingRegistered=0,pendingCulled=0;
            double frameMs=0.0;  // sum of the intervals between frames
        };
        std::chrono::steady_clock::time_point lastFrame{};
        Census census;
        struct alignas(64) RegisterSlot { std::atomic<std::uint64_t> ticks{0},calls{0}; };
        std::array<RegisterSlot,32> registerSlots;
        std::atomic<std::uint32_t> registerThreads{0};
        GpuTimer testTimer;  // render thread
        std::uint64_t tscStart=0;
        std::chrono::steady_clock::time_point clockStart;

        std::uintptr_t At(std::uintptr_t rva) { return REL::Module::get().base()+rva; }
        // A trampoline of its own: on VR the Hi-Z hooks (Occlusion::Install) already hold CommonLib's global one, and a
        // second create() on that replaces it and frees the memory their jumps live in. Never destroyed (the engine can
        // still call through it while the process exits).
        SKSE::Trampoline& OwnTrampoline() { static auto* trampoline=new SKSE::Trampoline("VRDepthCulling"); return *trampoline; }
        template<std::size_t N>
        bool Bytes(std::uintptr_t at,const std::array<std::uint8_t,N>& expected) { return std::memcmp(reinterpret_cast<const void*>(at),expected.data(),N)==0; }

        // One value as the game's INI reader sees it: the last `key` in `section`, nothing when absent.
        std::optional<std::string> IniValue(const std::filesystem::path& path,std::string_view section,std::string_view key) {
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
            std::optional<std::string> out;
            std::string line;
            bool inSection=false;
            while(std::getline(in,line)) {
                const auto t=trim(line);
                if(t.empty() || t.front()==';' || t.front()=='#') continue;
                if(t.front()=='[') { inSection=t.back()==']' && same(trim(t.substr(1,t.size()-2)),section); continue; }
                const auto equals=t.find('=');
                if(inSection && equals!=std::string_view::npos && same(trim(t.substr(0,equals)),key)) out=std::string(trim(t.substr(equals+1)));
            }
            return out;
        }
        // A player's own choice in SkyrimVR.ini or SkyrimPrefs.ini (no shipped INI names these keys); SkyrimPrefs.ini
        // is read last.
        std::optional<std::string> UserIni(std::string_view section,std::string_view key) {
            const auto logs=SKSE::log::log_directory();  // Documents/My Games/Skyrim VR/SKSE
            if(!logs) return std::nullopt;
            std::optional<std::string> out;
            for(const char* name:{"SkyrimVR.ini","SkyrimPrefs.ini"})
                if(auto value=IniValue(logs->parent_path()/name,section,key)) out=std::move(value);
            return out;
        }

        void ReadSettings() {
            const auto& config=Config::Get();  // SkyrimPrecombines.ini
            const auto number=[&](const char* key,float fallback) { const auto it=config.find(key); return it!=config.end() && it->is_number() ? it->get<float>() : fallback; };
            const auto flag=[&](const char* key,bool fallback) { const auto it=config.find(key); return it!=config.end() && it->is_boolean() ? it->get<bool>() : fallback; };
            settings.enabled=flag("vr_depth_culling",true);
            settings.recovery=flag("vr_depth_culling_recovery",true);
            settings.log=flag("vr_depth_culling_log",false);
            const float extent=number("vr_min_occludee_extent",10.f);
            settings.minExtent=std::isfinite(extent) ? std::clamp(extent,0.f,1000.f) : 10.f;
            settings.budget=static_cast<std::uint32_t>(std::clamp(number("vr_depth_culling_budget",64.f),0.f,1024.f));
            settings.logFrames=static_cast<std::uint32_t>(std::clamp(number("vr_depth_culling_log_frames",5400.f),30.f,1e6f));
            settings.flush=flag("vr_depth_culling_flush",false);
            settings.flushAb=flag("vr_depth_culling_flush_ab",false);
        }

        // The box corners are centre +-x +-y +-z; the farthest pair from the centre bounds them all.
        bool BoxSphere(const float (&m)[4][4],RE::NiPoint3& centre,float& radius) {
            double best=0.0;
            for(std::uint32_t corner=0;corner<4;++corner) {
                const double sy=corner&1 ? 1.0:-1.0,sz=corner&2 ? 1.0:-1.0;
                double d2=0.0;
                for(int k=0;k<3;++k) { const double v=double(m[k][0])+sy*m[k][1]+sz*m[k][2]; d2+=v*v; }
                best=std::max(best,d2);
            }
            centre={m[0][3],m[1][3],m[2][3]};
            radius=static_cast<float>(std::sqrt(best));
            return std::isfinite(best) && std::isfinite(radius) && std::isfinite(centre.x) && std::isfinite(centre.y) && std::isfinite(centre.z);
        }

        // The results read back now come from a test rendered one frame ago. When the view has turned or moved since,
        // an object the test found hidden can be in view now with nothing in front of it; it would stay missing until
        // the next result. Restore up to `budget` of them: those whose bound, grown by how far the view moved, reaches
        // into the frustum, nearest to directly visible and largest on screen first.
        void Recover(std::byte* manager) {
            readbacks.fetch_add(1,std::memory_order_relaxed);
            auto* camera=RE::Main::WorldRootCamera();
            if(!tested.valid || !camera || !settings.budget) return;
            const auto& view=camera->world;
            double trace=0.0;
            for(int r=0;r<3;++r) for(int c=0;c<3;++c) trace+=double(tested.rotate.entry[r][c])*view.rotate.entry[r][c];
            const double cosine=std::clamp((trace-1.0)*0.5,-1.0,1.0);
            const double moved=std::sqrt(double((view.translate-tested.translate).SqrLength()));
            // Coherent: under 0.05 degrees and 0.1 units, as in Community Shaders.
            if(!std::isfinite(cosine) || !std::isfinite(moved) || (cosine>=0.9999996192282494 && moved<=0.1)) return;
            const double chord=2.0*std::sqrt(std::max(0.0,(1.0-cosine)*0.5));

            std::uint32_t count,selector;
            std::memcpy(&count,manager+kCount,4); std::memcpy(&selector,manager+kSelector,4);
            if(!count || count>kMaxBoxes || selector>1) return;
            const float (*boxes)[4][4];
            std::uint32_t* results;
            std::memcpy(&boxes,manager+kBoxes,8); std::memcpy(&results,manager+kResults+selector*8,8);
            if(!boxes || !results) return;
            recoveries.fetch_add(1,std::memory_order_relaxed);

            struct Candidate { std::uint32_t index; float score; bool visible; };
            const auto before=[](const Candidate& a,const Candidate& b) {
                if(a.visible!=b.visible) return a.visible;
                if(a.score!=b.score) return a.score>b.score;
                return a.index<b.index;
            };
            static std::vector<Candidate> heap;  // the best `budget` so far, worst at the front (render thread only)
            heap.clear();
            heap.reserve(settings.budget);
            for(std::uint32_t i=0;i<count;++i) {
                if(results[i]) continue;
                RE::NiPoint3 centre; float radius;
                if(!BoxSphere(boxes[i],centre,radius)) continue;
                const double distance=std::sqrt(double((centre-view.translate).SqrLength()));
                const float grown=static_cast<float>(radius+moved+(distance+moved)*chord);
                if(!std::isfinite(grown)) continue;
                const bool visible=camera->PointInFrustum(centre,radius);
                if(!visible && !camera->PointInFrustum(centre,grown)) continue;
                const Candidate candidate{i,static_cast<float>(double(radius)*radius/std::max(distance*distance,1.0)),visible};
                if(heap.size()<settings.budget) { heap.push_back(candidate); std::push_heap(heap.begin(),heap.end(),before); }
                else if(before(candidate,heap.front())) {
                    std::pop_heap(heap.begin(),heap.end(),before);
                    heap.back()=candidate;
                    std::push_heap(heap.begin(),heap.end(),before);
                }
            }
            for(const auto& c:heap) results[c.index]=1;
            restored.fetch_add(heap.size(),std::memory_order_relaxed);
        }

        struct ReadbackHook {
            static void thunk(std::byte* manager) {
                if(!settings.log) {
                    func(manager);
                    if(settings.recovery && manager) Recover(manager);
                    return;
                }
                if(manager) {
                    std::uint32_t registered,culled;
                    std::memcpy(&registered,manager+kCount,4); std::memcpy(&culled,manager+kCulled,4);
                    census.pendingRegistered=registered; census.pendingCulled=culled; census.readThisFrame=true;
                }
                const std::uint64_t begin=__rdtsc();
                func(manager);
                const std::uint64_t read=__rdtsc();
                if(settings.recovery && manager) Recover(manager);
                census.readbackTicks+=read-begin; census.recoverTicks+=__rdtsc()-read;
            }
            static inline REL::Relocation<decltype(thunk)> func;
        };
        bool flushing=false;  // render thread: the flush experiment's current state
        struct RenderHook {
            static void thunk(void* manager,std::uint32_t pass) {
                auto* renderer=settings.log || flushing ? RE::BSGraphics::Renderer::GetSingleton() : nullptr;
                auto* context=renderer ? reinterpret_cast<ID3D11DeviceContext*>(renderer->GetRuntimeData().context) : nullptr;
                if(settings.log) {
                    auto* device=renderer ? reinterpret_cast<ID3D11Device*>(renderer->GetRuntimeData().forwarder) : nullptr;
                    if(device && context) testTimer.Begin(device,context);
                    const std::uint64_t begin=__rdtsc();
                    func(manager,pass);
                    if(device && context) testTimer.End(context);
                    // The pass ends by copying the results into the staging buffer the next frame maps.
                    if(flushing && context) context->Flush();
                    census.renderTicks+=__rdtsc()-begin;
                } else {
                    func(manager,pass);
                    if(flushing && context) context->Flush();
                }
                if(auto* camera=RE::Main::WorldRootCamera()) tested={camera->world.rotate,camera->world.translate,true};
                else tested.valid=false;
            }
            static inline REL::Relocation<decltype(thunk)> func;
        };
        // Census only: each box registration, timed on its culling thread.
        struct RegisterHook {
            static std::uintptr_t thunk(std::uintptr_t manager,std::uintptr_t geometry,std::uintptr_t a3,std::uintptr_t a4) {
                thread_local RegisterSlot* slot=&registerSlots[registerThreads.fetch_add(1,std::memory_order_relaxed)%registerSlots.size()];
                const std::uint64_t begin=__rdtsc();
                const auto result=func(manager,geometry,a3,a4);
                slot->ticks.store(slot->ticks.load(std::memory_order_relaxed)+(__rdtsc()-begin),std::memory_order_relaxed);
                slot->calls.store(slot->calls.load(std::memory_order_relaxed)+1,std::memory_order_relaxed);
                return result;
            }
            static inline REL::Relocation<decltype(thunk)> func;
        };
        template<int Site> struct UploadHook {
            static void thunk(std::byte* manager) {
                const std::uint64_t begin=__rdtsc();
                func(manager);
                census.uploadTicks+=__rdtsc()-begin; ++census.uploads;
            }
            static inline REL::Relocation<decltype(thunk)> func;
        };

        // A census hook goes in only where the call is the expected one (the engine function, not another plugin's).
        template<class Hook>
        bool HookEngineCall(std::uintptr_t site,std::uintptr_t target) {
            const auto* bytes=reinterpret_cast<const std::uint8_t*>(At(site));
            std::int32_t displacement;
            std::memcpy(&displacement,bytes+1,4);
            if(bytes[0]!=0xE8 || At(site)+5+displacement!=At(target)) {
                logger::warn("VR depth culling census: the call at {:X} is not the expected engine call; not timed",site);
                return false;
            }
            Hook::func=OwnTrampoline().write_call<5>(At(site),Hook::thunk);
            return true;
        }

        void LogCensus() {
            auto& c=census;
            const double micros=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-clockStart).count();
            const double perMicro=micros>0.0 ? double(__rdtsc()-tscStart)/micros : 0.0;
            const auto us=[&](std::uint64_t ticks,std::uint64_t n) { return perMicro>0.0 && n ? double(ticks)/perMicro/double(n) : 0.0; };
            std::uint64_t registerTicks=0,registerCalls=0;
            for(auto& slot:registerSlots) { registerTicks+=slot.ticks.exchange(0,std::memory_order_relaxed); registerCalls+=slot.calls.exchange(0,std::memory_order_relaxed); }
            double gpuMs=0.0; std::uint32_t gpuSpans=0;
            testTimer.Take(gpuMs,gpuSpans);
            const auto f=double(c.frames),nf=double(std::max<std::uint64_t>(c.nativeFrames,1));
            const auto capture=Occlusion::TakeCaptureCost();
            const auto cost=Occlusion::ReadCost();
            logger::info("VR cull census, {} frames ({} with native results): native {:.0f} boxes tested, {:.0f} culled per frame; CPU per native frame "
                         "readback {:.1f} us, test submission {:.1f} us, box upload {:.1f} us, registration {:.1f} us ({:.0f} calls), recovery {:.1f} us; "
                         "GPU box test {:.1f} us ({} frames) | Hi-Z fade nodes {:.1f} tested, {:.1f} culled, geometry {:.1f} tested, {:.1f} culled per frame; "
                         "CPU capture {:.1f} us ({}), prepare {:.1f} us ({}), helper build {:.1f} us ({}), culling tests {:.1f} us per sampled frame ({} sampled), "
                         "GPU reduction {:.1f} us ({} frames); flush {}; frame {:.2f} ms",
                c.frames,c.nativeFrames,double(c.registered)/nf,double(c.culled)/nf,us(c.readbackTicks,c.nativeFrames),us(c.renderTicks,c.nativeFrames),
                us(c.uploadTicks,c.nativeFrames),us(registerTicks,c.nativeFrames),double(registerCalls)/nf,us(c.recoverTicks,c.nativeFrames),
                gpuSpans ? gpuMs*1000.0/gpuSpans : 0.0,gpuSpans,f>0 ? double(c.fadeTested)/f : 0.0,f>0 ? double(c.fadeCulled)/f : 0.0,
                f>0 ? double(c.geometryTested)/f : 0.0,f>0 ? double(c.geometryCulled)/f : 0.0,capture.captureMicros,capture.captures,capture.prepareMicros,
                capture.prepares,capture.buildMicros,capture.builds,cost.hizMicros,cost.sampledFrames,capture.gpuMicros,capture.gpuSpans,
                flushing ? "on":"off",f>0 ? c.frameMs/f : 0.0);
            c=Census{};
            // The flush experiment alternates per line, so both states share the same scene and conditions.
            if(settings.flushAb) flushing=!flushing;
        }

        // Copy the model-bound centre from its own offsets: the disp32 of each load becomes the offset its store writes.
        bool FixBoundCopy() {
            const auto fn=At(kCopyBounds);
            for(const auto load:kCopyLoads) {
                const std::uint32_t to=0x148+(load-0xD9)/3;  // 0x148, 0x14C, 0x150
                const std::array<std::uint8_t,6> store{0x89,0x87,std::uint8_t(to),std::uint8_t(to>>8),0,0};
                const std::array<std::uint8_t,2> op{0x8B,0x86};
                std::uint32_t from;
                std::memcpy(&from,reinterpret_cast<const void*>(fn+load+2),4);
                if(!Bytes(fn+load,op) || !Bytes(fn+load+6,store) || (from!=to-0x10 && from!=to)) {
                    logger::warn("VR depth culling: BSGeometry::CopyTransformAndBounds +0x{:X} is not the expected code; bound fix not applied",load);
                    return false;
                }
            }
            for(const auto load:kCopyLoads) REL::safe_write(fn+load+2,std::uint32_t(0x148+(load-0xD9)/3));
            return true;
        }
    }

    void Install() {
        if(!REL::Module::IsVR() || !SupportedRuntime()) return;
        // Community Shaders (its VR feature, any fork) sets these values every frame and applies the same bound fix.
        if(REX::W32::GetModuleHandleW(L"CommunityShaders.dll")) { logger::info("VR depth culling: Community Shaders manages it; left to it"); return; }
        ReadSettings();
        if(!settings.enabled) { logger::info("VR depth culling: off (bDepthCulling=0)"); return; }
        if(const auto user=UserIni("VRPerformance","bDepthBufferCulling")) {
            const char c=user->empty() ? '\0' : static_cast<char>(std::tolower(static_cast<unsigned char>((*user)[0])));
            if(c=='0' || c=='f' || c=='n') { logger::info("VR depth culling: off (bDepthBufferCulling={} in the game's INI)",*user); return; }
        }
        if(const auto user=UserIni("VR","fMinOccludeeBoxExtent")) {
            char* end=nullptr;
            const float value=std::strtof(user->c_str(),&end);
            if(end!=user->c_str() && std::isfinite(value)) userExtent=value;
        }
        const bool fixed=FixBoundCopy();
        bool hooked=false;
        tscStart=__rdtsc(); clockStart=std::chrono::steady_clock::now();
        flushing=settings.flush;
        if(settings.recovery || settings.log) {
            if(*reinterpret_cast<const std::uint8_t*>(At(kReadbackCall))==0xE8 && *reinterpret_cast<const std::uint8_t*>(At(kRenderCall))==0xE8 &&
               Bytes(At(kReadbackCall)-7,kReadbackLoad) && Bytes(At(kRenderCall)-7,kRenderLoad)) {
                auto& trampoline=OwnTrampoline();
                trampoline.create(settings.log ? 128 : 32);
                // Each call is chained to whatever it calls now: the engine or a plugin that hooked it first.
                ReadbackHook::func=trampoline.write_call<5>(At(kReadbackCall),ReadbackHook::thunk);
                RenderHook::func=trampoline.write_call<5>(At(kRenderCall),RenderHook::thunk);
                hooked=true;
                if(settings.log) {
                    const bool timed=HookEngineCall<RegisterHook>(kRegisterCall,kRegister) & HookEngineCall<UploadHook<0>>(kUploadCalls[0],kUpload) &
                                     HookEngineCall<UploadHook<1>>(kUploadCalls[1],kUpload);
                    logger::info("VR depth culling census on: one line every {} frames{}",settings.logFrames,timed ? "" : " (some engine calls not timed)");
                }
            } else logger::warn("VR depth culling: a readback or test call site is not the expected code; view-motion recovery not installed");
        }
        installed=true;
        logger::info("VR depth culling: bound fix {}, view-motion recovery {} (up to {} objects a frame)",fixed ? "applied":"not applied",
            hooked && settings.recovery ? "installed":"off",settings.budget);
    }

    void OnFrame() {
        if(!installed || !settings.log) return;
        auto& c=census;
        ++c.frames;
        const auto now=std::chrono::steady_clock::now();
        if(lastFrame.time_since_epoch().count()) c.frameMs+=std::chrono::duration<double,std::milli>(now-lastFrame).count();
        lastFrame=now;
        if(c.readThisFrame) { ++c.nativeFrames; c.registered+=c.pendingRegistered; c.culled+=c.pendingCulled; c.readThisFrame=false; }
        const auto hiz=Occlusion::Read();
        c.fadeTested+=hiz.fadeTested; c.fadeCulled+=hiz.fadeCulled; c.geometryTested+=hiz.geometryTested; c.geometryCulled+=hiz.geometryCulled;
        if(c.frames>=settings.logFrames) LogCensus();
    }

    void Apply() {
        if(!installed) return;
        auto* enabled=reinterpret_cast<bool*>(At(kEnabled));
        auto* extent=reinterpret_cast<float*>(At(kMinExtent));
        const bool was=*enabled;
        const float previous=*extent;
        *enabled=true;
        if(!userExtent) *extent=settings.minExtent;
        logger::info("VR depth culling ON (was {}); boxes from the {}, half-extents widened to at least {:.1f} ({}); ToggleDepthBufferCulling in the console switches it",
            was ? "on":"off",*reinterpret_cast<const std::uint8_t*>(At(kModelBoxes)) ? "model bound":"bound sphere",*extent,
            userExtent ? "the game's INI" : std::format("was {:.1f}",previous));
    }
}
