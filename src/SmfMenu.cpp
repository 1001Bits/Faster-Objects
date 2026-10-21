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

#include "SmfMenu.h"
#include "Precombines.h"
#include "Settings.h"

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <format>
#include <string>
#include <string_view>

// SKSE Menu Framework (Nexus SE 120352, QTR-Modding/SKSE-Menu-Framework-3; version 2 has the same exports) draws its Mod
// Control Panel with Dear ImGui and exports the cimgui functions (igCheckbox, ...) next to its own (AddSectionItem). The
// few used here are looked up by name rather than through the framework's header, whose ImGui wrappers call whatever
// GetProcAddress returns: nothing is linked, a missing DLL or export only leaves the page out, and no pointer that was
// not found is ever called. Only export names and their C signatures are declared (cimgui, Dear ImGui 1.90).
//
// The page shows Settings: each frame it reads it, and a change goes to Settings at once (live options apply, the INI is
// written in the background). The framework calls a page once per frame, inside its ImGui frame on the game's render
// thread and only while that page is open, so the page takes nothing but the Settings lock (and reads Precombines::Failed,
// an atomic), never stateMutex, and does no file I/O. The page is named after the mod ("Faster Objects"); the DLL, INI and log keep the SkyrimPrecombines name.
namespace SmfMenu {
    namespace {
        using RenderFunction=void(__stdcall*)();
        using TextV=void(*)(const char*,va_list);

        // Every flags argument is 0, because flag values differ between ImGui versions: Settings clamps what a slider's
        // Ctrl+click entry gives, and the text field is committed when it loses focus.
        struct Api {
            void(*AddSectionItem)(const char* path,RenderFunction render)=nullptr;
            bool(*Checkbox)(const char* label,bool* value)=nullptr;
            bool(*SliderFloat)(const char* label,float* value,float min,float max,const char* format,int flags)=nullptr;
            bool(*InputText)(const char* label,char* buffer,std::size_t size,int flags,void* callback,void* user)=nullptr;
            bool(*IsItemActive)()=nullptr;
            bool(*IsItemDeactivatedAfterEdit)()=nullptr;
            TextV SetItemTooltipV=nullptr;
            void(*TextUnformatted)(const char* text,const char* end)=nullptr;
            TextV TextWrappedV=nullptr;
            TextV TextDisabledV=nullptr;
            void(*SeparatorText)(const char* label)=nullptr;
            void(*Spacing)()=nullptr;
        };
        Api api;  // complete before the page is added (Register), read only by its callback

        template<class F> bool Resolve(REX::W32::HMODULE module,const char* name,F& out) {
            out=reinterpret_cast<F>(REX::W32::GetProcAddress(module,name));
            if(!out) logger::info("SKSE Menu Framework has no {}; no menu page",name);
            return out!=nullptr;
        }

        // ImGui's text functions take a printf format; text always goes in as its argument, so a '%' in it is shown.
        void CallV(TextV f,const char* fmt,...) { va_list a; va_start(a,fmt); f(fmt,a); va_end(a); }
        void Tooltip(const char* help) { CallV(api.SetItemTooltipV,"%s",help); }

        // A setting's row: its shipped INI name (Settings' id), label and help, and what its status line (when
        // Settings::Status has one) starts with.
        struct Row { const char* name; const char* label; const char* help; const char* status="Status"; };
        // On is live, with one limit: the runtime bake reads bEnabled once, at startup (Precombines ReadSettings,
        // BakePlugin), so in a game started with it off (bRuntimeBake=1, no developer_mode) switching it on shows the cells
        // merged before (Precombines ResumeRuntimeHandOff) and merges new ones only after a restart; off in play leaves the
        // bake running until then. Settings::Status names both.
        constexpr Row kEnabled{"bEnabled","Precombined meshes",
            "Draws each cell's static objects as a few merged meshes, for fewer draw calls. Applies at once. Off: the original objects come back at "
            "once, and merging new cells stops after a restart. If the game started with it off, cells not merged before are merged only after a "
            "restart.","Precombines status"};
        // bRuntimeBake=0 also keeps the cells merged before from being read (Precombines ReadRuntimeParts), and the release
        // ships none, so it switches off what precombines draw.
        constexpr Row kRuntimeBake{"bRuntimeBake","Runtime bake: merge and use cells (restart)",
            "Merges each cell in the background the first time you visit it, from your own load order and meshes, and uses the cells merged before. "
            "Off: no cell is merged or used, so precombined meshes have nothing to draw. Applies after you restart the game."};
        constexpr Row kCacheFolder{"sCacheFolder","Cache folder (restart)",
            "Where merged cells are kept. Empty: the game's Data folder (with Mod Organizer 2, its overwrite folder). Or a full path outside the game, "
            "such as C:\\Games\\SkyrimPrecombinesCache. Applies after you restart the game."};
        // On SE and AE a game started with bHiZ=0 may run without the Hi-Z hooks (main.cpp HooksUnneeded): turning it on
        // then waits for a restart, which Settings::Status says.
        constexpr Row kHiZ{"bHiZ","Hi-Z occlusion culling",
            "Objects completely hidden behind walls, rocks and buildings are not drawn. Applies immediately, or after a restart if the game started "
            "with it off.","Hi-Z status"};
        constexpr Row kMinFrameMs{"fHiZMinFrameMs","Minimum frame time",
            "Hi-Z works only while a frame takes at least this long (5 ms is about 200 FPS), where it pays off. 0: always. Applies immediately."};
        constexpr Row kMinYield{"fHiZMinYield","Minimum share hidden",
            "Hi-Z tests every frame only while it hides at least this share of the objects it tests (0.010 = 1%); otherwise it tests a few frames now "
            "and then. 0: always. Applies immediately."};
        // VRDepthCulling::Install: Community Shaders, or bDepthBufferCulling=0 in the player's SkyrimVR.ini, leaves the game's
        // culling alone, and an fMinOccludeeBoxExtent there wins over the extent.
        constexpr Row kDepthCulling{"bDepthCulling","Game's occlusion culling (restart)",
            "Skyrim VR's own occlusion culling (bDepthBufferCulling), which the game ships switched off. With Community Shaders installed, Community "
            "Shaders manages it instead; bDepthBufferCulling=0 under [VRPerformance] in SkyrimVR.ini keeps it off. Applies after you restart the game."};
        constexpr Row kMinExtent{"fMinOccludeeExtent","Smallest tested object (restart)",
            "Each tested box is at least twice this thick (the value is a half-size; default 10), so thin objects are hidden less often. Not used when "
            "SkyrimVR.ini sets fMinOccludeeBoxExtent, or with Community Shaders. Applies after you restart the game."};

        // "<label>##<name>": the label shown and an ID unique on the page, without an allocation each frame.
        struct Id {
            char text[128];
            explicit Id(const Row& row) { *std::format_to_n(text,sizeof text-1,"{}##{}",row.label,row.name).out='\0'; }
        };

        // Under a row, what Settings::Status says about it now (a change that waits for a restart, Hi-Z hooks missing);
        // nothing when it says nothing.
        void Note(const Row& row) {
            const auto status=Settings::Status(row.name);
            if(status.empty()) return;
            CallV(api.TextDisabledV,"%s",std::format("{}: {}",row.status,status).c_str());
        }

        // Rows are drawn only for what has an effect on this game and install (Settings::Available). Precombines that switched
        // themselves off after an error show unticked, as the game has them (bEnabled stays on), so one click retries (on
        // clears the error) and none writes bEnabled=0.
        void Toggle(const Row& row) {
            if(!Settings::Available(row.name)) return;
            bool value=Settings::GetBool(row.name) && !(&row==&kEnabled && Precombines::Failed());
            if(api.Checkbox(Id(row).text,&value)) Settings::SetBool(row.name,value);
            Tooltip(row.help);
            Note(row);
        }

        // Set on every frame the slider moves, so a live gate follows the drag; the INI is written 300 ms after the last.
        void Slider(const Row& row,float min,float max,const char* format) {
            if(!Settings::Available(row.name)) return;
            float value=Settings::GetFloat(row.name);
            if(api.SliderFloat(Id(row).text,&value,min,max,format,0)) Settings::SetFloat(row.name,value);
            Tooltip(row.help);
            Note(row);
        }

        // ImGui keeps its own copy of the text while the field has focus, and the change is committed when it loses it.
        // Otherwise the field shows the stored value, loaded again whenever a setting changed (also the text Settings
        // cleaned up), after a refused change, and when the focus went without a commit (the panel closed).
        void CacheFolder(const Row& row) {
            if(!Settings::Available(row.name)) return;
            static char buffer[1025]{};  // Settings refuses more than 1024 bytes, so a stored value always fits
            static std::uint64_t loaded=~0ull;
            static bool editing=false;
            if(!editing && Settings::Generation()!=loaded) {
                loaded=Settings::Generation();  // first: a change between the two reads loads again next frame
                const auto text=Settings::GetString(row.name);
                const auto size=std::min(text.size(),sizeof buffer-1);
                std::memcpy(buffer,text.data(),size);
                buffer[size]='\0';
            }
            api.InputText(Id(row).text,buffer,sizeof buffer,0,nullptr,nullptr);
            const bool active=api.IsItemActive();
            if(api.IsItemDeactivatedAfterEdit()) { if(!Settings::SetString(row.name,buffer)) loaded=~0ull; }
            else if(editing && !active) loaded=~0ull;
            editing=active;
            Tooltip(row.help);
            Note(row);
        }

        // A page that fails leaves out the rest of that frame's page; it is logged once, and nothing reaches the framework.
        void Failed(const char* what) {
            static bool logged=false;  // the render thread only
            if(logged) return;
            logged=true;
            logger::warn("SKSE Menu Framework page: {}; the rest of the page is left out while this happens",what);
        }

        void __stdcall RenderSettings() {
            try {
                static const std::string intro=
                    std::format("Faster Objects {}. Changes are saved to Data\\SKSE\\Plugins\\SkyrimPrecombines.ini.",Settings::Version());
                CallV(api.TextDisabledV,"%s",intro.c_str());
                if(Settings::SaveFailed())
                    CallV(api.TextWrappedV,"%s","Not saved: SkyrimPrecombines.ini could not be written (see SkyrimPrecombines.log). The changes last until you "
                        "quit; the next change tries again.");
                api.SeparatorText("Precombines");
                Toggle(kEnabled);
                Toggle(kRuntimeBake);
                CacheFolder(kCacheFolder);
                // Settings::Available decides: hidden where Hi-Z cannot run in this game (Skyrim VR without hiz_vr); on SE and
                // AE without the hooks at startup the rows stay, and the status line names the restart.
                if(Settings::Available(kHiZ.name)) {
                    api.SeparatorText("Occlusion culling");
                    Toggle(kHiZ);
                    Slider(kMinFrameMs,0.f,20.f,"%.1f ms");
                    Slider(kMinYield,0.f,0.1f,"%.3f");
                }
                if(REL::Module::IsVR()) {
                    api.SeparatorText("Skyrim VR");
                    Toggle(kDepthCulling);
                    Slider(kMinExtent,0.f,100.f,"%.0f");
                }
                api.Spacing();
                if(Settings::RestartPending()) CallV(api.TextWrappedV,"%s","Restart the game to apply the changes marked (restart) or noted above.");
            } catch(const std::exception& e) { Failed(e.what()); } catch(...) { Failed("unknown error"); }
        }
    }

    void Register() {
        try {
            const auto module=REX::W32::GetModuleHandleW(L"SKSEMenuFramework.dll");
            if(!module) { logger::info("SKSE Menu Framework not loaded; no menu page"); return; }
            Api found;
            const bool complete=Resolve(module,"AddSectionItem",found.AddSectionItem) && Resolve(module,"igCheckbox",found.Checkbox) &&
                Resolve(module,"igSliderFloat",found.SliderFloat) && Resolve(module,"igInputText",found.InputText) &&
                Resolve(module,"igIsItemActive",found.IsItemActive) && Resolve(module,"igIsItemDeactivatedAfterEdit",found.IsItemDeactivatedAfterEdit) &&
                Resolve(module,"igSetItemTooltipV",found.SetItemTooltipV) && Resolve(module,"igTextUnformatted",found.TextUnformatted) &&
                Resolve(module,"igTextWrappedV",found.TextWrappedV) && Resolve(module,"igTextDisabledV",found.TextDisabledV) &&
                Resolve(module,"igSeparatorText",found.SeparatorText) && Resolve(module,"igSpacing",found.Spacing);
            if(!complete) return;
            api=found;
            // Once, at kPostLoad: the framework's menu tree has no lock, and nothing renders yet. The framework splits the
            // path at '/' (section, then page), so the names hold no other.
            api.AddSectionItem("Faster Objects/Settings",RenderSettings);
            logger::info("SKSE Menu Framework: Faster Objects page added");
        } catch(const std::exception& e) {
            logger::warn("SKSE Menu Framework page not added: {}",e.what());
        } catch(...) {
            logger::warn("SKSE Menu Framework page not added: unknown error");
        }
    }
}
