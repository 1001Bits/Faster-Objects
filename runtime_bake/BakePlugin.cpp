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

// The runtime bake's start from the shared settings Data/SKSE/Plugins/SkyrimPrecombines.ini (on unless bRuntimeBake=0,
// or bEnabled=0 outside developer mode): RuntimeBake::Load, called by SkyrimPrecombines.dll at kDataLoaded. With
// PRECOMBINES_BAKE_PLUGIN this file is also the standalone SkyrimPrecombinesBake plugin's entry point (development).
#include "RuntimeBake.h"
#include "Config.h"

namespace RuntimeBake {
    void Load() {
        Settings settings;
        bool precombines=true;
        try {
            // Config's readers, as Precombines.cpp reads the same file: a value of the wrong type keeps that
            // setting's default with one warning (runtime_bake=1 is a number, which value<bool> threw for, turning the whole
            // bake off while the hand-off stayed on). The INI is UTF-8 (Config.h); fs::path from a std::string would read a
            // tool path in the ANSI code page.
            const auto path=[](const char* key) { const auto text=Config::Text(key,std::string()); return std::filesystem::path(std::u8string(text.begin(),text.end())); };
            // On without the INI, as shipped and as Precombines.cpp reads it (the code said off, the INI on).
            settings.enabled=Config::Flag("runtime_bake",true);
            // bEnabled=0 stops the bake too: nothing would show its chunks. Developer mode keeps it, since its
            // toggle key can switch precombines on in play.
            precombines=Config::Flag("enabled",true) || Config::Flag("developer_mode",false);
            settings.tool=path("runtime_bake_tool");
            settings.chunkSize=Config::Number("runtime_bake_chunk_size",4096.f);
            settings.lightBudget=Config::Number("runtime_bake_light_budget",6);
            settings.timeoutSeconds=Config::Number("runtime_bake_timeout",600u);
            settings.firstPartReferences=Config::Number("runtime_bake_first_part",150u);
            settings.partReferences=Config::Number("runtime_bake_part_size",400u);
            settings.prebakeDoors=Config::Flag("runtime_prebake_doors",false);
            settings.doorDistance=Config::Number("runtime_prebake_door_distance",768.f);
            settings.detectLightLimitFix=Config::Flag("runtime_bake_detect_llf",true);
            settings.previs=Config::Flag("runtime_previs",false);
            settings.previsTool=path("runtime_previs_tool");
            settings.previsGpu=Config::Text("runtime_previs_gpu","hardware");
            settings.previsFaceSize=Config::Number("runtime_previs_face_size",128);
            settings.previsGpuDuty=Config::Number("runtime_previs_gpu_duty",0.5);
            settings.previsTimeoutSeconds=Config::Number("runtime_previs_timeout",14400u);
            settings.previsMethod=Config::Text("runtime_previs_method","render");
            settings.previsThreads=std::clamp(Config::Number("runtime_previs_threads",4),1,16);
            settings.previsExterior=Config::Flag("runtime_previs_exterior",false);
            if(settings.previsMethod!="render" && settings.previsMethod!="portals") {
                logger::warn("Unknown runtime_previs_method \"{}\" (render or portals); using render",settings.previsMethod);
                settings.previsMethod="render";
            }
        } catch(const std::exception& e) { logger::error("Runtime bake: unreadable configuration: {}",e.what()); return; }
        if(!settings.enabled || !precombines) { logger::info("Runtime bake off ({})",settings.enabled ? "bEnabled=0" : "bRuntimeBake=0"); return; }
        if(!(settings.chunkSize>=128.f && settings.chunkSize<=4096.f) || settings.lightBudget>7 || !(settings.doorDistance>=0.f && settings.doorDistance<=8192.f)) {
            logger::error("Runtime bake: invalid settings"); return;
        }
        // Folders, threads and tool checks: any failure leaves the bake off for the session. An exception must not reach
        // SKSE's message dispatch, which has no handler (a Data folder the game cannot write ended the game at
        // kDataLoaded).
        try {
            Configure(settings);
            Start();
        } catch(const std::exception& e) { logger::error("Runtime bake disabled: {}",e.what()); }
        catch(...) { logger::error("Runtime bake disabled: unknown error"); }
    }
}

#ifdef PRECOMBINES_BAKE_PLUGIN
namespace {
    void OnMessage(SKSE::MessagingInterface::Message* message) {
        if(message && message->type==SKSE::MessagingInterface::kDataLoaded) RuntimeBake::Load();
    }
}

SKSEPluginLoad(const SKSE::LoadInterface* a_skse) {
    // The runtime plugin's versions only (SupportedRuntime), refused before SKSE::Init loads an address library; the
    // runtime plugin refuses every part on any other version anyway.
    if(!SupportedRuntime()) return false;
    SKSE::Init(a_skse);
    // Every line at once: the bake runs for minutes in the background and a session reads this log while it does.
    if(auto log=spdlog::default_logger()) log->flush_on(spdlog::level::info);
    logger::info("SkyrimPrecombinesBake (runtime-bake prototype) loading");
    const auto messaging=SKSE::GetMessagingInterface();
    return messaging && messaging->RegisterListener(OnMessage);
}
#endif
