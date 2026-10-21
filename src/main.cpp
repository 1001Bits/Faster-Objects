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

#include "PCH.h"
#include "Config.h"
#include "Precombines.h"
#include "Occlusion.h"
#include "VRDepthCulling.h"
#include "RuntimeBake.h"
#include "Settings.h"
#include "SmfMenu.h"

namespace
{
    void MessageHandler(SKSE::MessagingInterface::Message* a_message)
    {
        // SKSE's dispatcher has no handler, so an exception from here would end the game (an unwritable Data
        // folder threw out of the runtime bake's start at kDataLoaded). Each step guards itself; this is the last line.
        try {
            switch (a_message->type) {
            case SKSE::MessagingInterface::kPostLoad:
                // Every plugin is loaded now, so Community Shaders (which manages VR depth culling itself) can be seen.
                VRDepthCulling::Install();
                // SKSE Menu Framework (optional) is loaded by now if it is installed; its menu tree is filled before any frame.
                SmfMenu::Register();
                break;
            case SKSE::MessagingInterface::kDataLoaded:
                logger::info("Game data loaded");
                VRDepthCulling::Apply();
                Precombines::Start();
                // The runtime bake (built in since 1.0; bRuntimeBake in SkyrimPrecombines.ini) hands its parts to the
                // manager Start has just set up.
                RuntimeBake::Load();
                break;
            case SKSE::MessagingInterface::kPreLoadGame:
                Precombines::BeforeLoad();
                break;
            case SKSE::MessagingInterface::kPostLoadGame:
                logger::info("Save game loaded");
                Precombines::AfterLoad();
                Occlusion::CheckOtherCullers();
                break;
            case SKSE::MessagingInterface::kNewGame:
                logger::info("New game started");
                Precombines::AfterLoad();
                Occlusion::CheckOtherCullers();
                break;
            }
        } catch (const std::exception& e) {
            logger::error("SKSE message {}: {}", a_message->type, e.what());
        } catch (...) {
            logger::error("SKSE message {}: unknown error", a_message->type);
        }
    }

    // SE and AE: true only when nothing that needs the main-camera hooks can run this session: Hi-Z
    // ("hiz_occlusion", default on), previs ("previs": it hides chunks from the main camera through them), the box test
    // ("box_occlusion"), developer mode ("developer_mode") and the developer shortcut keys ("developer_shortcuts",
    // bDeveloperShortcuts; both: the keys switch the others on in play) and the frame log and Hi-Z A/B ("frame_log",
    // "hiz_ab": both are recorded in the main draw hook) are all off. Read as Precombines.cpp reads them (Config::Flag):
    // true/false or 0/1, any other value the default with one warning. A new setting that needs the hooks belongs here
    // too. Without the hooks the in-game menu switches Hi-Z on only after a restart (Settings::Status), and the shortcut
    // keys turned on in play give only the precombines key until a restart (previs, too, needs the hooks).
    bool HooksUnneeded()
    {
        try {
            return !Config::Flag("hiz_occlusion", true) && !Config::Flag("previs", false) && !Config::Flag("box_occlusion", false) &&
                   !Config::Flag("developer_mode", false) && !Config::Flag("developer_shortcuts", false) && !Config::Flag("frame_log", false) &&
                   !Config::Flag("hiz_ab", false);
        } catch (const std::exception& e) {
            logger::warn("Hi-Z occlusion: settings unreadable at plugin load ({}); hooks installed", e.what());
            return false;
        }
    }
}

// ── Plugin declaration (CommonLibSSE-NG style) ─────────────────────────────
// This macro handles SE/AE/VR multi-runtime automatically
SKSEPluginLoad(const SKSE::LoadInterface* a_skse)
{
    // Other game versions are refused before SKSE::Init loads an address library (SKSE's own log names the plugin as
    // not loaded): the hook sites are known only for the versions SupportedRuntime names.
    if (!SupportedRuntime()) return false;

    // SKSE::Init also creates the plugin log (pattern "[time] [thread] [level] message").
    SKSE::Init(a_skse);

    // No built-in crash handler in releases (it would compete with crash-logger mods); warnings and errors flush at once.
    if (auto log = spdlog::default_logger()) log->flush_on(spdlog::level::warn);

    const auto plugin = SKSE::PluginDeclaration::GetSingleton();
    logger::info("{} v{} loading", plugin->GetName(), plugin->GetVersion());

    // Detect runtime version
    const auto ver = REL::Module::get().version();
    logger::info("Detected Skyrim{} v{}", REL::Module::IsVR() ? " VR" : "", ver.string());

    // The settings the in-game menu shows and changes, and the developer shortcut keys, from SkyrimPrecombines.ini (read
    // here).
    Settings::Init();

    // Hi-Z occlusion hooks (on by default). They go in at plugin load, before the game reads its data. On SE and AE,
    // bHiZ=0 with previs, the box test, developer mode, the developer shortcut keys and the test logs off (their defaults)
    // installs none, so this plugin is then out of the main draw entirely; otherwise bHiZ=0 only stops the culling and the
    // chained hooks pass every call on. VR installs them only with hiz_vr=true (Occlusion::Install).
    if (!REL::Module::IsVR() && HooksUnneeded())
        logger::info("Hi-Z occlusion off (bHiZ=0; previs, box test, developer mode and developer shortcut keys off): hooks not installed");
    else
        Occlusion::Install();

    // Register message listener
    const auto messaging = SKSE::GetMessagingInterface();
    if (!messaging || !messaging->RegisterListener(MessageHandler)) {
        logger::critical("Failed to register message listener");
        return false;
    }

    logger::info("{} loaded successfully", plugin->GetName());
    return true;
}
