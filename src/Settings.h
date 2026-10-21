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

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

// The settings the SKSE Menu Framework page shows (SmfMenu.cpp), and the developer shortcut keys. SkyrimPrecombines.ini
// is their only store: values start from Config's readers at plugin load; a menu change applies at once where it can and
// is written back (Config::Write) on one background thread. What the shortcut keys switch lasts for the session only and
// is never written. A name is the shipped INI name (bHiZ) or the internal one (hiz_occlusion), not case-sensitive. Any
// thread.
namespace Settings {
    // SKSEPluginLoad, after SKSE::Init and before Occlusion::Install (main.cpp). Never throws.
    void Init();
    // kNo: this session only (the shortcut keys); kYes: also saved to SkyrimPrecombines.ini.
    enum class Save : bool { kNo, kYes };
    bool GetBool(std::string_view name);
    float GetFloat(std::string_view name);
    std::string GetString(std::string_view name);
    // A shortcut key by its internal name (toggle_key, ...): its DirectInput scan code, 0 for none.
    std::int32_t GetInt(std::string_view name);
    // False, and nothing changes, for an unknown name, a value of another type or an option without an effect here
    // (Available). A float is clamped to its option's range. Never throws.
    bool SetBool(std::string_view name,bool value,Save save=Save::kYes);
    bool SetFloat(std::string_view name,float value,Save save=Save::kYes);
    bool SetString(std::string_view name,std::string_view value,Save save=Save::kYes);
    bool Available(std::string_view name);       // the option or key has an effect on this game and install
    std::string Status(std::string_view name);   // short menu text, "" when there is nothing to say
    bool RestartPending();                       // a saved change takes effect only at the next start
    bool SaveFailed();                           // the last save of SkyrimPrecombines.ini failed (logged); the next change retries
    std::uint64_t Generation();                  // moves on with every Set
    bool ShortcutsOn();                          // bDeveloperShortcuts (live) or developer_mode (testing, fixed at startup)
    struct Shortcut { std::string_view key; std::string_view label; };
    std::span<const Shortcut> Shortcuts();       // the shortcut keys in the menu's order
    std::string KeyName(std::uint32_t scanCode);
    std::string Version();                       // "1.0.0", from SKSE::PluginDeclaration
}
