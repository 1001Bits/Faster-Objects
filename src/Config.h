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

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// The settings file Data/SKSE/Plugins/SkyrimPrecombines.ini, read once (any thread). Each key=value line becomes a value
// under the setting's internal name, which the readers look up as before:
// - the shipped names map to one (bEnabled is "enabled", see Config.cpp) and are typed by their prefix as in the game's
//   own INIs: b reads 1/0/true/false, f and i a number (both after a trailing ; or # comment and quotes are removed),
//   s text;
// - an internal name (development and test settings, e.g. hiz_toggle_key) is taken as it is, its value as JSON (true, 5,
//   [1,2], "text") or else as text.
// Names are not case-sensitive and sections only group them. Without the file every setting has its default, which is
// the shipped file's value.
// The SKSE Menu Framework page saves its changes into the same file (Write); Get() stays what was read at startup.
namespace Config {
    const nlohmann::json& Get();

    // A setting by its internal name, read without ever throwing: a value of another type (enabled=1 is a number,
    // toggle_key=F10 text) is reported once and its default used, so one bad line costs only that setting (enabled=1
    // once threw out of Precombines::Start and left precombines, previs and Hi-Z off for the session).
    // Flag: true/false, or the numbers 0 and 1 as the shipped settings write them.
    bool Flag(std::string_view key,bool fallback);
    // A number (a boolean is none) that T holds: an unsigned setting refuses a negative value rather than wrapping it
    // (toggle_key=-1 was 4294967295). T is uint32_t, int, float or double.
    template<class T> T Number(std::string_view key,T fallback);
    std::string Text(std::string_view key,std::string fallback);
    // Form IDs: one value or a list, each a number or hex text with or without 0x (000152AA as xEdit shows it, 0x21000801;
    // [000152AA, 0001A26F] unquoted too). A bare 21000801 is JSON's decimal number: read so, and reported. Anything else
    // is reported and skipped.
    std::vector<std::uint32_t> FormIds(std::string_view key);

    // One value to save under its internal name ("hiz_occlusion"): a JSON bool, number or string.
    struct Change { std::string key; nlohmann::json value; };
    // `text` (the INI file's bytes, possibly empty) with every change applied; every other byte is kept. Pure: no file
    // access, no logging. A change updates the last line that sets it (the one Get() reads), under its shipped or its
    // internal name and in that line's spelling (bHiZ=0, hiz_occlusion=false). On a b, f or i line the quotes and the
    // comment after the value stay (bHiZ=0 ; testing); on others all after the '=' and its spaces is the value. Without
    // such a line the shipped name is added to its section ([Occlusion], appended when missing), below the section's last
    // setting and the comment lines right under it; a name that is not shipped goes under [Developer]. A value that
    // cannot be written under that name (text for a b setting, a line break, a number that is not finite) is skipped,
    // and UTF-16 text is never changed. Text without any change comes back byte for byte; changed text always ends with
    // a line ending.
    std::string Rewrite(std::string_view text,std::span<const Change> changes);
    // Applies the changes to Data/SKSE/Plugins/SkyrimPrecombines.ini (created when missing) through a temporary file. Never
    // throws. False, after one warning, when the file cannot be read or written. Should a write in place (the file could
    // not be replaced) stop partway, SkyrimPrecombines.ini.tmp keeps the whole new text and the next Write starts from it.
    bool Write(std::span<const Change> changes);

    // sCacheFolder: the folder runtime-baked cells are kept in, or empty (the default) for the game's Data folder, which
    // under Mod Organizer 2 is the overwrite folder. Checked once: a folder that cannot be created leaves the default. Its
    // path may hold any character, as the bake writes paths into JSON as UTF-8. Absolute either way.
    const std::filesystem::path& CacheFolder();
    // Part manifests, room groups, assets, previs tables and the work folder: <cache folder>/RuntimeBake, or
    // Data/SKSE/Plugins/Precombines/RuntimeBake.
    std::filesystem::path BakeDir();
    // The root runtime-baked chunk meshes (precombined/skyrim/rb/...) are written under: <cache folder>/meshes, or
    // Data/meshes. Meshes outside Data are read from their files, not through the game's resource system.
    std::filesystem::path MeshRoot();
}
