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

#include "RE/Skyrim.h"
#include "SKSE/SKSE.h"

#include <spdlog/sinks/basic_file_sink.h>

namespace logger = SKSE::log;

using namespace std::literals;

#define DLLEXPORT __declspec(dllexport)

// The game versions whose hand-written hook sites, call targets and member offsets were checked in Ghidra: SE 1.5.97,
// AE 1.6.1170 (Steam), 1.6.1179 (GOG) and 1.7.104, and VR 1.4.15. Plugins refuse any other version before SKSE::Init
// loads an address library.
inline bool SupportedRuntime()
{
    const auto version = REL::Module::get().version();
    if (REL::Module::IsVR()) return version == REL::Version(1, 4, 15, 0);
    return version == REL::Version(1, 5, 97, 0) || version == REL::Version(1, 6, 1170, 0) || version == REL::Version(1, 6, 1179, 0) ||
           version == REL::Version(1, 7, 104, 0);
}
