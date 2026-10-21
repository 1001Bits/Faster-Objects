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
