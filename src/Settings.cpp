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

#include "Settings.h"
#include "Config.h"
#include "Occlusion.h"
#include "Precombines.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <format>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace Settings {
    namespace {
        using json=nlohmann::json;
        using Clock=std::chrono::steady_clock;
        constexpr auto relaxed=std::memory_order_relaxed;

        enum class Type : std::uint8_t { kBool, kFloat, kString };
        struct Option {
            std::string_view name;  // shipped INI name and menu id, e.g. "bHiZ"
            std::string_view key;   // internal name, e.g. "hiz_occlusion"
            Type type;
            bool live;              // false: applies after a restart
            bool vrOnly;
            float min, max;         // kFloat: accepted range, values are clamped
            float fallback;         // without an INI line (kBool: 0 or 1): the shipped file's value
        };
        // Menu order. A float's range is what the menu and the INI may set (values are clamped); the menu's sliders show a
        // narrower part of it. The fallbacks are the other readers' defaults (Precombines ReadSettings, BakePlugin.cpp,
        // VRDepthCulling), which are the shipped file's values.
        constexpr std::array kOptions{
            Option{"bEnabled","enabled",Type::kBool,true,false,0.f,1.f,1.f},
            Option{"bRuntimeBake","runtime_bake",Type::kBool,false,false,0.f,1.f,1.f},
            Option{"sCacheFolder","cache_folder",Type::kString,false,false,0.f,0.f,0.f},
            Option{"bHiZ","hiz_occlusion",Type::kBool,true,false,0.f,1.f,1.f},
            Option{"fHiZMinFrameMs","hiz_min_frame_ms",Type::kFloat,true,false,0.f,100.f,5.f},
            Option{"fHiZMinYield","hiz_min_yield",Type::kFloat,true,false,0.f,1.f,0.01f},
            Option{"bDepthCulling","vr_depth_culling",Type::kBool,false,true,0.f,1.f,1.f},
            Option{"fMinOccludeeExtent","vr_min_occludee_extent",Type::kFloat,false,true,0.f,1000.f,10.f},
            Option{"bDeveloperShortcuts","developer_shortcuts",Type::kBool,true,false,0.f,1.f,0.f},
        };
        enum Index : size_t { kEnabled,kRuntimeBake,kCacheFolder,kHiZ,kMinFrameMs,kMinYield,kDepthCulling,kMinExtent,kDeveloperShortcuts };
        static_assert(kOptions.size()==kDeveloperShortcuts+1 && kOptions[kEnabled].key=="enabled" && kOptions[kHiZ].key=="hiz_occlusion" &&
                      kOptions[kMinFrameMs].key=="hiz_min_frame_ms" && kOptions[kMinYield].key=="hiz_min_yield" &&
                      kOptions[kDeveloperShortcuts].key=="developer_shortcuts");

        // The shortcut keys in the menu's order, changed only in SkyrimPrecombines.ini.
        constexpr std::array kShortcuts{
            Shortcut{"toggle_key","Precombines on/off"},
            Shortcut{"hiz_toggle_key","Hi-Z occlusion on/off"},
            Shortcut{"hiz_geometry_toggle_key","Hi-Z per-object tests on/off"},
            Shortcut{"box_occlusion_toggle_key","GPU box test on/off (experimental)"},
            Shortcut{"previs_toggle_key","Previs on/off"},
            Shortcut{"hiz_probe_key","Hi-Z probe (needs hiz_diagnostics)"},
        };
        enum Key : size_t { kToggleKey,kHiZKey,kGeometryKey,kBoxKey,kPrevisKey,kProbeKey };
        static_assert(kShortcuts.size()==kProbeKey+1 && kShortcuts[kBoxKey].key=="box_occlusion_toggle_key");
        // F10, F8, F7, F6 (DirectInput scan codes): unbound in play. F1 opens SKSE Menu Framework, F5 and F9 save and load,
        // F12 is Steam's screenshot key. Previs and the probe have no default key.
        constexpr std::array<std::uint32_t,kShortcuts.size()> kKeyDefaults{0x44,0x42,0x41,0x40,0,0};
        // The box test needs one eye (SE and AE): no default key on VR.
        std::uint32_t KeyDefault(size_t index) { return index==kBoxKey && REL::Module::IsVR() ? 0 : kKeyDefaults[index]; }

        using Value=std::variant<bool,float,std::string>;
        Value Default(const Option& option) {
            switch(option.type) {
            case Type::kBool: return Value(option.fallback!=0.f);
            case Type::kFloat: return Value(option.fallback);
            default: return Value(std::string());
            }
        }
        json ToJson(const Value& value) { return std::visit([](const auto& v) { return json(v); },value); }

        // value: what the game uses now; startup: from the INI; saved: the last value the menu set, which the writer puts in
        // the INI while dirty, so what the next start reads (the restart notes go by it). The shortcut keys change only
        // value.
        struct Slot { Value value,startup,saved; bool dirty=false; };
        // Lock order. `order` is held through a whole Set, so the applies happen in the order of the stores. `mutex` (the
        // store) is a leaf: nothing is called while it is held, so any code may read Settings, also under stateMutex
        // (Precombines ReadSettings) or in an SKSE task. Under `order` an apply takes Occlusion's switch mutex or queues an
        // SKSE task; SKSE may hold its task queue while tasks run, and tasks take stateMutex, so nothing under stateMutex
        // may call a Set. Sets come from the menu pages, the input sink and, through the sink, from inside an SKSE task
        // (simulated key presses sent from a task). That is
        // safe only while every Set runs on the thread that runs SKSE's tasks (the game's main thread, which also runs the
        // input sink and draws SKSE Menu Framework's pages), where SKSE's queue lock is re-entrant: a Set on another thread
        // could hold `order` and wait for that lock while a running task waits for `order`.
        struct Store {
            std::mutex order;
            std::mutex mutex;
            std::condition_variable wake;
            std::array<Slot,kOptions.size()> slots;
            std::array<std::uint32_t,kShortcuts.size()> keys{};
            std::uint64_t requests=0,taken=0;  // saving Sets; those the writer has collected
            Clock::time_point lastRequest{};
            bool writer=false;                 // the writer thread was started
            Store() {
                for(size_t i=0;i<kOptions.size();++i) slots[i].value=slots[i].startup=slots[i].saved=Default(kOptions[i]);
                for(size_t i=0;i<kShortcuts.size();++i) keys[i]=KeyDefault(i);
            }
        };
        // Never destroyed: the detached writer may still wait on it while the process exits.
        Store& S() { static auto* store=new Store; return *store; }
        std::atomic<std::uint64_t> generation{0};
        std::atomic_bool shortcuts{false};      // bDeveloperShortcuts, live
        std::atomic_bool developerMode{false};  // "developer_mode" (testing), fixed at Init
        std::atomic_bool saveFailed{false};     // the writer's last save of SkyrimPrecombines.ini failed (SaveFailed)
        // How long the writer waits after the last change: an SKSE Menu Framework slider sets its value every frame while
        // it is dragged, and the file is written once.
        constexpr auto kSettle=std::chrono::milliseconds(300);

        bool Same(std::string_view a,std::string_view b) {
            return a.size()==b.size() && std::equal(a.begin(),a.end(),b.begin(),[](char x,char y) {
                return std::tolower(static_cast<unsigned char>(x))==std::tolower(static_cast<unsigned char>(y));
            });
        }
        std::optional<size_t> OptionIndex(std::string_view name) {
            for(size_t i=0;i<kOptions.size();++i) if(Same(name,kOptions[i].name) || Same(name,kOptions[i].key)) return i;
            return std::nullopt;
        }
        std::optional<size_t> KeyIndex(std::string_view name) {
            for(size_t i=0;i<kShortcuts.size();++i) if(Same(name,kShortcuts[i].key)) return i;
            return std::nullopt;
        }
        std::string_view Trim(std::string_view t) {
            while(!t.empty() && std::isspace(static_cast<unsigned char>(t.front()))) t.remove_prefix(1);
            while(!t.empty() && std::isspace(static_cast<unsigned char>(t.back()))) t.remove_suffix(1);
            return t;
        }

        // Through Config's readers, as the engine reads the same settings: a value of the wrong type keeps its default
        // with one warning (Config reports each setting once, whoever reads it).
        Value FromConfig(const Option& option) {
            switch(option.type) {
            case Type::kBool: return Value(Config::Flag(option.key,option.fallback!=0.f));
            case Type::kFloat: {
                const float number=Config::Number<float>(option.key,option.fallback);
                return Value(std::isfinite(number) ? std::clamp(number,option.min,option.max) : option.fallback);
            }
            default: return Value(Config::Text(option.key,std::string()));
            }
        }
        // A key: its scan code from 0 to 255 (0: none), else the default; without a line, the default.
        std::uint32_t KeyFromConfig(size_t index) {
            const auto key=kShortcuts[index].key;
            const auto fallback=KeyDefault(index);
            const auto code=Config::Number<std::uint32_t>(key,fallback);
            if(code<=255) return code;
            logger::warn("SkyrimPrecombines.ini: {}={} is not a scan code from 0 to 255; {} is used",key,code,KeyName(fallback));
            return fallback;
        }

        // SE and AE install the Hi-Z hooks at plugin load unless nothing could use them (main.cpp HooksUnneeded: bHiZ=0
        // and the developer features off), or they failed; bHiZ then applies after a restart. VR installs them only with
        // hiz_vr (the menu hides Hi-Z there otherwise).
        bool HooksMissing() { return !REL::Module::IsVR() && !Occlusion::Installed(); }
        template<class T> T Now(const Store& s,size_t index) { const auto* v=std::get_if<T>(&s.slots[index].value); return v ? *v : T{}; }
        template<class T> T AtStart(const Store& s,size_t index) { const auto* v=std::get_if<T>(&s.slots[index].startup); return v ? *v : T{}; }
        template<class T> T Saved(const Store& s,size_t index) { const auto* v=std::get_if<T>(&s.slots[index].saved); return v ? *v : T{}; }
        // The restart notes compare the saved values (what the next start reads) with the startup ones, so a shortcut
        // key's session-only change predicts nothing. Under the store's lock.
        // The runtime bake reads bEnabled once (Precombines ReadSettings, BakePlugin.cpp), so outside developer mode a game
        // started with it off merges no new cell until a restart, although precombines switched on in play read the cells
        // merged before at once (Precombines ResumeRuntimeHandOff); one started with it on goes on merging new cells after
        // it is switched off, until a restart.
        bool BakeStarts(const Store& s) {
            return !developerMode.load(relaxed) && !AtStart<bool>(s,kEnabled) && Saved<bool>(s,kEnabled) && Saved<bool>(s,kRuntimeBake);
        }
        bool BakeStops(const Store& s) {
            return !developerMode.load(relaxed) && AtStart<bool>(s,kEnabled) && AtStart<bool>(s,kRuntimeBake) && !Saved<bool>(s,kEnabled);
        }
        // Hi-Z switched in a session without its hooks.
        bool HiZWaits(const Store& s) { return HooksMissing() && Saved<bool>(s,kHiZ)!=AtStart<bool>(s,kHiZ); }
        // The shortcut keys turned on in a session that left the hooks out because they were off (main.cpp HooksUnneeded):
        // only the precombines key works until a restart (the input sink refuses the others).
        bool KeysWait(const Store& s) {
            return HooksMissing() && !developerMode.load(relaxed) && !AtStart<bool>(s,kDeveloperShortcuts) && Saved<bool>(s,kDeveloperShortcuts);
        }

        // A live option's effect, under `order` (not the store's lock), so applies happen in the order of the Sets. Only
        // atomics, Occlusion's switch mutex and SKSE tasks (see Store for the lock order).
        void Apply(size_t index,const Value& value) {
            switch(index) {
            case kEnabled: Precombines::SetEnabled(std::get<bool>(value)); break;
            // Without the hooks the value waits for the next start (HiZWaits).
            case kHiZ: if(Occlusion::Installed()) Occlusion::SetEnabled(std::get<bool>(value)); break;
            case kMinFrameMs: Occlusion::SetMinFrameMs(std::get<float>(value)); break;
            case kMinYield: Occlusion::SetMinYield(std::get<float>(value)); break;
            case kDeveloperShortcuts: {
                const bool on=std::get<bool>(value);
                if(shortcuts.exchange(on,relaxed)!=on) logger::info("Developer shortcut keys {}",on ? "ON":"OFF");
                break;
            }
            default: break;
            }
        }

        // One thread writes every menu change to SkyrimPrecombines.ini, in order, once kSettle has passed without another.
        // A failed write marks its options again, so the next change retries them (Config::Write has logged why), and the
        // page says so (SaveFailed) until a save succeeds.
        void Writer() {
            auto& s=S();
            for(;;) {
                std::vector<size_t> written;
                try {
                    std::vector<Config::Change> changes;
                    {
                        std::unique_lock lock(s.mutex);
                        s.wake.wait(lock,[&] { return s.requests!=s.taken; });
                        while(Clock::now()<s.lastRequest+kSettle) s.wake.wait_until(lock,s.lastRequest+kSettle);
                        s.taken=s.requests;
                        for(size_t i=0;i<kOptions.size();++i) {
                            auto& slot=s.slots[i];
                            if(!slot.dirty) continue;
                            slot.dirty=false;
                            written.push_back(i);
                            changes.push_back(Config::Change{std::string(kOptions[i].key),ToJson(slot.saved)});
                        }
                    }
                    if(changes.empty()) continue;
                    const bool saved=Config::Write(changes);
                    saveFailed.store(!saved,relaxed);
                    if(saved) continue;
                } catch(...) {
                    // Nothing leaves this thread (std::terminate would close the game); no busy retry either.
                    saveFailed.store(true,relaxed);
                    std::this_thread::sleep_for(kSettle);
                }
                std::lock_guard lock(s.mutex);
                for(const auto i:written) s.slots[i].dirty=true;
            }
        }
        void StartWriter() {
            try { std::thread(Writer).detach(); }
            catch(const std::exception& e) {
                { std::lock_guard lock(S().mutex); S().writer=false; }
                saveFailed.store(true,relaxed);
                logger::warn("SkyrimPrecombines.ini could not be saved ({}); the change lasts until the game is closed",e.what());
            }
        }

        bool Put(size_t index,Value value,Save save) {
            auto& s=S();
            std::lock_guard ordered(s.order);
            bool start=false;
            {
                std::lock_guard lock(s.mutex);
                auto& slot=s.slots[index];
                slot.value=value;
                if(save==Save::kYes) {
                    slot.saved=value; slot.dirty=true;
                    ++s.requests; s.lastRequest=Clock::now();
                    start=!std::exchange(s.writer,true);
                }
                generation.fetch_add(1,relaxed);
            }
            if(kOptions[index].live) Apply(index,value);
            if(start) StartWriter();
            if(save==Save::kYes) s.wake.notify_one();
            return true;
        }

        // Under the store's lock.
        std::string KeyList(const Store& s) {
            std::string out;
            for(size_t i=0;i<kShortcuts.size();++i) {
                const auto code=s.keys[i];
                if(!code) continue;
                if(!out.empty()) out+=", ";
                out+=std::format("{} {}",KeyName(code),kShortcuts[i].label);
            }
            return out.empty() ? std::string("no keys") : out;
        }
    }

    void Init() {
        try {
            auto& s=S();
            developerMode.store(Config::Flag("developer_mode",false),relaxed);
            std::lock_guard lock(s.mutex);  // SkyrimPrecombines.ini, through Config's readers
            for(size_t i=0;i<kOptions.size();++i) {
                auto& slot=s.slots[i];
                slot.value=FromConfig(kOptions[i]);
                slot.startup=slot.saved=slot.value;
            }
            for(size_t i=0;i<kShortcuts.size();++i) s.keys[i]=KeyFromConfig(i);
            shortcuts.store(std::get<bool>(s.slots[kDeveloperShortcuts].value),relaxed);
            if(ShortcutsOn())
                logger::info("Developer shortcut keys ON ({}): {}",developerMode.load(relaxed) ? "developer_mode" : "bDeveloperShortcuts",KeyList(s));
        } catch(const std::exception& e) { logger::error("SkyrimPrecombines.ini settings could not be taken over ({}); defaults are used",e.what()); }
    }

    bool GetBool(std::string_view name) {
        const auto index=OptionIndex(name);
        if(!index) return false;
        auto& s=S();
        std::lock_guard lock(s.mutex);
        const auto* value=std::get_if<bool>(&s.slots[*index].value);
        return value && *value;
    }

    float GetFloat(std::string_view name) {
        const auto index=OptionIndex(name);
        if(!index) return 0.f;
        auto& s=S();
        std::lock_guard lock(s.mutex);
        const auto* value=std::get_if<float>(&s.slots[*index].value);
        return value ? *value : 0.f;
    }

    std::string GetString(std::string_view name) {
        try {
            const auto index=OptionIndex(name);
            if(!index) return {};
            auto& s=S();
            std::lock_guard lock(s.mutex);
            const auto* value=std::get_if<std::string>(&s.slots[*index].value);
            return value ? *value : std::string();
        } catch(...) { return {}; }
    }

    std::int32_t GetInt(std::string_view name) {
        const auto index=KeyIndex(name);
        if(!index) return 0;
        auto& s=S();
        std::lock_guard lock(s.mutex);
        return static_cast<std::int32_t>(s.keys[*index]);
    }

    bool SetBool(std::string_view name,bool value,Save save) {
        try {
            const auto index=OptionIndex(name);
            if(!index || kOptions[*index].type!=Type::kBool || !Available(name)) return false;
            return Put(*index,Value(value),save);
        } catch(...) { return false; }
    }

    bool SetFloat(std::string_view name,float value,Save save) {
        try {
            const auto index=OptionIndex(name);
            if(!index || kOptions[*index].type!=Type::kFloat || !Available(name) || !std::isfinite(value)) return false;
            const auto& option=kOptions[*index];
            return Put(*index,Value(std::clamp(value,option.min,option.max)),save);
        } catch(...) { return false; }
    }

    bool SetString(std::string_view name,std::string_view value,Save save) {
        try {
            const auto index=OptionIndex(name);
            if(!index || kOptions[*index].type!=Type::kString || !Available(name)) return false;
            value=Trim(value);
            if(value.size()>=2 && value.front()=='"' && value.back()=='"') value=value.substr(1,value.size()-2);
            // One INI line: no line breaks (nor NUL), and a path's length.
            if(value.size()>1024 || value.find_first_of("\r\n\0"sv)!=std::string_view::npos) return false;
            return Put(*index,Value(std::string(value)),save);
        } catch(...) { return false; }
    }

    bool Available(std::string_view name) {
        if(const auto index=OptionIndex(name)) {
            if(kOptions[*index].vrOnly) return REL::Module::IsVR();
            // Hi-Z and its gates: on SE and AE also without the hooks, where a change is kept for the next start (Status;
            // the gates take effect whenever Hi-Z runs); on VR only with them (hiz_vr). The menu groups the gates with bHiZ.
            if(*index==kHiZ || *index==kMinFrameMs || *index==kMinYield) return Occlusion::Installed() || !REL::Module::IsVR();
            return true;
        }
        if(const auto index=KeyIndex(name)) {
            switch(*index) {
            case kHiZKey: case kGeometryKey: return Occlusion::Installed();
            case kBoxKey: return !REL::Module::IsVR() && Occlusion::Installed();
            // Previs is off on VR (Precombines ReadSettings) and needs the main-camera hooks, without which it would hide
            // chunks from water reflections too (the input sink refuses its key then); neither key has a default.
            case kPrevisKey: return !REL::Module::IsVR() && Occlusion::Installed() && GetInt(kShortcuts[*index].key)!=0;
            case kProbeKey: return Occlusion::Installed() && GetInt(kShortcuts[*index].key)!=0;
            default: return true;
            }
        }
        return false;
    }

    std::string Status(std::string_view name) {
        try {
            const auto index=OptionIndex(name);
            if(!index) return {};
            auto& s=S();
            if(*index==kHiZ) {
                std::lock_guard lock(s.mutex);
                if(HiZWaits(s)) return "Applies after a restart";
                if(!Now<bool>(s,kHiZ)) return {};
                if(HooksMissing()) return "Not running (see SkyrimPrecombines.log)";
                return {};
            }
            if(*index==kEnabled) {
                const bool failed=Precombines::Failed();
                std::lock_guard lock(s.mutex);
                if(failed && Now<bool>(s,kEnabled)) return "Off after an error (see SkyrimPrecombines.log); tick to try again";
                if(BakeStarts(s)) return "New cells are merged after a restart";
                return BakeStops(s) ? "New cells are still merged until a restart" : "";
            }
            if(*index==kDeveloperShortcuts) {
                std::lock_guard lock(s.mutex);
                return KeysWait(s) ? "Only the precombines key works until a restart" : "";
            }
            return {};
        } catch(...) { return {}; }
    }

    bool RestartPending() {
        auto& s=S();
        std::lock_guard lock(s.mutex);
        for(size_t i=0;i<kOptions.size();++i) if(!kOptions[i].live && s.slots[i].saved!=s.slots[i].startup) return true;
        return HiZWaits(s) || BakeStarts(s) || BakeStops(s);
    }

    bool SaveFailed() { return saveFailed.load(relaxed); }

    std::uint64_t Generation() { return generation.load(relaxed); }

    bool ShortcutsOn() { return shortcuts.load(relaxed) || developerMode.load(relaxed); }

    std::span<const Shortcut> Shortcuts() { return kShortcuts; }

    std::string KeyName(std::uint32_t scanCode) {
        if(scanCode==0) return "None";
        if(scanCode>=0x3B && scanCode<=0x44) return std::format("F{}",scanCode-0x3B+1);
        if(scanCode==0x57) return "F11";
        if(scanCode==0x58) return "F12";
        return std::format("Key {}",scanCode);  // decimal, as the INI takes it
    }

    std::string Version() {
        const auto* plugin=SKSE::PluginDeclaration::GetSingleton();
        if(!plugin) return {};
        const auto version=plugin->GetVersion();
        return std::format("{}.{}.{}",version.major(),version.minor(),version.patch());
    }
}
