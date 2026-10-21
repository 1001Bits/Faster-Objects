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
#include "Precombines.h"
#include "PeriodicValidation.h"
#include "Occlusion.h"
#include "Config.h"
#include "Settings.h"
#include "PrecombineMeshCore.h"  // kOutputVersion: cached runtime bakes of another baker are not read
#include "RE/S/SendHUDMessage.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <condition_variable>
#include <deque>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace Precombines {
namespace {
    using json = nlohmann::json;
    namespace fs = std::filesystem;
    using Clock = std::chrono::steady_clock;
    constexpr auto manifestDir = "Data/SKSE/Plugins/Precombines";
    // Previs tables live in a subfolder, so readers of the manifest folder never mistake one for a manifest.
    constexpr auto previsDir = "Data/SKSE/Plugins/Precombines/Previs";
    // Activation per task is bounded by time (activation_budget_us); the count only caps a run of very cheap attaches.
    // With meshes prepared off the game thread an attach costs about 0.3-0.5 ms, so 4 per task took 19 s for
    // Ustengrav02's 666 runtime-baked chunks.
    constexpr size_t attachesPerTick = 16;
    constexpr auto relaxed = std::memory_order_relaxed;

    std::string normalized(std::string path) {
        for (auto& c : path) { if (c == '\\') c = '/'; else c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }
        return path;
    }
    bool safeMesh(const std::string& path) {
        if (path.empty() || path.size() > 240 || path.find(':') != std::string::npos || path.front() == '/' || path.find('\\') != std::string::npos) return false;
        // Split at '/' by hand: fs::path(path) reads the bytes in the ANSI code page, and a model path in UTF-8 (which the
        // runtime bake passes on as the game holds it) throws there on a double-byte code page.
        std::string_view name;
        for (size_t at = 0; at <= path.size();) {
            const auto end = std::min(path.find('/', at), path.size());
            name = std::string_view(path).substr(at, end - at);
            if (name == ".." || name == ".") return false;
            at = end + 1;
        }
        return name.size() > 4 && normalized(std::string(name.substr(name.size() - 4))) == ".nif";
    }
    void need(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
    // Manifest form references. Old manifests name Skyrim.esm forms by their form ID, a number below 0x1000000
    // (Skyrim.esm is always load index 0). Plugin-relative manifests name a form as {"plugin": "<file>", "id": <local
    // form ID>} or ["<file>", <local form ID>], resolved through the data handler for the actual load order.
    // Returns 0 for a well-formed reference whose plugin is not loaded; throws for a malformed one.
    // `plugins`: the plugins one manifest names, each looked up in the load order once, by lower-case file name as
    // LookupModByName compares them (two linear scans of the plugin list per form, LookupModByName and again
    // inside LookupFormID). LookupFormID adds the local ID to the plugin's index bits (compile index, light-plugin index,
    // VR's own layout), so the bits of local ID 0 serve every form of the plugin.
    struct LoadedPlugin { const RE::TESFile* file=nullptr; RE::FormID base=0; };
    using Plugins=std::unordered_map<std::string,LoadedPlugin>;
    uint32_t resolveForm(const json& value,Plugins* plugins=nullptr) {
        if(value.is_number_integer()) {
            const auto id=value.get<int64_t>();
            need(id>0 && id<0x1000000,"invalid Skyrim.esm form ID");
            return static_cast<uint32_t>(id);
        }
        std::string plugin; int64_t local=0;
        if(value.is_object()) { plugin=value.at("plugin").get<std::string>(); local=value.at("id").get<int64_t>(); }
        else { need(value.is_array() && value.size()==2,"invalid form reference"); plugin=value.at(0).get<std::string>(); local=value.at(1).get<int64_t>(); }
        need(!plugin.empty() && plugin.size()<=260 && local>0 && local<0x1000000,"invalid form reference");
        auto* handler=RE::TESDataHandler::GetSingleton();
        if(!plugins) {
            const auto* file=handler ? handler->LookupModByName(plugin) : nullptr;
            if(!file || file->GetCompileIndex()==0xFF) return 0;
            need(!file->IsLight() || local<0x1000,"invalid light plugin form ID");
            return handler->LookupFormID(static_cast<RE::FormID>(local),plugin);
        }
        std::string name(plugin);
        for(auto& c:name) c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        auto [found,added]=plugins->try_emplace(std::move(name));
        if(added && handler) {
            found->second.file=handler->LookupModByName(plugin);
            if(found->second.file && found->second.file->GetCompileIndex()!=0xFF) found->second.base=handler->LookupFormID(0,plugin);
        }
        const auto* file=found->second.file;
        if(!file || file->GetCompileIndex()==0xFF) return 0;
        need(!file->IsLight() || local<0x1000,"invalid light plugin form ID");
        return found->second.base+static_cast<RE::FormID>(local);
    }
    bool relativeForm(const json& value) { return !value.is_number_integer(); }
    // Runtime-bake hand-off (config "runtime_bake"): parts the runtime bake publishes for cells without a manifest in
    // the manifest folder, under the cache folder (Config::BakeDir, "sCacheFolder").
    fs::path runtimePartsDir() { return Config::BakeDir()/"parts"; }
    // Previs tables the bake renders for runtime-baked cells: <cell key>-<bake>.pvs.json.
    fs::path runtimePrevisDir() { return Config::BakeDir()/"previs"; }
    // "<cell key, 8 or 16 hex>-<bake, 16 hex>-<part>.json", as the bake plugin names them; anything else is refused.
    bool runtimePrevisName(const std::string& name) {
        const auto hex=[](char c) { return (c>='0' && c<='9') || (c>='a' && c<='f'); };
        const auto key=name.find('-');
        if((key!=8 && key!=16) || !name.ends_with(".pvs.json") || name.size()<key+1+16+9) return false;
        for(size_t i=0;i<key+17;++i) if(i!=key && !hex(name[i])) return false;
        if(name.size()==key+1+16+9) return true;
        // Exterior previs: "-from-" and the camera cell's key (8 or 16 hex) before ".pvs.json".
        const auto from=name.substr(key+17,name.size()-9-(key+17));
        if(!from.starts_with("-from-") || (from.size()!=6+8 && from.size()!=6+16)) return false;
        for(size_t i=6;i<from.size();++i) if(!hex(from[i])) return false;
        return true;
    }
    bool runtimePartName(const std::string& name) {
        const auto hex=[](char c) { return (c>='0' && c<='9') || (c>='a' && c<='f'); };
        const auto key=name.find('-');
        if(key!=8 && key!=16) return false;
        if(name.size()<key+24 || name.size()>key+32 || !name.ends_with(".json") || name[key+17]!='-') return false;
        for(size_t i=0;i<key+17;++i) if(i!=key && !hex(name[i])) return false;
        for(size_t i=key+18;i<name.size()-5;++i) if(name[i]<'0' || name[i]>'9') return false;
        return true;
    }
    RE::NiPoint3 point(const json& v) {
        need(v.is_array() && v.size() == 3, "invalid vector");
        RE::NiPoint3 p{v.at(0), v.at(1), v.at(2)};
        need(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z), "nonfinite vector"); return p;
    }
    bool closeEnough(float a, float b, float tolerance = 0.001f) { return std::isfinite(a) && std::isfinite(b) && std::abs(a-b) <= tolerance; }
    bool closeEnough(const RE::NiPoint3& a, const RE::NiPoint3& b, float tolerance) { return closeEnough(a.x,b.x,tolerance) && closeEnough(a.y,b.y,tolerance) && closeEnough(a.z,b.z,tolerance); }
    bool closeEnough(const RE::NiTransform& a, const RE::NiTransform& b) {
        if (!closeEnough(a.translate,b.translate,0.05f) || !closeEnough(a.scale,b.scale,0.0005f)) return false;
        for (int i=0;i<3;++i) for(int j=0;j<3;++j) if(!closeEnough(a.rotate.entry[i][j],b.rotate.entry[i][j],0.0001f)) return false;
        return true;
    }
    RE::NiTransform transform(const json& j) {
        RE::NiTransform t; t.translate=point(j.at("position")); t.scale=j.at("scale");
        need(std::isfinite(t.scale) && t.scale>0 && t.scale<=100, "invalid scale");
        const auto& matrix=j.at("matrix"); need(matrix.size()==3, "invalid matrix");
        for(int i=0;i<3;++i) { const auto row=point(matrix.at(i)); t.rotate.entry[i][0]=row.x; t.rotate.entry[i][1]=row.y; t.rotate.entry[i][2]=row.z; }
        return t;
    }
    uint64_t fnv1a(const std::string& bytes) {
        uint64_t hash=14695981039346656037ull;
        for(const unsigned char c:bytes) hash=(hash^c)*1099511628211ull;
        return hash;
    }
    std::string readFile(const fs::path& path) {
        std::ifstream in(path,std::ios::binary); need(in.good(),"cannot open file");
        return std::string(std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>());
    }
    bool vanilla(const RE::TESForm* form) {
        const auto* file=form ? form->GetFile() : nullptr;
        return file && normalized(std::string(file->GetFilename()))=="skyrim.esm";
    }
    // True when nothing below `object` can render or animate: every leaf is app-culled and there
    // are no controllers or switch nodes. Any other un-culled leaf (geometry, light, particles,
    // unknown types) counts as visible, so its reference root is never culled.
    bool FullyHidden(RE::NiAVObject* object,unsigned depth=0) {
        if(!object || depth>64) return false;
        if(object->GetAppCulled()) return true;
        if(object->controllers) return false;
        auto* node=object->AsNode();
        if(!node || node->AsSwitchNode()) return false;
        for(const auto& child:node->GetChildren()) if(child && !FullyHidden(child.get(),depth+1)) return false;
        return true;
    }

    // SE 1.5.97 UpdateWorldData (0xC61A50) copies the old world into previousWorld before computing the new one and
    // reseeds only a zero previousWorld (or a node flagged 0x40 in flags02). A freshly cloned chunk is updated once, so
    // it would keep the model's own pre-placement transform as previousWorld and report motion from there to its
    // placement, which TAA reprojects as movement (FO4CS fixed the same in 1e1e5741). Chunks never move.
    void SeedPreviousWorld(RE::NiAVObject* object,unsigned depth=0) {
        if(!object || depth>64) return;
        object->previousWorld=object->world;
        if(auto* node=object->AsNode()) for(const auto& child:node->GetChildren()) SeedPreviousWorld(child.get(),depth+1);
    }

    struct Source {
        uint32_t refID{}, baseID{}, vertices{}, triangles{}, bytes{};
        uint64_t hash{};
        std::string model, modelKey;  // modelKey is normalized once, at load
        std::string checkKey;  // Manager::CheckKey of model, bytes and hash, built once at load
        std::vector<std::string> path;
        RE::NiPoint3 position, rotation;
        float scale{};
        RE::NiTransform world;
        // Plugin-relative sources were baked from the load order's winning records, so an override passes when every
        // checked field still matches; old (Skyrim.esm) sources keep the rule that any override rejects.
        bool relative=false;
        // Optional "enable_parent": {"ref": <form>, "opposite": <bool>}: the source's live enable parent must be this
        // reference, with this flag. Chunks group sources by parent, so the whole chunk follows one state.
        uint32_t enableParent=0;
        bool enableOpposite=false;
        // Optional "alternate_textures": [{"name3d", "index3d", "texture_set": <form>}]: the base's alternate-texture
        // (MODS) list it was baked with. The base's live list must be the same; the material check then compares the
        // swapped texture paths. Without it, a base with alternate textures rejects.
        struct Alternate { std::string name3D; uint32_t index3D=0; uint32_t textureSet=0; };
        std::vector<Alternate> alternates;
    };
    // Precomputed visibility for one cell's chunks: for each view cell (a grid box the camera
    // can occupy), the chunks visible from anywhere inside it. Keys are global grid indices.
    struct Previs {
        float size{};
        size_t words{};
        std::unordered_map<uint64_t,std::vector<uint64_t>> visible;
        std::vector<size_t> definitions;  // bit index -> definition index
        // Runtime exterior previs: merged from tables baked from several camera cells; a view cell in a camera cell that
        // no table covers is unknown (the bake may not have run there yet), so nothing is hidden near it.
        bool exterior=false;
        std::set<std::pair<int32_t,int32_t>> covered;  // camera cells (grid x, y) whose view cells are in `visible`
    };
    // Reference previs (Fallout 4 style, every static reference): Previs/References/<cell>.json lists the cell's static,
    // tree and movable-static references with the positions they were baked at. A target is used only after its
    // reference was found at that position with such a base once its cell attached; anything else is never hidden.
    struct ReferencePrevis {
        Previs pvs;  // pvs.definitions stays empty; bit k is forms[k]
        std::vector<uint32_t> forms;
        std::vector<RE::NiPoint3> positions;
        std::vector<bool> usable;
        bool checked=false;
    };
    uint64_t viewCellKey(int64_t x,int64_t y,int64_t z) {
        constexpr int64_t bias=1<<20, mask=(1<<21)-1;
        return (static_cast<uint64_t>((x+bias)&mask)<<42)|(static_cast<uint64_t>((y+bias)&mask)<<21)|static_cast<uint64_t>((z+bias)&mask);
    }
    std::vector<uint8_t> base64(const std::string& text) {
        std::vector<uint8_t> out; uint32_t buffer=0; int bits=0;
        for(const char c:text) {
            int v;
            if(c>='A'&&c<='Z') v=c-'A'; else if(c>='a'&&c<='z') v=c-'a'+26; else if(c>='0'&&c<='9') v=c-'0'+52;
            else if(c=='+') v=62; else if(c=='/') v=63; else if(c=='=') break; else throw std::runtime_error("invalid base64");
            buffer=(buffer<<6)|static_cast<uint32_t>(v); bits+=6;
            if(bits>=8) { bits-=8; out.push_back(static_cast<uint8_t>((buffer>>bits)&0xFF)); }
        }
        return out;
    }

    struct Definition {
        uint32_t cell{}, vertices{}, triangles{}, bytes{};
        uint64_t hash{};
        bool exterior=false;  // Exterior sources fade with distance; that is not an invalidation.
        std::string mesh;
        std::string parseKey;  // "parse:" + mesh, the preload key of its parse, built once at load
        RE::NiPoint3 origin;
        std::vector<Source> sources;
    };
    // Runtime-bake parts append definitions during play. A reallocation moves each Definition, which keeps its
    // sources' buffer, so ReferenceCheck::expected (a pointer into it) stays valid.
    static_assert(std::is_nothrow_move_constructible_v<Definition>);

    // Engine-internal cell event (not the per-reference Papyrus TESCellAttachDetachEvent).
    // SE 1.5.97: the source lives at *(ID 514281)+8. Cell attach (RVA 263BE0) sends 0 then 1;
    // cell detach (RVA 26C5D0, ID 18655) sends 2 before any 3D teardown and 3 at the end.
    struct CellAttachDetachEvent {
        enum Status : uint32_t { kPreAttach=0, kPostAttach=1, kPreDetach=2, kPostDetach=3 };
        RE::TESObjectCELL* cell;
        uint32_t status;
    };
    static_assert(sizeof(CellAttachDetachEvent)==0x10);
    struct ReferenceCheck {
        RE::TESObjectREFR* reference{};
        RE::NiAVObject* root{};
        const RE::TESBoundObject* base{};
        const RE::TESObjectCELL* cell{};
        const RE::NiNode* parent{};
        const Source* expected{};  // Definitions are immutable while active.
        // A source with a declared enable parent stays valid only while the parent's state enables it.
        RE::NiPointer<RE::TESObjectREFR> enableParent;
        ValidationResult result;
    };
    struct AncestorCheck {
        // Hold the original hierarchy alive even when the engine replaces 3D.
        RE::NiPointer<RE::NiAVObject> node;
        const RE::NiNode* parent{};
        RE::BSFadeNode* fade{};
        ValidationResult result;
    };
    struct Binding {
        RE::NiPointer<RE::TESObjectREFR> reference;
        RE::NiPointer<RE::NiAVObject> root;
        RE::NiPointer<RE::BSTriShape> shape;
        const RE::TESBoundObject* base{};
        const RE::TESObjectCELL* cell{};
        std::shared_ptr<ReferenceCheck> referenceCheck;
        std::vector<std::shared_ptr<AncestorCheck>> ancestors;
        ValidatedSnapshot<RE::NiTransform> worldCheck;
    };
    struct Active {
        size_t definition{};
        bool valid=true;
        RE::NiPointer<RE::NiNode> parent, mesh;
        RE::NiTransform parentWorld;
        ValidatedSnapshot<RE::NiTransform> parentWorldCheck;
        std::vector<Binding> sources;
        RE::NiPointer<RE::ShadowSceneNode> lightingScene;
        uint64_t materialFlags=0;  // the chunk's shader flags as loaded (its sources' NIF material)
    };
    // A rejected or dissolved chunk is retried with exponential backoff (0.5 s to
    // 60 s). Every attempt repeats the full acceptance checks, so a source that is
    // still disabled, moved, fading or hidden keeps its originals visible.
    struct Slot {
        bool active=false;
        uint8_t failures=0;
        Clock::time_point retryAt{};
        // Leading sources whose fingerprints are known (Manager::Prepared). Fingerprints are never forgotten, so a pending
        // chunk's later checks start here.
        uint32_t prepared=0;
    };

    class Manager : public RE::BSTEventSink<RE::InputEvent*>, public RE::BSTEventSink<CellAttachDetachEvent> {
    public:
        std::atomic_bool suspended{true}, queued{false};
        // `failed`: precombines switched themselves off after an error (the tick task), until switched on again; the menu
        // shows it (Settings::Status). `announce`: the precombines key asks SetEnabled's task to report the state it reached.
        std::atomic_bool failed{false}, announce{false};
        bool enabled=true, started=false, cellSinkRegistered=false;
        // Developer mode (config "developer_mode", default false): the 10 s diagnostic reports. The shortcut keys work while
        // it or bDeveloperShortcuts is on (Settings::ShortcutsOn).
        bool developerMode=false;
        // "frame_log" (default false): Data/../SKSE log folder SkyrimPrecombines_frames.csv, one row per second with every
        // frame time, which features are on, the cell and the player position (a manual play-through has no other record).
        bool frameLog=false;
        std::ofstream frameFile;
        // "hiz_ab" (developer): SkyrimPrecombines_ab.csv in the SKSE log folder, one row per finished A/B block.
        bool abLog=false;
        std::ofstream abFile;
        std::vector<Occlusion::AbBlock> abBatch;
        void WriteAbRows() {
            if(!abLog) return;
            Occlusion::TakeAbBlocks(abBatch);
            if(abBatch.empty()) return;
            if(!abFile.is_open()) {
                const auto dir=SKSE::log::log_directory();
                if(!dir) { abLog=false; return; }
                abFile.open(*dir/"SkyrimPrecombines_ab.csv",std::ios::trunc);
                abFile<<"utc_ms,round,state,frames,mean_ms,max_ms,tested,culled,precombines,cell,cell_name,x,y,z\n";
            }
            uint32_t cellID=0; std::string cellName;
            if(auto* player=RE::PlayerCharacter::GetSingleton())
                if(auto* cell=player->GetParentCell()) { cellID=cell->GetFormID(); if(const char* id=cell->GetFormEditorID()) cellName=id; }
            std::erase(cellName,',');
            const auto utc=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            static constexpr const char* names[4]={"off","new","before","eligible_first"};
            for(const auto& b:abBatch)
                abFile<<std::format("{},{},{},{},{:.4f},{:.2f},{},{},{},{:08X},{},{:.0f},{:.0f},{:.0f}\n",utc,b.round,names[uint8_t(b.state)],b.frames,
                    b.frames ? b.sumMs/b.frames : 0.0,b.maxMs,b.tested,b.culled,enabled && !suspended.load() ? 1 : 0,cellID,cellName,b.eye[0],b.eye[1],b.eye[2]);
            abFile.flush();
        }
        Clock::time_point lastFrameRow{};
        std::vector<float> frameBatch;
        void WriteFrameRow(Clock::time_point now) {
            if(!frameLog || now-lastFrameRow<std::chrono::seconds(1)) return;
            lastFrameRow=now;
            Occlusion::TakeFrameTimes(frameBatch);
            if(!frameFile.is_open()) {
                const auto dir=SKSE::log::log_directory();
                if(!dir) { frameLog=false; return; }
                frameFile.open(*dir/"SkyrimPrecombines_frames.csv",std::ios::trunc);
                frameFile<<"utc_ms,frames,mean_ms,max_ms,menu,precombines,hiz,previs,active_chunks,hidden_sources,hiz_tested,hiz_culled,cell,cell_name,x,y,z,frame_ms\n";
            }
            if(frameBatch.empty()) return;
            double sum=0; float worst=0;
            for(const float ms:frameBatch) { sum+=ms; worst=std::max(worst,ms); }
            auto* ui=RE::UI::GetSingleton();
            const bool menu=!ui || ui->GameIsPaused() || ui->IsMenuOpen("Loading Menu") || ui->IsMenuOpen("Main Menu") || ui->IsMenuOpen("Console");
            const auto hiz=Occlusion::Read();
            uint32_t cellID=0; std::string cellName; RE::NiPoint3 at{};
            if(auto* player=RE::PlayerCharacter::GetSingleton()) {
                at=player->GetPosition();
                if(auto* cell=player->GetParentCell()) { cellID=cell->GetFormID(); if(const char* id=cell->GetFormEditorID()) cellName=id; }
            }
            size_t hidden=0; for(const auto& chunk:active) hidden+=chunk.sources.size();
            const auto utc=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            std::string times; times.reserve(frameBatch.size()*6);
            for(const float ms:frameBatch) { if(!times.empty()) times+=';'; times+=std::format("{:.2f}",ms); }
            std::erase(cellName,',');
            frameFile<<std::format("{},{},{:.3f},{:.2f},{},{},{},{},{},{},{},{},{:08X},{},{:.0f},{:.0f},{:.0f},{}\n",utc,frameBatch.size(),sum/frameBatch.size(),worst,
                menu ? 1 : 0,enabled && !suspended.load() ? 1 : 0,hiz.enabled ? 1 : 0,previs ? 1 : 0,active.size(),hidden,
                hiz.fadeTested+hiz.geometryTested,hiz.fadeCulled+hiz.geometryCulled,cellID,cellName,at.x,at.y,at.z,times);
            frameFile.flush();
        }
        // A reference whose every shape is merged into active chunks has its root culled, so the
        // engine skips its traversal and fade work too, not only its draws (config
        // "cull_covered_roots"). Only roots this plugin culled are ever un-culled.
        bool cullRoots=true;
        std::unordered_map<RE::NiAVObject*,RE::NiPointer<RE::NiAVObject>> culledRoots;
        bool syncLights=true;  // config "sync_source_lights"
        // Previs (config "previs"): chunks outside the camera's view-cell neighbourhood are culled.
        bool previs=true;
        std::unordered_map<uint32_t,Previs> previsByCell;
        std::vector<int64_t> previsBit;  // definition index -> bit index in its cell's table, or -1
        size_t previsHidden=0;
        std::unordered_map<uint32_t,ReferencePrevis> referencePrevisByCell;
        size_t referencePrevisHidden=0;
        std::set<uint32_t> attachedCells;  // every attached cell this task, with or without chunk definitions
        // Developer shortcut keys, DirectInput scan codes from Settings (0 = none; its defaults when the INI has no line).
        // They work only while Settings::ShortcutsOn.
        uint32_t toggleKey=68;  // config "toggle_key", default 68 = F10
        uint32_t previsToggleKey=0;  // config "previs_toggle_key" (e.g. 87 = F11), no default
        uint32_t hizToggleKey=0;  // config "hiz_toggle_key", default 66 = F8
        uint32_t hizGeometryToggleKey=0;  // config "hiz_geometry_toggle_key", default 65 = F7: per-geometry Hi-Z tests
        uint32_t boxToggleKey=0;  // config "box_occlusion_toggle_key", default 64 = F6 on SE and AE: the GPU box test
        uint32_t probeKey=0;  // config "hiz_probe_key" (needs "hiz_diagnostics"), no default: log what Hi-Z decided under the crosshair
        std::chrono::microseconds activationBudget{3000};  // config "activation_budget_us": no new chunk activation after this much of a task
        // Attached cells with chunk definitions, recomputed every task from the engine's own
        // interior cell and loaded grid. Several exterior cells are attached at once.
        std::set<uint32_t> eligibleCells;
        // Between pre-detach and post-detach/attach. A task on another thread must not
        // re-attach a chunk in the instant between our cleanup and the engine's teardown.
        std::set<uint32_t> detachingCells;
        std::set<uint32_t> excluded;
        std::vector<Definition> definitions;
        std::unordered_map<uint32_t,std::vector<size_t>> byCell;
        std::unordered_map<uint32_t,uint64_t> manifestHashes;  // FNV-1a of each accepted manifest's bytes
        std::vector<Slot> slots;
        // Runtime-bake hand-off (config "runtime_bake", default true like the shipped bRuntimeBake=1; off with bEnabled=0
        // outside developer mode). Part names arrive on the bake plugin's worker thread (SkyrimPrecombines_RuntimeBakePart);
        // the next task loads them under stateMutex.
        std::atomic_bool runtimeBake{false};
        std::mutex runtimePartsMutex;
        // Parsed on the bake plugin's worker thread (SkyrimPrecombines_RuntimeBakePart), or on the cache reader for a cached
        // cell (`cached`): a part is up to a few MB of JSON, tens of milliseconds to parse, which made a frame on the game
        // thread (28-62 ms frames at each part's hand-off). A null manifest means the read or parse failed;
        // the task reports it.
        struct PendingPart { std::string name; std::shared_ptr<const json> manifest; std::string error; bool cached=false; };
        std::deque<PendingPart> runtimePartsPending;
        std::vector<std::string> runtimePrevisPending;  // SkyrimPrecombines_RuntimeBakePrevis, under runtimePartsMutex
        struct RuntimeCell { std::string bake; std::set<size_t> parts; std::set<std::string> coverage; };
        std::unordered_map<uint32_t,RuntimeCell> runtimeCells;  // cells whose chunks come from runtime parts
        std::set<std::string> runtimeMeshes;
        // Cached cells (the bake's index.json, read at data load) whose parts are not read yet. A cell's parts, then its
        // previs tables, are read on the cache reader when the cell is first attached and handed to the tasks as the bake's
        // own parts are, so a session costs time and memory only for the cells it visits (every cached part
        // was parsed on the main thread at data load and every chunk mesh preloaded, wherever the player was).
        struct CachedCell { std::vector<std::string> parts, tables; };
        std::unordered_map<uint32_t,CachedCell> cachedCells;      // by cell form ID; a cell leaves it when requested
        std::unordered_map<std::string,uint32_t> cachedKeys;      // index.json cell key -> cell form ID
        // The cache reader's work, under runtimePartsMutex. runtimeReading counts the cells requested and not yet handed
        // over in full; previs tables wait for it to reach 0.
        std::deque<CachedCell> runtimeRequests;
        std::condition_variable runtimeReader;
        size_t runtimeReading=0;
        static constexpr size_t kCachedPartsWaiting=8;  // parsed parts waiting for a task at most (a large one is tens of MB as JSON)
        static constexpr auto kPartBudget=std::chrono::milliseconds(2);  // part registration per task after the first part
        // Config "material_check" (default true): a source whose live material differs from the baked one (runtime
        // snow, texture swaps) keeps the originals.
        bool materialCheck=true;
        std::vector<std::weak_ptr<ReferenceCheck>> parentedChecks;  // checked every task, not every fifth
        std::vector<Active> active;
        ValidationSchedule validation;
        // Weak entries cannot keep unloaded game objects alive. These maps are
        // used only when attaching; periodic checks follow the saved pointers.
        std::unordered_map<uint32_t,std::weak_ptr<ReferenceCheck>> referenceChecks;
        ValidationBuckets<ReferenceCheck> referenceSchedule;
        std::unordered_map<RE::NiAVObject*,std::weak_ptr<AncestorCheck>> ancestorChecks;
        std::map<std::string,bool> meshChecks;  // Resource fingerprints; files do not change during a session.
        std::map<std::string,size_t> rejections;
        Clock::time_point lastReport{};
        std::vector<size_t> candidates;  // reused activation order across eligible cells
        size_t cursor=0, dissolved=0;
        std::vector<uint32_t> tickMicros;  // since the last report
        uint64_t stageTicks=0, referenceNs=0, sceneNs=0, activationNs=0;
        std::set<uint32_t> tickThreads;
        std::recursive_mutex stateMutex;
        // Diagnostic counters, readable without the state lock.
        std::atomic<uint64_t> statTicks{0}, statTickNs{0}, statTickMaxNs{0}, statAttaches{0}, statDissolves{0};
        std::atomic<uint64_t> statLightsSynced{0};  // lights given to chunks from their sources (sync_source_lights)
        std::atomic<uint32_t> statActive{0}, statHidden{0}, statThread{0}, statCulled{0}, statPrevis{0};

        static Manager& Get() { static auto* manager=new Manager; return *manager; }

        // One manifest's chunks, validated completely before any is used; shared by the manifest folder and runtime-bake
        // parts. Chunk meshes must live under `prefix`. A chunk whose forms name a plugin that is not loaded is skipped
        // (counted); anything malformed throws, which rejects the whole manifest.
        struct Parsed {
            uint32_t cell=0;
            bool interior=true;
            size_t unloaded=0;
            std::vector<Definition> definitions;
        };
        static Parsed ParseManifest(const json& manifest,std::set<std::string>& coverage,std::set<std::string>& meshes,std::string_view prefix) {
            Parsed parsed;
            Plugins plugins;  // the plugins this manifest names, each looked up once
            const int schema=manifest.at("schema");
            parsed.interior=manifest.at("interior");
            const auto& cell=manifest.at("cell");
            // Old manifests name Skyrim.esm forms only; plugin-relative ones may come from any plugin of the load order.
            need((relativeForm(cell) || manifest.at("plugin")=="Skyrim.esm") && ((schema==1 && parsed.interior) || (schema==2 && !parsed.interior)), "unsupported manifest");
            parsed.cell=resolveForm(cell,&plugins);
            need(parsed.cell!=0,"the cell's plugin is not loaded");
            if(!parsed.interior) {
                const auto& grid=manifest.at("grid");
                need(grid.is_array() && grid.size()==2 && grid.at(0).is_number_integer() && grid.at(1).is_number_integer() && resolveForm(manifest.at("worldspace").at("form_id"),&plugins)!=0, "invalid exterior cell");
            }
            for (const auto& item : manifest.at("chunks")) {
                Definition def; def.cell=parsed.cell; def.exterior=!parsed.interior; def.mesh=item.at("mesh"); def.parseKey="parse:"+def.mesh; def.origin=point(item.at("origin"));
                need(safeMesh(def.mesh) && normalized(def.mesh).starts_with(prefix) && meshes.insert(normalized(def.mesh)).second, "invalid or duplicate chunk path");
                def.vertices=item.at("vertices"); def.triangles=item.at("triangles"); def.bytes=item.at("mesh_bytes");
                def.hash=std::stoull(item.at("mesh_hash").get<std::string>(),nullptr,16);
                need(def.vertices>0 && def.vertices<=65535 && def.triangles>0 && def.triangles<=65535 && def.bytes>0 && def.bytes<=64*1024*1024, "invalid geometry limits");
                uint64_t totalVertices=0,totalTriangles=0;
                bool loaded=true;
                for (const auto& j : item.at("sources")) {
                    Source source;
                    source.refID=resolveForm(j.at("ref"),&plugins); source.baseID=resolveForm(j.at("base"),&plugins);
                    source.relative=relativeForm(j.at("ref")) || relativeForm(j.at("base"));
                    if(j.contains("enable_parent")) {
                        const auto& parent=j.at("enable_parent");
                        source.enableParent=resolveForm(parent.at("ref"),&plugins); source.enableOpposite=parent.value("opposite",false);
                        loaded=loaded && source.enableParent!=0;
                    }
                    if(j.contains("alternate_textures")) {
                        for(const auto& entry:j.at("alternate_textures")) {
                            Source::Alternate alternate{entry.at("name3d").get<std::string>(),static_cast<uint32_t>(entry.at("index3d").get<int64_t>()),resolveForm(entry.at("texture_set"),&plugins)};
                            loaded=loaded && alternate.textureSet!=0;
                            source.alternates.push_back(std::move(alternate));
                        }
                    }
                    loaded=loaded && source.refID!=0 && source.baseID!=0;
                    source.model=j.at("model"); source.modelKey=normalized(source.model);
                    source.path=j.at("path").get<std::vector<std::string>>();
                    source.position=point(j.at("position")); source.rotation=point(j.at("rotation")); source.scale=j.at("scale"); source.world=transform(j.at("world"));
                    source.vertices=j.at("vertices"); source.triangles=j.at("triangles"); source.bytes=j.at("mesh_bytes"); source.hash=std::stoull(j.at("mesh_hash").get<std::string>(),nullptr,16);
                    need(safeMesh(source.model) && !source.path.empty() && source.path.size()<=32 && std::isfinite(source.scale) && source.scale>0, "invalid source shape");
                    need(source.bytes>0 && source.bytes<=64*1024*1024 && source.vertices>0 && source.triangles>0, "invalid source limits");
                    source.checkKey=CheckKey(source.model,source.bytes,source.hash);
                    const auto identity=source.refID ? std::to_string(source.refID) : j.at("ref").dump();
                    need(coverage.insert(identity+j.at("path").dump()).second, "source shape covered more than once");
                    totalVertices+=source.vertices; totalTriangles+=source.triangles;
                    def.sources.push_back(std::move(source));
                }
                need(def.sources.size()>=2 && totalVertices==def.vertices && totalTriangles==def.triangles, "source totals differ from chunk");
                if(loaded) parsed.definitions.push_back(std::move(def));
                else ++parsed.unloaded;
            }
            return parsed;
        }

        // SkyrimPrecombines.ini through readers that never throw: a value of the wrong type keeps its default with one
        // warning, so one bad line costs only that setting (enabled=1 threw out of here before Hi-Z was
        // configured, leaving precombines, previs and Hi-Z off). Release defaults: precombines, the runtime bake and Hi-Z
        // on, previs off (no previs tables are shipped). The settings and keys the in-game menu shows come from Settings,
        // which read them with the same readers at plugin load: the INI's values, or the menu's if it changed them already.
        void ReadSettings() {
            enabled=Settings::GetBool("bEnabled"); toggleKey=static_cast<uint32_t>(Settings::GetInt("toggle_key"));
            developerMode=Config::Flag("developer_mode",false);
            cullRoots=Config::Flag("cull_covered_roots",true); syncLights=Config::Flag("sync_source_lights",true);
            previs=Config::Flag("previs",false); previsToggleKey=static_cast<uint32_t>(Settings::GetInt("previs_toggle_key"));
            // Previs hides chunks through the main-camera culling hooks, which VR does not get yet (Occlusion::Install);
            // the AppCulled fallback would also hide them from water reflections.
            if(REL::Module::IsVR() && (previs || previsToggleKey)) {
                logger::warn("Previs is not available on Skyrim VR yet; it stays off");
                previs=false; previsToggleKey=0;
            }
            activationBudget=std::chrono::microseconds(Config::Number("activation_budget_us",3000u));
            ConfigureHiZ();
            for(const auto cell:Config::FormIds("excluded_cells")) excluded.insert(cell);
            materialCheck=Config::Flag("material_check",true);
            // bEnabled=0 stops the bake's hand-off too, except in developer mode, where the toggle key can turn precombines
            // on in play (the bake kept merging every cell visited and its parts were read for nothing). The
            // default and the rule match the bake's own switch (BakePlugin.cpp). Turned on later in play (the menu, the
            // shortcut key), precombines read the cached cells then (ResumeRuntimeHandOff).
            runtimeBake=Config::Flag("runtime_bake",true) && (enabled || developerMode);
        }
        // Hi-Z in a block of its own: nothing else read here can leave it unconfigured. A failure leaves it at
        // its compiled default, off.
        void ConfigureHiZ() {
            try {
                Occlusion::Settings hiz;
                hiz.enabled=Settings::GetBool("bHiZ"); hiz.fadeNodes=Config::Flag("hiz_fade_nodes",true);
                hiz.fadeRule=Config::Number("hiz_fade_rule",1u);
                hiz.geometry=Config::Flag("hiz_geometry",true); hiz.gridWidth=Config::Number("hiz_grid_width",128u);
                hiz.nearExempt=Config::Number("hiz_near_exempt",64.f); hizToggleKey=static_cast<uint32_t>(Settings::GetInt("hiz_toggle_key"));
                hizGeometryToggleKey=static_cast<uint32_t>(Settings::GetInt("hiz_geometry_toggle_key"));
                hiz.excludeMoving=Config::Flag("hiz_exclude_moving",true); hiz.waitMicros=Config::Number("hiz_wait_us",0u); hiz.splitBuild=Config::Flag("hiz_split_build",false);
                hiz.minFrameMs=Settings::GetFloat("fHiZMinFrameMs");
                hiz.visibleRetest=Config::Number("hiz_visible_retest",8u); hiz.diagnostics=Config::Flag("hiz_diagnostics",false);
                hiz.audit=Config::Flag("hiz_audit",false); hiz.auditEvery=Config::Number("hiz_audit_every",8u);
                hiz.frameLog=frameLog=Config::Flag("frame_log",false);
                hiz.ab=abLog=Config::Flag("hiz_ab",false); hiz.abFrames=Config::Number("hiz_ab_frames",60u); hiz.abSkip=Config::Number("hiz_ab_skip",8u);
                hiz.minYield=Settings::GetFloat("fHiZMinYield"); hiz.probeInterval=Config::Number("hiz_probe_interval",32u);
                hiz.probeBurst=Config::Number("hiz_probe_burst",4u);
                hiz.dumpSeconds=Config::Number("hiz_dump_seconds",0.f); hiz.dumpMax=Config::Number("hiz_dump_max",40u);
                // Developer: the GPU box test (Occlusion::Boxes; SE and AE).
                hiz.boxes=Config::Flag("box_occlusion",false); hiz.boxMaxShift=Config::Number("box_occlusion_max_shift",16.f);
                hiz.boxDownsample=Config::Number("box_occlusion_downsample",2u); boxToggleKey=static_cast<uint32_t>(Settings::GetInt("box_occlusion_toggle_key"));
                probeKey=static_cast<uint32_t>(Settings::GetInt("hiz_probe_key"));
                const auto& config=Config::Get();
                if(const auto mode=config.find("hiz_mode");mode!=config.end() && !(mode->is_string() && mode->get<std::string>()=="cpu"))
                    logger::warn("\"hiz_mode\": {} is no longer supported; Hi-Z uses CPU mode",mode->dump(-1,' ',false,json::error_handler_t::replace));
                Occlusion::Configure(hiz);
            } catch(const std::exception& e) { logger::error("Hi-Z occlusion settings could not be applied ({}); Hi-Z stays off",e.what()); }
        }

        void ReadFiles() {
            ReadSettings();
            if (!fs::exists(manifestDir)) { logger::info("No generated manifests; plugin remains idle"); return; }
            std::set<std::string> coverage, meshes;
            size_t unloaded=0;
            for (const auto& entry : fs::directory_iterator(manifestDir)) {
                if (normalized(entry.path().extension().string())!=".json") continue;
                // Validate a complete manifest before adding any of its chunks.
                auto nextCoverage=coverage, nextMeshes=meshes;
                try {
                    need(fs::file_size(entry.path()) <= 64*1024*1024, "manifest too large");
                    const auto raw=readFile(entry.path());
                    auto parsed=ParseManifest(json::parse(raw),nextCoverage,nextMeshes,"precombined/skyrim/");
                    need(!manifestHashes.contains(parsed.cell),"second manifest for one cell");
                    definitions.insert(definitions.end(),std::make_move_iterator(parsed.definitions.begin()),std::make_move_iterator(parsed.definitions.end()));
                    coverage=std::move(nextCoverage); meshes=std::move(nextMeshes);
                    manifestHashes.emplace(parsed.cell,fnv1a(raw));
                    unloaded+=parsed.unloaded;
                } catch (const std::exception& e) { logger::error("Rejected manifest {}: {}",entry.path().string(),e.what()); }
            }
            if(unloaded) logger::info("Skipped {} chunks whose sources name plugins that are not loaded",unloaded);
            slots.assign(definitions.size(),{});
            for (size_t i=0;i<definitions.size();++i) byCell[definitions[i].cell].push_back(i);
            ReadPrevis();
            ReadReferencePrevis();
            const auto exterior=std::count_if(definitions.begin(),definitions.end(),[](const auto& d){ return d.exterior; });
            logger::info("Loaded {} chunk definitions ({} exterior) in {} cells; enabled={}, toggle scan code={}, previs={} (toggle scan code {}), {} previs tables, {} reference previs tables",
                definitions.size(),exterior,byCell.size(),enabled,toggleKey,previs,previsToggleKey,previsByCell.size(),referencePrevisByCell.size());
        }

        // `Previs/<cell>.json` covers one cell's chunks and names the exact manifest it was baked
        // from (FNV-1a of its bytes). Any mismatch with the loaded definitions rejects the whole
        // table, which only means no culling for that cell.
        // Bit b of a view cell is bit (b % 8) of byte (b / 8) of its base64 string.
        // A table's view cells: [ix, iy, iz, base64 bits], bit b for pvs.definitions[b].
        static void DecodeViewCells(Previs& pvs,const json& cells) {
            pvs.words=(pvs.definitions.size()+63)/64;
            for(const auto& item:cells) {
                need(item.is_array() && item.size()==4,"invalid view cell");
                const auto bytes=base64(item.at(3).get<std::string>());
                need(bytes.size()*8>=pvs.definitions.size(),"view cell bitset too short");
                std::vector<uint64_t> words(pvs.words);
                for(size_t b=0;b<pvs.definitions.size();++b) if(bytes[b/8]&(1u<<(b%8))) words[b/64]|=1ull<<(b%64);
                need(pvs.visible.emplace(viewCellKey(item.at(0),item.at(1),item.at(2)),std::move(words)).second,"duplicate view cell");
            }
            need(!pvs.visible.empty(),"previs table has no view cells");
        }
        void ReadPrevis() {
            previsBit.assign(definitions.size(),-1);
            if(!fs::exists(previsDir)) return;
            std::unordered_map<std::string,size_t> byMesh;
            for(size_t i=0;i<definitions.size();++i) byMesh.emplace(normalized(definitions[i].mesh),i);
            for(const auto& entry:fs::directory_iterator(previsDir)) {
                if(normalized(entry.path().extension().string())!=".json") continue;
                try {
                    need(fs::file_size(entry.path())<=256*1024*1024,"previs table too large");
                    const auto table=json::parse(readFile(entry.path()));
                    need(table.at("schema")==1 && table.at("kind")=="precombine_pvs","unsupported previs table");
                    const uint32_t cell=table.at("cell");
                    need(!previsByCell.contains(cell),"second previs table for one cell");
                    const auto manifest=manifestHashes.find(cell);
                    need(manifest!=manifestHashes.end() && std::stoull(table.at("manifest_fnv1a").get<std::string>(),nullptr,16)==manifest->second,
                         "previs table was baked from another manifest");
                    Previs pvs; pvs.size=table.at("view_cell_size");
                    need(std::isfinite(pvs.size) && pvs.size>=32 && pvs.size<=4096,"invalid view cell size");
                    std::set<size_t> seen;
                    for(const auto& mesh:table.at("objects")) {
                        const auto found=byMesh.find(normalized(mesh.get<std::string>()));
                        need(found!=byMesh.end() && definitions[found->second].cell==cell && seen.insert(found->second).second,"previs object is not a unique chunk of this cell");
                        pvs.definitions.push_back(found->second);
                    }
                    DecodeViewCells(pvs,table.at("view_cells"));
                    for(size_t b=0;b<pvs.definitions.size();++b) previsBit[pvs.definitions[b]]=static_cast<int64_t>(b);
                    logger::info("Previs for cell {:08X}: {} chunks, {} view cells of {} units",cell,pvs.definitions.size(),pvs.visible.size(),pvs.size);
                    previsByCell.emplace(cell,std::move(pvs));
                } catch(const std::exception& e) { logger::error("Rejected previs table {}: {}",entry.path().string(),e.what()); }
            }
        }

        void ReadReferencePrevis() {
            const auto folder=fs::path(previsDir)/"References";
            if(!fs::exists(folder)) return;
            for(const auto& entry:fs::directory_iterator(folder)) {
                if(normalized(entry.path().extension().string())!=".json") continue;
                try {
                    need(fs::file_size(entry.path())<=256*1024*1024,"reference previs table too large");
                    const auto table=json::parse(readFile(entry.path()));
                    need(table.at("schema")==1 && table.at("kind")=="reference_pvs" && table.at("plugin")=="Skyrim.esm","unsupported reference previs table");
                    const uint32_t cell=table.at("cell");
                    need(!referencePrevisByCell.contains(cell),"second reference previs table for one cell");
                    ReferencePrevis rp; rp.pvs.size=table.at("view_cell_size");
                    need(std::isfinite(rp.pvs.size) && rp.pvs.size>=32 && rp.pvs.size<=4096,"invalid view cell size");
                    std::set<uint32_t> seen;
                    for(const auto& object:table.at("objects")) {
                        const uint32_t form=object.at("form");
                        const auto& at=object.at("position");
                        need(seen.insert(form).second && at.is_array() && at.size()==3,"invalid reference previs object");
                        rp.forms.push_back(form);
                        rp.positions.push_back({at.at(0).get<float>(),at.at(1).get<float>(),at.at(2).get<float>()});
                    }
                    need(!rp.forms.empty(),"reference previs table has no objects");
                    rp.pvs.words=(rp.forms.size()+63)/64;
                    for(const auto& item:table.at("view_cells")) {
                        need(item.is_array() && item.size()==4,"invalid view cell");
                        const auto bytes=base64(item.at(3).get<std::string>());
                        need(bytes.size()*8>=rp.forms.size(),"view cell bitset too short");
                        std::vector<uint64_t> words(rp.pvs.words);
                        for(size_t b=0;b<rp.forms.size();++b) if(bytes[b/8]&(1u<<(b%8))) words[b/64]|=1ull<<(b%64);
                        need(rp.pvs.visible.emplace(viewCellKey(item.at(0),item.at(1),item.at(2)),std::move(words)).second,"duplicate view cell");
                    }
                    need(!rp.pvs.visible.empty(),"reference previs table has no view cells");
                    rp.usable.assign(rp.forms.size(),false);
                    logger::info("Reference previs for cell {:08X}: {} references, {} view cells of {} units",cell,rp.forms.size(),rp.pvs.visible.size(),rp.pvs.size);
                    referencePrevisByCell.emplace(cell,std::move(rp));
                } catch(const std::exception& e) { logger::error("Rejected reference previs table {}: {}",entry.path().string(),e.what()); }
            }
        }

        // Once the cell is attached: each target must still be a static, tree or movable static at its baked position.
        static void CheckReferenceTable(ReferencePrevis& rp,uint32_t cell) {
            size_t usable=0;
            for(size_t k=0;k<rp.forms.size();++k) {
                const auto* reference=RE::TESForm::LookupByID<RE::TESObjectREFR>(rp.forms[k]);
                const auto* base=reference ? reference->GetBaseObject() : nullptr;
                const auto type=base ? base->GetFormType() : RE::FormType::None;
                const bool kind=type==RE::FormType::Static || type==RE::FormType::Tree || type==RE::FormType::MovableStatic;
                rp.usable[k]=kind && reference->GetPosition().GetDistance(rp.positions[k])<=2.f;
                usable+=rp.usable[k];
            }
            rp.checked=true;
            logger::info("Reference previs for cell {:08X}: {} of {} references usable",cell,usable,rp.forms.size());
        }

        // Visible chunks from the camera's view cell and its 26 neighbours. Empty means the camera
        // is outside this table's view cells, so nothing in that cell may be culled.
        static std::vector<uint64_t> Neighbourhood(const Previs& pvs,const RE::NiPoint3& at) {
            const auto ix=static_cast<int64_t>(std::floor(at.x/pvs.size));
            const auto iy=static_cast<int64_t>(std::floor(at.y/pvs.size));
            const auto iz=static_cast<int64_t>(std::floor(at.z/pvs.size));
            std::vector<uint64_t> words;
            if(pvs.exterior)
                for(int64_t dx=-1;dx<=1;++dx) for(int64_t dy=-1;dy<=1;++dy) {
                    const auto cellOf=[&](int64_t i) { return static_cast<int32_t>(std::floor(double(i)*pvs.size/4096.0)); };
                    if(!pvs.covered.contains({cellOf(ix+dx),cellOf(iy+dy)})) return {};
                }
            for(int64_t dx=-1;dx<=1;++dx) for(int64_t dy=-1;dy<=1;++dy) for(int64_t dz=-1;dz<=1;++dz) {
                const auto found=pvs.visible.find(viewCellKey(ix+dx,iy+dy,iz+dz));
                if(found==pvs.visible.end()) continue;
                if(words.empty()) words.assign(pvs.words,0);
                for(size_t w=0;w<pvs.words;++w) words[w]|=found->second[w];
            }
            return words;
        }

        // Runs after activation in every task; a chunk without a table entry is never culled. With the culling
        // hooks installed (SE 1.5.97) previs hides chunks from the main camera only, so water reflections, shadow
        // maps and cubemaps keep drawing them (AppCulled chunks vanished from water reflections).
        void ApplyPrevis() {
            const bool mainCameraOnly=Occlusion::Installed();
            std::vector<RE::NiPointer<RE::NiAVObject>> mainHidden;
            const auto* camera=RE::PlayerCamera::GetSingleton();
            const bool haveCamera=previs && camera && camera->cameraRoot;
            const auto at=haveCamera ? camera->cameraRoot->world.translate : RE::NiPoint3{};
            std::unordered_map<uint32_t,std::vector<uint64_t>> neighbourhoods;
            size_t hidden=0;
            for(auto& chunk:active) {
                const auto bit=previsBit[chunk.definition];
                bool visible=true;
                if(haveCamera && bit>=0) {
                    const auto cell=definitions[chunk.definition].cell;
                    auto found=neighbourhoods.find(cell);
                    if(found==neighbourhoods.end()) found=neighbourhoods.emplace(cell,Neighbourhood(previsByCell.at(cell),at)).first;
                    const auto& words=found->second;
                    visible=words.empty() || ((words[static_cast<size_t>(bit)/64]>>(bit%64))&1);
                }
                if(mainCameraOnly) {
                    if(!visible) mainHidden.emplace_back(chunk.mesh.get());
                } else if(chunk.mesh->GetAppCulled()==visible) chunk.mesh->SetAppCulled(!visible);
                hidden+=!visible;
            }
            previsHidden=hidden;
            if(mainCameraOnly) Occlusion::SetPrevisHidden(std::move(mainHidden));
            // Reference previs needs the main-camera hooks: the engine's own references are never AppCulled.
            if(!mainCameraOnly) return;
            std::vector<uint32_t> forms;
            float size=0.f;
            if(haveCamera) for(const auto id:attachedCells) {
                const auto found=referencePrevisByCell.find(id);
                if(found==referencePrevisByCell.end()) continue;
                auto& rp=found->second;
                if(!rp.checked) CheckReferenceTable(rp,id);
                if(size!=0.f && rp.pvs.size!=size) continue;  // one view cell size per published set
                const auto words=Neighbourhood(rp.pvs,at);
                if(words.empty()) continue;  // the camera is outside this table's view cells: hide nothing of it
                size=rp.pvs.size;
                for(size_t b=0;b<rp.forms.size();++b)
                    if(rp.usable[b] && !((words[b/64]>>(b%64))&1)) forms.push_back(rp.forms[b]);
            }
            referencePrevisHidden=forms.size();
            const std::array<int64_t,3> cell=size>0.f ? std::array<int64_t,3>{static_cast<int64_t>(std::floor(at.x/size)),
                static_cast<int64_t>(std::floor(at.y/size)),static_cast<int64_t>(std::floor(at.z/size))} : std::array<int64_t,3>{};
            Occlusion::SetReferencePrevisHidden(std::move(forms),cell,size);
        }

        // Runtime-baked chunks are prepared on a worker thread, as the engine's own loaders parse models off the main thread:
        // the fingerprints of the chunk mesh and of each source mesh not checked yet, and the chunk mesh's parse. Activating
        // a chunk then only checks references, clones and attaches (activation tasks took a median 21-26 ms,
        // up to 73 ms, hashing and loading meshes on a game thread). Parts handed over in play request their chunk meshes
        // when registered (their cell is attached); a chunk with anything still unprepared requests it, ahead of everything
        // queued, and waits. Parsed meshes are kept until taken, up to kPreloadBytes of mesh files, the oldest dropped
        // first, and dropped with their cell's detach or retirement and at a save load (DropPreloaded).
        static constexpr size_t kPreloadBytes=256ull<<20;
        struct MeshJob { std::string model; uint32_t bytes=0; uint64_t hash=0; bool parse=false; };
        struct ParsedMesh { RE::NiPointer<RE::NiNode> node; size_t bytes=0; uint64_t sequence=0; };
        std::mutex preloadMutex;
        std::condition_variable preloadReady;
        std::deque<MeshJob> preloadQueue;
        std::unordered_set<std::string> preloadPending;           // CheckKey of fingerprint jobs, "parse:" + mesh of parses
        std::unordered_map<std::string,bool> preparedChecks;      // fingerprints computed by the worker, by CheckKey
        std::unordered_map<std::string,ParsedMesh> preloaded;         // by mesh; a null node: the fingerprint or parse failed
        std::deque<std::pair<std::string,uint64_t>> preloadOrder;
        size_t preloadedBytes=0;
        uint64_t preloadSequence=0;
        bool preloadStarted=false;
        static std::string CheckKey(const std::string& model,uint32_t bytes,uint64_t hash) {
            return normalized(model)+":"+std::to_string(bytes)+":"+std::to_string(hash);
        }
        // Queues a job not queued or done yet; `front`: a chunk being activated waits for it. Caller holds preloadMutex.
        void QueueLocked(MeshJob job,bool front=false) {
            const auto key=CheckKey(job.model,job.bytes,job.hash);
            if(job.parse) {
                if(preloaded.contains(job.model) || !preloadPending.insert("parse:"+job.model).second) return;
            } else if(preparedChecks.contains(key) || !preloadPending.insert(key).second) return;
            if(front) preloadQueue.push_front(std::move(job)); else preloadQueue.push_back(std::move(job));
            if(preloadStarted) return;
            preloadStarted=true;
            std::thread([this] {
                while(true) {
                    MeshJob job;
                    {
                        std::unique_lock lock(preloadMutex);
                        preloadReady.wait(lock,[&] { return !preloadQueue.empty(); });
                        job=std::move(preloadQueue.front()); preloadQueue.pop_front();
                    }
                    const auto key=CheckKey(job.model,job.bytes,job.hash);
                    const bool valid=HashMatches(job.model,job.bytes,job.hash);
                    RE::NiPointer<RE::NiNode> node=valid && job.parse ? LoadMeshDirect(job.model) : nullptr;
                    std::lock_guard lock(preloadMutex);
                    preparedChecks[key]=valid;
                    if(!job.parse) { preloadPending.erase(key); continue; }
                    preloadPending.erase("parse:"+job.model);
                    const size_t bytes=node ? job.bytes : 0;
                    preloaded[job.model]=ParsedMesh{std::move(node),bytes,++preloadSequence};
                    preloadOrder.emplace_back(job.model,preloadSequence);
                    preloadedBytes+=bytes;
                    while(preloadedBytes>kPreloadBytes && !preloadOrder.empty()) {
                        const auto [oldest,sequence]=preloadOrder.front(); preloadOrder.pop_front();
                        const auto it=preloaded.find(oldest);
                        if(it==preloaded.end() || it->second.sequence!=sequence) continue;
                        preloadedBytes-=it->second.bytes; preloaded.erase(it);
                    }
                }
            }).detach();
        }
        void Preload(std::vector<MeshJob> jobs) {
            {
                std::lock_guard lock(preloadMutex);
                for(auto& job:jobs) QueueLocked(std::move(job));
            }
            preloadReady.notify_one();
        }
        // True when the chunk's mesh is parsed (or failed) and every source fingerprint is known; otherwise queues what is
        // missing, at the front: nothing queued for later delays a chunk being activated. Fingerprints are never
        // forgotten (meshChecks, preparedChecks), so the slot remembers how many leading sources are known, and the keys
        // are the ones ParseManifest built: a chunk waiting for its meshes costs no allocation per task.
        bool Prepared(const Definition& def,Slot& slot) {
            bool ready=true,queued=false;
            {
                std::lock_guard lock(preloadMutex);
                for(size_t i=slot.prepared;i<def.sources.size();++i) {
                    const auto& source=def.sources[i];
                    if(meshChecks.contains(source.checkKey) || preparedChecks.contains(source.checkKey)) {
                        if(ready) slot.prepared=static_cast<uint32_t>(i+1);
                        continue;
                    }
                    ready=false;
                    if(!preloadPending.contains(source.checkKey)) { QueueLocked({source.model,source.bytes,source.hash,false},true); queued=true; }
                }
                if(!preloaded.contains(def.mesh)) {
                    ready=false;
                    if(!preloadPending.contains(def.parseKey)) { QueueLocked({def.mesh,def.bytes,def.hash,true},true); queued=true; }
                }
            }
            if(queued) preloadReady.notify_one();
            return ready;
        }
        // The parsed mesh, now owned by the caller; null when none is parsed (a failed one is forgotten, to be retried).
        RE::NiPointer<RE::NiNode> TakePreloaded(const std::string& mesh) {
            std::lock_guard lock(preloadMutex);
            const auto it=preloaded.find(mesh);
            if(it==preloaded.end()) return nullptr;
            auto node=std::move(it->second.node);
            preloadedBytes-=it->second.bytes; preloaded.erase(it);
            return node;
        }
        // Drops the parsed meshes of these definitions (a cell that detached or was retired), or all of them (a save load):
        // only chunks about to attach need one (parsed meshes of cells nobody was in stayed resident up to
        // kPreloadBytes). Jobs still queued finish as before.
        void DropPreloaded(const std::vector<size_t>* indices) {
            std::lock_guard lock(preloadMutex);
            if(!indices) { preloaded.clear(); preloadOrder.clear(); preloadedBytes=0; return; }
            for(const auto index:*indices) {
                const auto it=preloaded.find(definitions[index].mesh);
                if(it==preloaded.end()) continue;
                preloadedBytes-=it->second.bytes; preloaded.erase(it);
            }
            // Order entries of meshes no longer held (dropped, or taken by TakePreloaded) are skipped by the eviction; this
            // keeps the list near the size of what is held.
            if(preloadOrder.size()>2*preloaded.size()+1024)
                std::erase_if(preloadOrder,[&](const auto& entry) { const auto it=preloaded.find(entry.first); return it==preloaded.end() || it->second.sequence!=entry.second; });
        }

        // A runtime-baked mesh's file when the cache folder is outside Data ("sCacheFolder"), where BSResource streams do
        // not look; nothing for every other mesh, which the game's resource system opens.
        static std::optional<fs::path> CachedMeshFile(const std::string& mesh) {
            if(Config::CacheFolder().empty() || !mesh.starts_with("precombined/skyrim/rb/")) return std::nullopt;
            return Config::MeshRoot()/mesh;
        }
        // A file's bytes as a stream for NiStream::Load1, which reads every stream (BSResource, NiFile, NiMemStream) through
        // its read function and its virtual seek and tell. The baker writes little-endian files only: a request to swap
        // bytes fails the load.
        class MemoryStream final : public RE::NiBinaryStream {
        public:
            explicit MemoryStream(std::string bytes) : data(std::move(bytes)) { _readFn=&Read; }
            bool good() const override { return true; }
            void seek(std::int32_t offset) override {
                position=static_cast<std::uint32_t>(std::clamp<std::int64_t>(std::int64_t(position)+offset,0,std::int64_t(data.size())));
                _absoluteCurrentPos=position;
            }
            std::uint32_t tell() const override { return position; }
            void set_endian_swap(bool doSwap) override { swap=doSwap; }
        private:
            static std::uint32_t Read(RE::NiBinaryStream* stream,void* buffer,std::uint32_t bytes,std::uint32_t*,std::uint32_t) {
                auto& self=*static_cast<MemoryStream*>(stream);
                if(self.swap) return 0;
                const auto count=std::min<std::uint32_t>(bytes,static_cast<std::uint32_t>(self.data.size()-self.position));
                std::memcpy(buffer,self.data.data()+self.position,count);
                self.position+=count;
                return count;
            }
            std::string data;
            std::uint32_t position=0;
            bool swap=false;
        };

        // A NIF loaded straight from a BSResource stream, as LoadBTRMesh (SE 1.5.97, RVA 4ACDC0) and the model loader at RVA
        // D2E880 do: NiStream's constructor (SE ID 68971, AE 70324, VR RVA C9EC40; the same 597-byte body and member
        // offsets in all three) on 0x630 bytes, BSStream's vtable, Load1 on the stream, the first top-level object, then
        // BSStream's own destructor (it frees its node references and then runs NiStream's). A runtime-baked mesh in a
        // cache folder outside Data is read from its file into a MemoryStream instead.
        static RE::NiPointer<RE::NiNode> LoadMeshDirect(const std::string& mesh) {
            std::optional<RE::BSResourceNiBinaryStream> resource;
            std::optional<MemoryStream> memory;
            RE::NiBinaryStream* file=nullptr;
            if(const auto cached=CachedMeshFile(mesh)) {
                std::error_code ec;
                const auto size=fs::file_size(*cached,ec);
                if(ec || size==0 || size>256ull*1024*1024) return nullptr;
                try { file=&memory.emplace(readFile(*cached)); } catch(const std::exception&) { return nullptr; }
            } else {
                file=&resource.emplace("meshes/"+mesh);
                if(!file->good()) return nullptr;
            }
            alignas(16) std::array<std::byte,0x630> storage{};
            using Construct=RE::NiStream*(RE::NiStream*);
            static REL::Relocation<Construct> construct{REL::VariantID(68971,70324,0xC9EC40)};
            static REL::Relocation<std::uintptr_t> bsStreamVtable{RE::VTABLE_BSStream[0]};
            auto* stream=construct(reinterpret_cast<RE::NiStream*>(storage.data()));
            *reinterpret_cast<std::uintptr_t*>(stream)=bsStreamVtable.address();
            strcpy_s(stream->inputFilePath,("meshes\\"+mesh).c_str());
            RE::NiPointer<RE::NiNode> root;
            if(stream->Load1(file) && stream->topObjects.size()>0 && stream->topObjects[0])
                root=RE::NiPointer<RE::NiNode>(stream->topObjects[0]->AsNode());
            stream->~NiStream();
            return root;
        }
        bool CheckMesh(const std::string& model, uint32_t bytes, uint64_t expected) {
            const auto key=CheckKey(model,bytes,expected);
            if (auto it=meshChecks.find(key);it!=meshChecks.end()) return it->second;
            std::optional<bool> prepared;
            {
                std::lock_guard lock(preloadMutex);
                if(const auto it=preparedChecks.find(key);it!=preparedChecks.end()) prepared=it->second;
            }
            const bool valid=prepared ? *prepared : HashMatches(model,bytes,expected);
            meshChecks.emplace(key,valid);
            if(!valid) logger::warn("Source fingerprint differs or mesh missing: {}",model);
            return valid;
        }
        // The file is exactly bytes long and its FNV-1a is expected. Any thread.
        static bool HashMatches(const std::string& model, uint32_t bytes, uint64_t expected) {
            if(const auto cached=CachedMeshFile(model)) {
                std::error_code ec;
                if(fs::file_size(*cached,ec)!=bytes || ec) return false;
                try { const auto data=readFile(*cached); return data.size()==bytes && fnv1a(data)==expected; } catch(const std::exception&) { return false; }
            }
            RE::BSResourceNiBinaryStream stream("meshes/"+model);
            bool valid=stream.good(); uint64_t hash=14695981039346656037ull;
            std::array<unsigned char,65536> buffer{};
            for(uint32_t remaining=bytes;valid && remaining;) {
                const auto length=std::min<uint32_t>(remaining,static_cast<uint32_t>(buffer.size()));
                valid=stream.read(buffer.data(),length);
                if (valid) for(uint32_t i=0;i<length;++i) hash=(hash^buffer[i])*1099511628211ull;
                remaining-=length;
            }
            unsigned char extra{};
            return valid && hash==expected && !stream.read(&extra,1);
        }

        RE::BSTriShape* Resolve(RE::NiAVObject* root, const Source& source) {
            auto* current=root;
            for(const auto& name:source.path) {
                auto* node=current ? current->AsNode():nullptr;
                if(!node) return nullptr;
                RE::NiAVObject* found=nullptr;
                for(const auto& child:node->GetChildren()) if(child && child->name.c_str()==name) {
                    if(found) return nullptr;
                    found=child.get();
                }
                current=found;
            }
            return current ? current->AsTriShape():nullptr;
        }

        // The live enable parent must be the declared one with the declared flag (bit 0 of ExtraEnableStateParent::flags
        // is XESP's "set enable state to opposite of parent"). A plugin-relative source without a declared parent must
        // have none; old Skyrim.esm sources were checked offline (the offline generator excluded every XESP).
        static bool CheckEnableParent(RE::TESObjectREFR* ref,const Source& source) {
            auto* extra=ref->extraList.GetByType<RE::ExtraEnableStateParent>();
            if(!source.enableParent) return !extra || !source.relative;
            if(!extra) return false;
            const auto parent=extra->parent.get();
            return parent && parent->GetFormID()==source.enableParent && ((extra->flags&1)!=0)==source.enableOpposite;
        }
        // An old source must still be the Skyrim.esm record (any override rejects). A plugin-relative source was baked
        // from the winning record, so any file may win as long as every checked field matches and the reference has
        // none of the extra data that let other records drive it (links, activation parents, emittance). Multibound (room)
        // references are not checked: the engine assigns them to nearly every interior reference at load.
        static bool CheckOrigin(RE::TESObjectREFR* ref,const RE::TESBoundObject* base,const Source& source) {
            if(!source.relative) return vanilla(ref) && vanilla(base);
            for(const auto type:{RE::ExtraDataType::kLinkedRef,RE::ExtraDataType::kLinkedRefChildren,RE::ExtraDataType::kActivateRef,
                                 RE::ExtraDataType::kActivateRefChildren,RE::ExtraDataType::kEmittanceSource,
                                 RE::ExtraDataType::kTeleport,RE::ExtraDataType::kLock,RE::ExtraDataType::kPrimitive})
                if(ref->extraList.HasType(type)) return false;
            return true;
        }
        // The base's live alternate-texture list equals the declared one (any order).
        static bool SameAlternates(const RE::TESObjectSTAT& stat,const Source& source) {
            if(stat.numAlternateTextures!=source.alternates.size()) return false;
            for(uint32_t i=0;i<stat.numAlternateTextures;++i) {
                const auto& live=stat.alternateTextures[i];
                const char* name=live.name3D.c_str();
                const bool found=std::any_of(source.alternates.begin(),source.alternates.end(),[&](const Source::Alternate& declared) {
                    return declared.index3D==live.index3D && live.textureSet && declared.textureSet==live.textureSet->GetFormID() &&
                           _stricmp(declared.name3D.c_str(),name ? name : "")==0;
                });
                if(!found) return false;
            }
            return true;
        }
        bool CheckRecord(RE::TESObjectREFR* ref,const Source& source,uint32_t cellID) {
            if(!ref || ref->IsDisabled() || ref->IsDeleted() || ref->IsPersistent()) return false;
            auto* base=ref->GetBaseObject(); auto* cell=ref->GetParentCell();
            if(!cell || cell->GetFormID()!=cellID || !base || base->GetFormID()!=source.baseID) return false;
            if(!CheckOrigin(ref,base,source) || !CheckEnableParent(ref,source)) return false;
            auto* stat=base->As<RE::TESObjectSTAT>();
            if(!stat || stat->data.materialObj || normalized(stat->GetModel())!=source.modelKey || !SameAlternates(*stat,source)) return false;
            return closeEnough(ref->GetPosition(),source.position,0.01f) && closeEnough(ref->GetAngle(),source.rotation,0.00001f) && closeEnough(ref->GetScale(),source.scale,0.0005f);
        }

        // Visibility, controllers, transform, counts and fade from the shape up to the reference root.
        bool CulledByUs(const RE::NiAVObject* node) const { return culledRoots.contains(const_cast<RE::NiAVObject*>(node)); }

        bool ShapeIntact(const Binding& binding,const Source& source,bool hidden,bool fadeMatters) const {
            auto* shape=binding.shape.get();
            if(!shape || shape->controllers || shape->GetAppCulled()!=hidden || !closeEnough(shape->world,source.world)) return false;
            const auto& data=shape->GetTrishapeRuntimeData();
            if(data.vertexCount!=source.vertices || data.triangleCount!=source.triangles) return false;
            // An interior chunk has no per-reference fade. Drop it as soon as any covered
            // reference starts fading, hides, or changes its hierarchy. Exterior references
            // fade with distance all the time, so only hiding and hierarchy changes count there.
            auto* node=static_cast<RE::NiAVObject*>(shape->parent);
            for(unsigned depth=0;node && depth<64;++depth) {
                if((node->GetAppCulled() && !CulledByUs(node)) || node->controllers) return false;
                if(auto* fade=node->AsFadeNode();fadeMatters && fade && !closeEnough(fade->GetRuntimeData().currentFade,1.f,0.001f)) return false;
                if(node==binding.root.get()) return true;
                node=node->parent;
            }
            return false;
        }

        // Full acceptance check, used when attaching.
        bool CheckShape(const Binding& binding,const Source& source,bool hidden,const Definition& def) {
            if(!CheckRecord(binding.reference.get(),source,def.cell) || binding.reference->Get3D()!=binding.root.get() || Resolve(binding.root.get(),source)!=binding.shape.get()) return false;
            return ShapeIntact(binding,source,hidden,!def.exterior);
        }

        void PrepareChecks(Binding& binding,const Source& source,const RE::NiNode* parent,bool fadeMatters) {
            auto& cachedRef=referenceChecks[source.refID];
            auto ref=cachedRef.lock();
            // Different expectations must never borrow another source's result,
            // even if a hand-written manifest gives the same reference two poses.
            const auto samePoint=[](const RE::NiPoint3& a,const RE::NiPoint3& b) {
                return a.x==b.x && a.y==b.y && a.z==b.z;
            };
            if(!ref || ref->reference!=binding.reference.get() || ref->root!=binding.root.get() ||
               ref->base!=binding.base || ref->cell!=binding.cell || ref->parent!=parent ||
               !samePoint(ref->expected->position,source.position) ||
               !samePoint(ref->expected->rotation,source.rotation) || ref->expected->scale!=source.scale ||
               ref->expected->enableParent!=source.enableParent || ref->expected->enableOpposite!=source.enableOpposite) {
                ref=std::make_shared<ReferenceCheck>();
                ref->reference=binding.reference.get(); ref->root=binding.root.get();
                ref->base=binding.base; ref->cell=binding.cell; ref->parent=parent; ref->expected=&source;
                cachedRef=ref;
                referenceSchedule.Add(source.refID,ref);
                if(source.enableParent) {
                    // CheckRecord has just matched the live parent to the declared one.
                    if(auto* extra=binding.reference->extraList.GetByType<RE::ExtraEnableStateParent>()) ref->enableParent=extra->parent.get();
                    need(ref->enableParent && ref->enableParent->GetFormID()==source.enableParent,"enable parent changed during validation setup");
                    parentedChecks.emplace_back(ref);
                }
            }
            binding.referenceCheck=std::move(ref);
            for(auto* node=static_cast<RE::NiAVObject*>(binding.shape->parent);node;node=node->parent) {
                auto& cachedNode=ancestorChecks[node];
                auto check=cachedNode.lock();
                // The fade policy is part of a cached check's identity.
                auto* fade=fadeMatters ? node->AsFadeNode() : nullptr;
                if(!check || check->parent!=node->parent || check->fade!=fade) {
                    check=std::make_shared<AncestorCheck>(); check->node.reset(node);
                    check->parent=node->parent; check->fade=fade; cachedNode=check;
                }
                binding.ancestors.push_back(std::move(check));
                if(node==binding.root.get()) return;
                // ShapeIntact already accepts at most 64 ancestors. Fail closed
                // if the hierarchy changes between acceptance and this snapshot.
                need(binding.ancestors.size()<64,"source hierarchy changed during validation setup");
            }
            throw std::runtime_error("source root detached during validation setup");
        }

        static bool CheckReference(ReferenceCheck& ref,uint64_t epoch) {
            return ref.result.Get(epoch,[&] {
                const auto* object=ref.reference;
                const auto& expected=*ref.expected;
                return object && !object->IsDisabled() && !object->IsDeleted() &&
                    object->GetBaseObject()==ref.base && object->GetParentCell()==ref.cell &&
                    closeEnough(object->GetPosition(),expected.position,0.01f) &&
                    closeEnough(object->GetAngle(),expected.rotation,0.00001f) &&
                    closeEnough(object->GetScale(),expected.scale,0.0005f) &&
                    object->Get3D()==ref.root && ref.root && ref.root->parent==ref.parent &&
                    // The child is enabled while its parent is enabled, or disabled with the opposite flag. The child's
                    // own disabled flag and 3D are checked above as well.
                    (!ref.enableParent || ref.enableParent->IsDisabled()==expected.enableOpposite);
            });
        }

        // Shader flags that runtime material edits set and NIF statics we merge do not carry: Seasons of Skyrim's
        // single-pass winter snow (SnowSwap.cpp ApplySinglePassSnow -> NiAVObject::SetProjectedUVData) sets kProjectedUV
        // and kSnow on every lighting shader under the reference and tags its root "SOS_SNOW_SHADER".
        static constexpr uint64_t runtimeMaterialFlags=(1ull<<23)|(1ull<<41)|(1ull<<60);  // kProjectedUV, kMultiIndexSnow, kSnow
        static uint64_t ShaderFlags(RE::BSGeometry* geometry) {
            auto* property=geometry ? geometry->GetGeometryRuntimeData().shaderProperty.get() : nullptr;
            return property ? property->flags.underlying() : ~0ull;
        }
        // Attach-time check that a source's live material is still the NIF material the chunk was built from: the same
        // runtime-material flags, the same material feature and the same texture paths (texture swaps), and no
        // Seasons of Skyrim snow tag on the root. Any difference keeps the originals.
        // Texture paths compare without case, slash style or a leading "data/" or "textures/": NIF slots and texture
        // sets (TXST, alternate textures) write the same file both with and without the prefix.
        static std::string TextureKey(const char* path) {
            auto key=normalized(path ? path : "");
            for(const std::string_view prefix:{"data/","textures/"}) if(key.starts_with(prefix)) key.erase(0,prefix.size());
            return key;
        }
        static bool SameMaterial(RE::BSGeometry* chunk,RE::BSTriShape* source,RE::NiAVObject* root) {
            static const RE::BSFixedString snowTag("SOS_SNOW_SHADER");
            if(root && root->GetExtraData(snowTag)) return false;
            if((ShaderFlags(chunk)^ShaderFlags(source))&runtimeMaterialFlags) return false;
            auto* a=chunk->lightingShaderProp_cast();
            auto* b=source->lightingShaderProp_cast();
            if(!a || !b) return false;
            auto* ma=static_cast<RE::BSLightingShaderMaterialBase*>(a->material);
            auto* mb=static_cast<RE::BSLightingShaderMaterialBase*>(b->material);
            if(!ma || !mb || ma->GetFeature()!=mb->GetFeature() || !ma->textureSet!=!mb->textureSet) return false;
            if(!ma->textureSet) return true;
            for(uint32_t t=0;t<RE::BSTextureSet::Texture::kTotal;++t) {
                const auto slot=static_cast<RE::BSTextureSet::Texture>(t);
                const char* x=ma->textureSet->GetTexturePath(slot); const char* y=mb->textureSet->GetTexturePath(slot);
                if(TextureKey(x)!=TextureKey(y)) return false;
            }
            return true;
        }

        // Shape and ancestor checks keep the definition's five-task schedule.
        // Reference fields have their own five-task schedule; current-task
        // failures still fan out to every dependent chunk before restoration.
        bool StillValid(Binding& binding,const Source& source,uint64_t epoch,uint64_t materialFlags) const {
            if(binding.referenceCheck->result.Failed(epoch)) return false;
            auto* shape=binding.shape.get();
            if(!shape || shape->parent!=binding.ancestors.front()->node.get() || shape->controllers ||
               !shape->GetAppCulled() || !binding.worldCheck.Get(shape->world,[&](const auto& world) { return closeEnough(world,source.world); })) return false;
            const auto& data=shape->GetTrishapeRuntimeData();
            if(data.vertexCount!=source.vertices || data.triangleCount!=source.triangles) return false;
            if(materialCheck && ((ShaderFlags(shape)^materialFlags)&runtimeMaterialFlags)) return false;
            for(const auto& ancestor:binding.ancestors) {
                if(!ancestor->result.Get(epoch,[&] {
                    const auto* node=ancestor->node.get();
                    return node->parent==ancestor->parent && (!node->GetAppCulled() || CulledByUs(node)) && !node->controllers &&
                        (!ancestor->fade || closeEnough(ancestor->fade->GetRuntimeData().currentFade,1.f,0.001f));
                })) return false;
            }
            return true;
        }

        static bool DependencyFailed(const Binding& binding,uint64_t epoch) {
            if(binding.referenceCheck->result.Failed(epoch)) return true;
            for(const auto& ancestor:binding.ancestors) if(ancestor->result.Failed(epoch)) return true;
            return false;
        }

        void Restore(Active& chunk) {
            // Re-show only the exact shape objects this plugin hid. Never touch
            // new 3D created after reload, or a reference's root/disabled state.
            if(chunk.mesh) chunk.mesh->SetAppCulled(true);
            if(chunk.lightingScene && chunk.mesh) {
                // SE 1.5.97 ID 99705 (12BAAA0; AE 106339, VR RVA 12F84D0): queue this
                // object for light-list removal, without touching the portal graph.
                // The queue holds its own reference and releases it when the engine
                // drains it.
                using RemoveObject=void(RE::ShadowSceneNode*,RE::NiAVObject*,bool,bool);
                static REL::Relocation<RemoveObject> remove{REL::VariantID(99705,106339,0x12F84D0)};
                remove(chunk.lightingScene.get(),chunk.mesh.get(),false,false);
                chunk.lightingScene.reset();
            }
            // Un-cull a root this plugin culled before showing its shapes again; the root's other
            // chunks stay active and keep their shapes hidden.
            for(auto& source:chunk.sources) UncullRoot(source.root.get());
            for(auto& source:chunk.sources) if(source.shape) source.shape->SetAppCulled(false);
            if(chunk.mesh && chunk.mesh->parent) chunk.mesh->parent->DetachChild(chunk.mesh.get());
        }
        void UncullRoot(RE::NiAVObject* root) {
            if(auto it=culledRoots.find(root);it!=culledRoots.end()) { root->SetAppCulled(false); culledRoots.erase(it); }
        }
        // After a chunk activates: cull each source root whose every leaf is now hidden.
        void CullCoveredRoots(const Active& chunk) {
            if(!cullRoots) return;
            for(const auto& source:chunk.sources) {
                auto* root=source.root.get();
                if(!root || root->GetAppCulled() || !root->AsFadeNode() || !FullyHidden(root)) continue;
                root->SetAppCulled(true);
                culledRoots.emplace(root,source.root);
            }
        }
        void PublishCounts() {
            uint32_t hidden=0; for(const auto& chunk:active) hidden+=static_cast<uint32_t>(chunk.sources.size());
            statActive.store(static_cast<uint32_t>(active.size()),relaxed); statHidden.store(hidden,relaxed);
            statCulled.store(static_cast<uint32_t>(culledRoots.size()),relaxed);
            statPrevis.store(static_cast<uint32_t>(previsHidden),relaxed);
        }
        // Keeps detachingCells: those follow the engine's events, not our enabled state.
        void Clear() {
            for(auto& chunk:active) Restore(chunk);
            for(auto& [root,held]:culledRoots) root->SetAppCulled(false);  // none expected after Restore
            culledRoots.clear(); previsHidden=0; referencePrevisHidden=0;
            if(Occlusion::Installed()) { Occlusion::SetPrevisHidden({}); Occlusion::SetReferencePrevisHidden({},{},0.f); }
            active.clear(); std::fill(slots.begin(),slots.end(),Slot{});
            referenceChecks.clear(); ancestorChecks.clear(); parentedChecks.clear();
            referenceSchedule.Clear();
            eligibleCells.clear(); cursor=0; dissolved=0; rejections.clear(); lastReport={};
            stageTicks=referenceNs=sceneNs=activationNs=0;
            PublishCounts();
        }
        void Defer(size_t index,Clock::time_point now) {
            auto& slot=slots[index];
            slot.active=false; slot.failures=static_cast<uint8_t>(std::min(slot.failures+1,8));
            slot.retryAt=now+std::min<Clock::duration>(std::chrono::seconds(60),std::chrono::milliseconds(250ll<<slot.failures));
        }

        // False also while a runtime-baked mesh is still being parsed (meshPending): the chunk is retried shortly, not deferred.
        bool meshPending=false;
        bool Attach(size_t index) {
            const auto& def=definitions[index];
            const auto reject=[this](const char* reason) { ++rejections[reason]; return false; };
            if(def.mesh.starts_with("precombined/skyrim/rb/") && !Prepared(def,slots[index])) { meshPending=true; return false; }
            Active chunk; chunk.definition=index;
            chunk.sources.reserve(def.sources.size());
            for(const auto& source:def.sources) {
                auto* ref=RE::TESForm::LookupByID<RE::TESObjectREFR>(source.refID);
                Binding binding; binding.reference.reset(ref);
                if(!CheckRecord(ref,source,def.cell)) return reject("record changed or unavailable");
                binding.base=ref->GetBaseObject(); binding.cell=ref->GetParentCell();
                binding.root.reset(ref->Get3D());
                if(!binding.root || !binding.root->parent) return reject("3D not attached");
                binding.shape.reset(Resolve(binding.root.get(),source));
                if(!CheckShape(binding,source,false,def)) return reject("shape, transform or visibility differs");
                if(!CheckMesh(source.model,source.bytes,source.hash)) return reject("source mesh differs");
                auto* parent=binding.root->parent;
                if(chunk.parent && chunk.parent.get()!=parent) return reject("different room parents");
                chunk.parent.reset(parent);
                chunk.sources.push_back(std::move(binding));
            }
            if(!chunk.parent || !CheckMesh(def.mesh,def.bytes,def.hash)) return reject("chunk mesh differs");
            RE::NiPointer<RE::NiNode> loaded;
            const bool runtimeMesh=def.mesh.starts_with("precombined/skyrim/rb/");
            loaded=runtimeMesh ? TakePreloaded(def.mesh) : nullptr;
            // The model database cannot find a mesh in a cache folder outside Data; LoadMeshDirect reads its file.
            auto code=loaded || CachedMeshFile(def.mesh) ? RE::BSResource::ErrorCode::kNone : RE::BSModelDB::Demand(def.mesh.c_str(),loaded,{});
            if((code!=RE::BSResource::ErrorCode::kNone || !loaded) && runtimeMesh) {
                // BSModelDB answered "does not exist" (error 1) for meshes the bake wrote during play under
                // MO2's virtual file system, although BSResource streams open them (CheckMesh). Read them as the engine's
                // own loaders do, through a BSStream on that stream.
                loaded=LoadMeshDirect(def.mesh);
                if(!loaded) {
                    static std::once_flag once;
                    std::call_once(once,[&] { logger::warn("Runtime-baked mesh {} did not load (model database error {}, direct stream load failed)",def.mesh,static_cast<int>(code)); });
                }
            }
            if(!loaded) return reject("model load failed");
            RE::NiPointer<RE::NiObject> clone(static_cast<RE::NiObject*>(loaded.get())->Clone());
            if(!clone || !clone->AsNode()) return reject("model clone failed");
            chunk.mesh.reset(clone->AsNode());
            chunk.mesh->SetAppCulled(true);
            const auto& children=chunk.mesh->GetChildren();
            auto* geometry=!children.empty() && children[0] ? children[0]->AsGeometry() : nullptr;
            if(!geometry) return reject("chunk mesh differs");
            chunk.materialFlags=ShaderFlags(geometry);
            if(materialCheck) for(const auto& source:chunk.sources)
                if(!SameMaterial(geometry,source.shape.get(),source.root.get())) return reject("source material changed at runtime (snow or texture swap)");
            chunk.parentWorld=chunk.parent->world;
            RE::NiTransform target; target.translate=def.origin;
            chunk.mesh->local=chunk.parentWorld.Invert()*target;
            // Recheck all sources after the potentially expensive load, before
            // making any visibility change. All mutations occur on an SKSE task.
            for(size_t i=0;i<chunk.sources.size();++i) if(!CheckShape(chunk.sources[i],def.sources[i],false,def)) return reject("source changed during load");
            for(size_t i=0;i<chunk.sources.size();++i) PrepareChecks(chunk.sources[i],def.sources[i],chunk.parent.get(),!def.exterior);
            // Track the transaction before changing the scene. If allocation or
            // a later operation throws, Tick's rollback can restore every shape.
            active.push_back(std::move(chunk));
            auto& live=active.back();
            live.parent->AttachChild(live.mesh.get(),true);
            RE::NiUpdateData update{};
            if(live.mesh->parent==live.parent.get()) live.mesh->Update(update);
            if(live.mesh->parent!=live.parent.get() || !closeEnough(live.mesh->world,target)) {
                Restore(live); active.pop_back(); return reject("chunk placement failed");
            }
            SeedPreviousWorld(live.mesh.get());
            for(auto& source:live.sources) source.shape->SetAppCulled(true);
            live.lightingScene.reset(RE::BSShaderManager::State::GetSingleton().shadowSceneNode[0]);
            if(!live.lightingScene) { Restore(live);active.pop_back();return reject("lighting scene unavailable"); }
            // NiNode::AttachChild alone does not add a new model to Skyrim's local-light lists.
            size_t lights=0;
            const bool assigned=AssignSourceLights(live,lights);
            live.mesh->SetAppCulled(false);
            if(!assigned) live.lightingScene->AttachObject(live.mesh.get());
            slots[index].active=true; slots[index].failures=0;
            CullCoveredRoots(live);
            statAttaches.fetch_add(1,relaxed);
            logger::info("Attached {}: {} shapes -> 1, {} source lights",def.mesh,live.sources.size(),lights);
            return true;
        }

        // ShadowSceneNode::AttachObject only queues an object. When the engine drains that queue
        // (SE 1.5.97 0x12B9DB0) it rebuilds each geometry's light list from its room's list
        // (0x12BB6B0, 0x12BB090), dropping every other light, and it skips AppCulled nodes.
        // The original shapes also hold lights that reached them through portals from each
        // light's own scene walk at cell load, so a queued chunk misses lights its sources hold
        // and gains room lights that reach none of them.
        // Give the chunk exactly the union of its sources' lists instead, through the engine's
        // own routine (BSTriShape 0x12BBA50, ID 99717; AE 106354, VR RVA 12F94D0: rebuilds the list from a null-terminated array and
        // registers the geometry with each light, so ID 99705 still removes them), without
        // queueing it. Called while the chunk is AppCulled, which light walks also skip, under
        // the lock the engine holds while it drains the queue. Later light walks (moving or new
        // lights) still add themselves as usual. Returns false when the engine queue must assign
        // lights instead (sync disabled, or no lit geometry to assign to).
        bool AssignSourceLights(Active& live,size_t& assigned) {
            if(!syncLights || !live.mesh || live.mesh->GetChildren().empty()) return false;
            auto* child=live.mesh->GetChildren()[0].get();
            auto* geometry=child ? child->AsGeometry():nullptr;
            auto* property=geometry ? geometry->lightingShaderProp_cast():nullptr;
            if(!property || !property->lightData) return false;
            std::vector<RE::BSLight*> lights;
            for(auto& source:live.sources) {
                auto* sp=source.shape ? source.shape->lightingShaderProp_cast():nullptr;
                if(!sp || !sp->lightData) continue;
                for(auto* light:sp->lightData->lights)
                    if(light && std::find(lights.begin(),lights.end(),light)==lights.end()) lights.push_back(light);
            }
            assigned=lights.size();
            lights.push_back(nullptr);
            using AssignLights=std::uint64_t(RE::BSGeometry*,void*,RE::BSLight**);
            static REL::Relocation<AssignLights> assign{REL::VariantID(99717,106354,0x12F94D0)};
            {
                RE::BSSpinLockGuard guard(live.lightingScene->GetRuntimeData().objectListLock);
                assign(geometry,nullptr,lights.data());
            }
            statLightsSynced.fetch_add(assigned,relaxed);
            return true;
        }

        // ---- Runtime-bake hand-off (config "runtime_bake", default true) ----
        // Each part's chunks become ordinary definitions: the same per-task activation limit, acceptance checks,
        // validation and restoration apply. A part is refused for a cell with a manifest in the manifest folder
        // (shipped, or merged by the bake plugin), so the two are never mixed. A newer bake of a cell retires the older
        // bake: its chunks are restored and its definitions are never activated again (they stay in place, so indices
        // and validation pointers remain valid).
        // Data load reads only index.json: each cached cell's parts and previs tables are read when the cell is first
        // attached (RequestCachedCell). An entry without a readable "cell" (none written so far) is loaded here as before.
        void ReadRuntimeParts() {
            if(!runtimeBake) return;
            size_t parts=0, unloaded=0, older=0, unreadable=0;
            const auto before=definitions.size();
            try {
                // Exterior previs tables ("<cell key>-<bake>-from-<camera cell key>.pvs.json") by bake: read after their
                // cell's parts. A name as UTF-8: path::string() throws for a stray file named outside the ANSI code page.
                std::unordered_map<std::string,std::vector<std::string>> exteriorTables;
                std::error_code ec;
                if(fs::is_directory(runtimePrevisDir(),ec))
                    for(const auto& file:fs::directory_iterator(runtimePrevisDir(),ec)) {
                        const auto text=file.path().filename().u8string();
                        const std::string name(text.begin(),text.end());
                        const auto from=name.find("-from-");
                        if(from==std::string::npos || from<17 || !runtimePrevisName(name)) continue;
                        exteriorTables[name.substr(from-16,16)].push_back(name);
                    }
                std::unordered_map<uint32_t,CachedCell> cells;
                std::unordered_map<std::string,uint32_t> keys;
                const auto path=Config::BakeDir()/"index.json";
                if(fs::exists(path)) {
                    const auto index=json::parse(readFile(path));
                    Plugins plugins;
                    // Each entry on its own: a malformed one is counted and skipped, the others still load.
                    for(const auto& [key,entry]:index.items()) try {
                        if(!entry.is_object() || entry.value("mode","")!="parts") continue;
                        // A bake of another baker version (older entries have none) is never shown: its key
                        // differs from what the bake computes now, so the cell is baked again on its next visit anyway.
                        if(entry.value("baker",0)!=precombine::kOutputVersion) { ++older; continue; }
                        CachedCell cached;
                        for(const auto& name:entry.value("parts",json::array())) cached.parts.push_back(name.get<std::string>());
                        if(entry.contains("previs")) cached.tables.push_back(entry.at("previs").get<std::string>());
                        if(const auto found=exteriorTables.find(entry.value("bake",""));found!=exteriorTables.end())
                            cached.tables.insert(cached.tables.end(),found->second.begin(),found->second.end());
                        const auto cell=CachedCellForm(entry,plugins);
                        if(!cell) {
                            for(const auto& name:cached.parts) parts+=LoadRuntimePart(name,false);
                            for(const auto& name:cached.tables) LoadRuntimePrevis(name);
                            continue;
                        }
                        // Cells whose plugin is not loaded, or that keep their originals anyway, are never read.
                        if(*cell==0) { ++unloaded; continue; }
                        if(manifestHashes.contains(*cell) || excluded.contains(*cell)) continue;
                        keys.insert_or_assign(key,*cell);
                        cells.insert_or_assign(*cell,std::move(cached));  // a later entry for one cell wins, as its parts retired the earlier
                    } catch(const std::exception&) { ++unreadable; }
                }
                if(!cells.empty()) {
                    std::thread([this] { ReadCachedCells(); }).detach();
                    cachedCells=std::move(cells); cachedKeys=std::move(keys);
                }
            } catch(const std::exception& e) { logger::error("Runtime bake index unreadable: {}",e.what()); }
            if(unreadable) logger::error("Runtime bake index: {} entries unreadable and skipped; their cells keep their originals",unreadable);
            logger::info("Runtime bake hand-off on: {} cached cells, read when first attached ({} skipped: plugin not loaded; {} of another baker version, "
                         "baked again when visited); {} parts with {} chunk definitions loaded at data load",
                cachedCells.size(),unloaded,older,parts,definitions.size()-before);
        }
        // Precombines switched on in play in a session that started with them off (bEnabled=0 outside developer mode kept
        // the hand-off and the bake from starting, ReadSettings): the cells merged before come back, their index read now as
        // at data load (each cell's parts still when it is first attached); cells not merged yet wait for a restart, since
        // the bake itself did not start (BakePlugin.cpp). Once; under stateMutex.
        void ResumeRuntimeHandOff() {
            if(runtimeBake || !Config::Flag("runtime_bake",true)) return;
            runtimeBake=true;
            logger::info("Precombines switched on in play: the runtime bake's cached cells are read now; new cells are merged after a restart");
            ReadRuntimeParts();
        }
        // An index entry's cell ("cell": "<plugin>|<local form ID, hex>", RuntimeBake's KeyText): its form ID, 0 when its
        // plugin is not loaded, nothing when the entry has no such field.
        static std::optional<uint32_t> CachedCellForm(const json& entry,Plugins& plugins) {
            const auto field=entry.find("cell");
            if(field==entry.end() || !field->is_string()) return std::nullopt;
            const auto& text=field->get_ref<const std::string&>();
            const auto bar=text.rfind('|');
            if(bar==std::string::npos || bar==0) return std::nullopt;
            uint32_t local=0;
            const auto [end,error]=std::from_chars(text.data()+bar+1,text.data()+text.size(),local,16);
            if(error!=std::errc() || end!=text.data()+text.size()) return std::nullopt;
            try { return resolveForm(json{{"plugin",text.substr(0,bar)},{"id",local}},&plugins); } catch(const std::exception&) { return std::nullopt; }
        }
        // Hands a cached cell to the cache reader, once. False when it is not cached or no longer needed: a bake of this
        // session is newer than the cache read at data load.
        bool RequestCachedCell(uint32_t cell) {
            auto node=cachedCells.extract(cell);
            if(!node || runtimeCells.contains(cell)) return false;
            {
                std::lock_guard lock(runtimePartsMutex);
                runtimeRequests.push_back(std::move(node.mapped()));
                ++runtimeReading;
            }
            runtimeReader.notify_all();
            return true;
        }
        // Every attached cell still cached, whatever `enabled` is (developer mode turns precombines on in play and finds its
        // cells registered, as when every part was read at data load). Not during loading screens, as UpdateEligibleCells.
        void RequestAttachedCells() {
            if(cachedCells.empty()) return;
            auto* tes=RE::TES::GetSingleton();
            if(!tes) return;
            const auto request=[this](RE::TESObjectCELL* cell) { if(cell && cell->IsAttached()) RequestCachedCell(cell->GetFormID()); };
            request(tes->interiorCell);
            if(auto* grid=tes->gridCells) for(uint32_t x=0;x<grid->length;++x) for(uint32_t y=0;y<grid->length;++y) request(grid->GetCell(x,y));
        }
        // The cache reader (its own thread, started at data load when anything is cached): reads and parses a requested
        // cell's parts as SkyrimPrecombines_RuntimeBakePart does, then queues its previs tables behind them. Never throws.
        void ReadCachedCells() {
            for(;;) {
                CachedCell request;
                {
                    std::unique_lock lock(runtimePartsMutex);
                    runtimeReader.wait(lock,[&] { return !runtimeRequests.empty(); });
                    request=std::move(runtimeRequests.front()); runtimeRequests.pop_front();
                }
                try {
                    // The bake deletes a changed cell's old parts, meshes and tables before baking it again (RuntimeBake.cpp
                    // RemoveOutput), and with the cache read at attach that can come first: a part that is gone means the
                    // cell is being baked again, so none of its cached files is used (each was logged as an error, and the
                    // old bake could end up half registered until the new one retired it). Checked before the first read
                    // and again when a read fails.
                    const auto missing=[](const fs::path& file) { std::error_code ec; return !fs::exists(file,ec) && !ec; };
                    std::string gone;
                    for(const auto& name:request.parts) if(gone.empty() && runtimePartName(name) && missing(runtimePartsDir()/name)) gone=name;
                    for(const auto& name:request.parts) {
                        if(!gone.empty()) break;
                        PendingPart part{name,nullptr,{},true};
                        try { part.manifest=std::make_shared<const json>(ReadRuntimePart(name)); }
                        catch(const std::exception& e) { part.error=e.what(); }
                        if(!part.manifest && runtimePartName(name) && missing(runtimePartsDir()/name)) { gone=name; break; }
                        std::unique_lock lock(runtimePartsMutex);
                        runtimeReader.wait(lock,[&] { return runtimePartsPending.size()<kCachedPartsWaiting; });
                        runtimePartsPending.push_back(std::move(part));
                    }
                    if(!gone.empty())
                        logger::info("Runtime bake: cached part {} is gone, as its cell is being baked again; the cell's other cached files are not used",gone);
                    else {
                        std::erase_if(request.tables,[&](const std::string& name) { return runtimePrevisName(name) && missing(runtimePrevisDir()/name); });
                        std::lock_guard lock(runtimePartsMutex);
                        runtimePrevisPending.insert(runtimePrevisPending.end(),request.tables.begin(),request.tables.end());
                    }
                } catch(...) {
                    static std::once_flag once;
                    std::call_once(once,[] { logger::error("Runtime bake: a cached cell could not be read; it keeps its originals"); });
                }
                std::lock_guard lock(runtimePartsMutex);
                --runtimeReading;
            }
        }

        // Parts by time: registering one (ParseManifest) runs on a game thread, and a cell entered with a warm cache hands
        // over all of its parts at once. Previs tables name chunks of parts handed over before them, so they wait until
        // every part handed over and every cached cell requested is registered.
        void LoadPendingRuntimeParts() {
            const auto begin=Clock::now();
            for(bool first=true;first || Clock::now()-begin<kPartBudget;first=false) {
                PendingPart part;
                {
                    std::lock_guard lock(runtimePartsMutex);
                    if(runtimePartsPending.empty()) break;
                    part=std::move(runtimePartsPending.front()); runtimePartsPending.pop_front();
                }
                runtimeReader.notify_all();
                if(part.manifest) LoadRuntimePart(part.name,true,part.manifest.get(),part.cached);
                else logger::error("Rejected runtime bake part {}: {}",part.name,part.error);
            }
            std::vector<std::string> tables;
            {
                std::lock_guard lock(runtimePartsMutex);
                if(runtimePartsPending.empty() && runtimeReading==0) tables.swap(runtimePrevisPending);
            }
            std::vector<std::string> waiting;
            for(auto& name:tables) {
                // A table the bake offers for a cell it found unchanged, before the cell's cached parts were read: it waits
                // for them.
                const auto key=cachedKeys.find(name.substr(0,name.find('-')));
                if(key!=cachedKeys.end() && RequestCachedCell(key->second)) waiting.push_back(std::move(name));
                else LoadRuntimePrevis(name);  // after the parts it names
            }
            if(!waiting.empty()) {
                std::lock_guard lock(runtimePartsMutex);
                runtimePrevisPending.insert(runtimePrevisPending.begin(),std::make_move_iterator(waiting.begin()),std::make_move_iterator(waiting.end()));
            }
        }

        // A runtime-baked cell's previs table (the bake plugin renders it after the cell's last part): for each view cell
        // its live navmesh touches, the chunks visible from there. It names the bake its chunks came from and lists them by
        // mesh, so it applies only while exactly those chunks are the cell's runtime definitions; a newer bake drops it.
        // ApplyPrevis then treats it as a shipped table (config "previs"; main camera only with the culling hooks).
        bool LoadRuntimePrevis(const std::string& name) {
            try {
                need(runtimePrevisName(name),"invalid previs table name");
                const auto path=runtimePrevisDir()/name;
                need(fs::file_size(path)<=256*1024*1024,"previs table too large");
                const auto table=json::parse(readFile(path));
                const auto kind=table.at("kind").get<std::string>();
                need(table.at("schema")==1 && (kind=="runtime_pvs" || kind=="runtime_pvs_exterior"),"unsupported previs table");
                const bool exterior=kind=="runtime_pvs_exterior";
                const auto cell=resolveForm(table.at("cell"));
                need(cell!=0,"the cell's plugin is not loaded");
                const auto runtime=runtimeCells.find(cell);
                const auto bake=table.at("bake").get<std::string>();
                need(runtime!=runtimeCells.end() && runtime->second.bake==bake,"the table belongs to another bake of the cell");
                need(name==(exterior ? std::format("{}-{}-from-{}.pvs.json",table.at("cell_key").get<std::string>(),bake,table.at("from_cell_key").get<std::string>())
                                     : std::format("{}-{}.pvs.json",table.at("cell_key").get<std::string>(),bake)),"table name does not match its contents");
                const auto own=byCell.find(cell);
                need(own!=byCell.end(),"the cell has no runtime chunks");
                std::unordered_map<std::string,size_t> byMesh;
                for(const auto index:own->second) byMesh.emplace(normalized(definitions[index].mesh),index);
                Previs pvs; pvs.size=table.at("view_cell_size");
                need(std::isfinite(pvs.size) && pvs.size>=32 && pvs.size<=4096,"invalid view cell size");
                std::set<size_t> seen;
                for(const auto& mesh:table.at("objects")) {
                    const auto found=byMesh.find(normalized(mesh.get<std::string>()));
                    need(found!=byMesh.end() && seen.insert(found->second).second,"previs object is not a unique runtime chunk of this cell");
                    pvs.definitions.push_back(found->second);
                }
                DecodeViewCells(pvs,table.at("view_cells"));
                if(exterior) {
                    // One camera cell's view cells: merged into the cell's table when it lists the same chunks at the same
                    // view-cell size, else it starts the table afresh.
                    pvs.exterior=true;
                    const auto& grid=table.at("from_grid");
                    const std::pair<int32_t,int32_t> from{grid.at(0).get<int32_t>(),grid.at(1).get<int32_t>()};
                    const auto existing=previsByCell.find(cell);
                    if(existing!=previsByCell.end() && existing->second.exterior && existing->second.size==pvs.size &&
                       existing->second.definitions==pvs.definitions) {
                        for(auto& [key,words]:pvs.visible) existing->second.visible.insert_or_assign(key,std::move(words));
                        existing->second.covered.insert(from);
                        logger::info("Runtime exterior previs for cell {:08X}: view cells of camera cell {},{} added ({} camera cells, {} view cells)",cell,
                            from.first,from.second,existing->second.covered.size(),existing->second.visible.size());
                        return true;
                    }
                    pvs.covered.insert(from);
                }
                DropRuntimePrevis(cell);
                previsBit.resize(definitions.size(),-1);
                for(size_t b=0;b<pvs.definitions.size();++b) previsBit[pvs.definitions[b]]=static_cast<int64_t>(b);
                logger::info("Runtime {}previs for cell {:08X}: {} of its chunks, {} view cells of {} units",exterior ? "exterior " : "",cell,
                    pvs.definitions.size(),pvs.visible.size(),pvs.size);
                previsByCell.insert_or_assign(cell,std::move(pvs));
                return true;
            } catch(const std::exception& e) { logger::error("Rejected runtime previs table {}: {}",name,e.what()); return false; }
        }
        void DropRuntimePrevis(uint32_t cell) {
            const auto found=previsByCell.find(cell);
            if(found==previsByCell.end() || !runtimeCells.contains(cell)) return;
            for(const auto index:found->second.definitions) if(index<previsBit.size()) previsBit[index]=-1;
            previsByCell.erase(found);
        }

        void RetireRuntimeCell(uint32_t cell) {
            DropRuntimePrevis(cell);
            const auto removed=RemoveChunks([cell](uint32_t id) { return id==cell; });
            if(auto it=byCell.find(cell);it!=byCell.end()) {
                DropPreloaded(&it->second);
                for(const auto index:it->second) { slots[index]=Slot{}; runtimeMeshes.erase(normalized(definitions[index].mesh)); }
                byCell.erase(it);
            }
            eligibleCells.erase(cell);
            logger::info("Runtime bake: cell {:08X} was baked again; its previous parts are retired ({} chunks restored)",cell,removed);
        }

        // One part manifest. Returns 1 when it was added.
        // Reads and parses a part file (any thread).
        static json ReadRuntimePart(const std::string& name) {
            need(runtimePartName(name),"invalid part name");
            const auto path=runtimePartsDir()/name;
            need(fs::file_size(path)<=64*1024*1024,"part too large");
            return json::parse(readFile(path));
        }
        // `preload`: queue its chunk meshes now (a part handed over in play: its cell is attached); otherwise Prepared asks
        // for them once the cell is attached and a chunk is tried (at data load every cached chunk was hashed
        // and parsed wherever the player was). `cached`: read from the cache, which never replaces a bake of this session.
        size_t LoadRuntimePart(const std::string& name,bool preload,const json* parsed=nullptr,bool cached=false) {
            try {
                need(runtimePartName(name),"invalid part name");
                const json read=parsed ? json() : ReadRuntimePart(name);
                const json& manifest=parsed ? *parsed : read;
                const auto& part=manifest.at("runtime_part");
                const auto key=part.at("cell_key").get<std::string>();
                const auto bake=part.at("bake").get<std::string>();
                const auto index=part.at("index").get<size_t>();
                need(name==std::format("{}-{}-{}.json",key,bake,index),"part name does not match its contents");
                need(part.value("baker",0)==precombine::kOutputVersion,"made by another version of the baker");
                const auto cell=resolveForm(manifest.at("cell"));
                need(cell!=0,"the cell's plugin is not loaded");
                need(!manifestHashes.contains(cell),"the cell has a manifest in the manifest folder");
                auto& runtime=runtimeCells[cell];
                if(runtime.bake!=bake) {
                    if(cached && !runtime.bake.empty()) {
                        logger::info("Runtime bake part {} not used: cell {:08X} was baked again this session",name,cell);
                        return 0;
                    }
                    if(!runtime.bake.empty()) RetireRuntimeCell(cell);
                    runtime=RuntimeCell{bake,{},{}};
                }
                if(runtime.parts.contains(index)) return 0;
                // The part's own chunk paths and source shapes, checked against those registered before it and merged in
                // only once the whole part is accepted, so it stays all or nothing without a copy of the global sets
                // (copying runtimeMeshes for each part made data load O(parts x chunks) and every hand-off
                // O(every cached chunk) on a game thread).
                std::set<std::string> coverage, meshes;
                auto parsed=ParseManifest(manifest,coverage,meshes,"precombined/skyrim/rb/");
                need(parsed.cell==cell,"part cell differs");
                for(const auto& mesh:meshes) need(!runtimeMeshes.contains(mesh),"invalid or duplicate chunk path");
                for(const auto& shape:coverage) need(!runtime.coverage.contains(shape),"source shape covered more than once");
                std::vector<MeshJob> jobs;
                for(auto& def:parsed.definitions) {
                    if(preload) jobs.push_back({def.mesh,def.bytes,def.hash,true});
                    byCell[cell].push_back(definitions.size());
                    definitions.push_back(std::move(def));
                }
                if(preload) Preload(std::move(jobs));
                slots.resize(definitions.size()); previsBit.resize(definitions.size(),-1);  // runtime parts have no previs table
                runtime.parts.insert(index); runtime.coverage.merge(coverage); runtimeMeshes.merge(meshes);
                logger::info("Runtime bake part {} loaded: {} chunk definitions for cell {:08X}",name,parsed.definitions.size(),cell);
                return 1;
            } catch(const std::exception& e) { logger::error("Rejected runtime bake part {}: {}",name,e.what()); return 0; }
        }

        void Report(Clock::time_point now) {
            size_t hidden=0; for(const auto& chunk:active) hidden+=chunk.sources.size();
            std::string cells; for(auto id:eligibleCells) cells+=std::format("{}{:08X}",cells.empty()?"":",",id);
            logger::info("Cells [{}]: {} active chunks ({} hidden by previs), {} hidden shapes, {} culled roots, {} dissolved since enabling, {} source lights assigned in total, {} references hidden by reference previs",
                cells,active.size(),previsHidden,hidden,culledRoots.size(),dissolved,statLightsSynced.load(relaxed),referencePrevisHidden);
            if(const auto hiz=Occlusion::Read();hiz.enabled)
                logger::info("Hi-Z occlusion: {} pyramids from {} captures; last frame fade nodes {}/{} culled, geometry {}/{} culled; render thread {} us, helper build {} us, grid age {}, {} texels landed, {} late pyramids, {} busy frames, {} moving references excluded, {} captures dropped for moving references, {} captures with a moving reference at the eye exempted, {} waits ({} timed out), frame {:.2f} ms, {} fast frames skipped, {} low-yield frames skipped; last frame {} recently visible objects not tested again, {} fade nodes kept whole for their fade",
                    hiz.builds,hiz.captures,hiz.fadeCulled,hiz.fadeTested,hiz.geometryCulled,hiz.geometryTested,hiz.prepareMicros,hiz.buildMicros,hiz.gridAge,hiz.landed,
                    hiz.lateBuilds,hiz.skippedFrames,hiz.movingExcluded,hiz.movingSkipped,hiz.eyeExempted,hiz.waits,hiz.waitTimeouts,hiz.frameMs,hiz.fastFrames,hiz.yieldSkipped,hiz.skipped,hiz.fadeKept);
            if(const auto hiz=Occlusion::Read();hiz.enabled && (hiz.boxes || hiz.boxPasses))
                logger::info("Box occlusion {}: last frame {} boxes registered, {} geometries hidden, {} without a box, {} at the screen edge or the near plane; "
                             "{} hidden in the result in use; {} passes, {} results read back, {} frames moved too far, {} frames without a fresh result",
                    hiz.boxes ? "ON":"OFF",hiz.boxRegistered,hiz.boxCulled,hiz.boxNoModel,hiz.boxEdge,hiz.boxTableHidden,hiz.boxPasses,hiz.boxReadbacks,
                    hiz.boxShifted,hiz.boxStale);
            if(const auto hiz=Occlusion::Read();hiz.enabled && (hiz.boxes || hiz.boxPasses)) {
                std::string types;
                for(size_t i=0;i<hiz.boxTypes.size();++i) if(hiz.boxTypes[i]) types+=std::format("{}type {}: {}",types.empty() ? "" : ", ",i,hiz.boxTypes[i]);
                logger::info("Box occlusion meshes: last frame without a box {} (other type or no renderer data {}, vertices on their way {}, vertices outside "
                             "the model bound {}); since the start {} boxes from GPU vertex copies, {} copies refused, {} boxes from the engine's CPU copy, "
                             "{} requests dropped, {} waiting; without a box by geometry type: {}",
                    hiz.boxNoModel,hiz.boxType,hiz.boxWaiting,hiz.boxInvalid,hiz.boxMeshReads,hiz.boxMeshInvalid,hiz.boxCpuBoxes,hiz.boxMeshDropped,
                    hiz.boxQueued,types.empty() ? std::string("none") : types);
            }
            if(const auto hiz=Occlusion::Read();hiz.enabled) {
                const auto cost=Occlusion::ReadCost();
                if(cost.sampledFrames) logger::info("Hi-Z cost per sampled frame ({} frames): Hi-Z {:.1f} us over {:.0f} calls (camera {:.1f} us; for {:.0f} main-camera calls pyramid {:.1f} us, visible-object table {:.1f} us; eligibility and occlusion tests {:.1f} us for {:.0f}), moving-reference notes {:.1f} us, previs lookups {:.1f} us; "
                             "pyramid published {:.0f} us and first test {:.0f} us after the build started, first test before the pyramid in {} of {} frames; "
                             "last frame {} tests while the pyramid was built; near exemption widened {} times at the eye, {} for bounds reaching the camera plane, widest {:.0f} units; "
                             "{} actor bounds replaced by the actor's box",
                    cost.sampledFrames,cost.hizMicros,cost.hizCallsPerFrame,cost.stageMicros[0],cost.candidatesPerFrame,cost.stageMicros[1],cost.stageMicros[2],
                    cost.stageMicros[3],cost.testsPerFrame,cost.movingMicros,cost.previsMicros,cost.publishMicros,cost.firstTestMicros,
                    cost.raceLost,cost.raceFrames,hiz.untested,hiz.eyeAtEye,hiz.eyeRect,cost.eyeWidest,hiz.boundsReplaced);
            }
            if(stageTicks) {
                const double divisor=1000.0*stageTicks;
                logger::info("Validation stages: {} tasks; reference mean {:.1f} us, scene mean {:.1f} us, activation/count mean {:.1f} us",
                    stageTicks,referenceNs/divisor,sceneNs/divisor,activationNs/divisor);
                stageTicks=referenceNs=sceneNs=activationNs=0;
            }
            if(!tickMicros.empty()) {
                std::sort(tickMicros.begin(),tickMicros.end());
                const auto at=[&](double q){ return tickMicros[std::min(tickMicros.size()-1,static_cast<size_t>(q*(tickMicros.size()-1)+0.5))]; };
                std::string threads; for(auto id:tickThreads) threads+=(threads.empty()?"":",")+std::to_string(id);
                logger::info("Task cost since last report: {} ticks, median {} us, p99 {} us, max {} us; threads {}",tickMicros.size(),at(.5),at(.99),tickMicros.back(),threads);
                tickMicros.clear(); tickThreads.clear();
            }
            for(const auto& [reason,count]:rejections) logger::info("Deferred/rejected since last report: {} ({})",reason,count);
            rejections.clear(); lastReport=now;
        }

        // The engine source exists once the cell system is initialized; retried every task until then.
        void RegisterCellSink() {
            if(cellSinkRegistered) return;
            // ICellAttachDetachEventSource*, written by its constructor; its event source is at +8 in all three runtimes.
            static REL::Relocation<std::uintptr_t*> holder{REL::VariantID(514281,400441,0x1F850D8)};
            const auto instance=*holder.get();
            if(!instance) return;
            reinterpret_cast<RE::BSTEventSource<CellAttachDetachEvent>*>(instance+8)->AddEventSink(static_cast<RE::BSTEventSink<CellAttachDetachEvent>*>(this));
            cellSinkRegistered=true;
            logger::info("Registered for engine cell attach/detach events");
        }

        // The interior cell and every attached grid cell that has definitions.
        void UpdateEligibleCells() {
            eligibleCells.clear(); attachedCells.clear();
            const auto add=[this](RE::TESObjectCELL* cell) {
                if(!cell || !cell->IsAttached()) return;
                const auto id=cell->GetFormID();
                if(!detachingCells.contains(id)) attachedCells.insert(id);
                if(byCell.contains(id) && !excluded.contains(id) && !detachingCells.contains(id)) eligibleCells.insert(id);
            };
            auto* tes=RE::TES::GetSingleton();
            if(!tes) return;
            add(tes->interiorCell);
            if(auto* grid=tes->gridCells) for(uint32_t x=0;x<grid->length;++x) for(uint32_t y=0;y<grid->length;++y) add(grid->GetCell(x,y));
        }

        // Restore every active chunk whose cell matches; used before the engine tears a cell down.
        size_t RemoveChunks(const std::function<bool(uint32_t)>& cellMatches) {
            size_t removed=0;
            for(auto it=active.begin();it!=active.end();) {
                if(!cellMatches(definitions[it->definition].cell)) { ++it; continue; }
                slots[it->definition]=Slot{};
                Restore(*it); it=active.erase(it); ++removed;
            }
            if(removed) {
                std::erase_if(referenceChecks,[](const auto& entry) { return entry.second.expired(); });
                std::erase_if(ancestorChecks,[](const auto& entry) { return entry.second.expired(); });
                PublishCounts();
            }
            return removed;
        }

        RE::BSEventNotifyControl ProcessEvent(const CellAttachDetachEvent* event,RE::BSTEventSource<CellAttachDetachEvent>*) override {
            if(!event || !event->cell) return RE::BSEventNotifyControl::kContinue;
            const auto id=event->cell->GetFormID();
            std::lock_guard lock(stateMutex);
            if(!byCell.contains(id)) return RE::BSEventNotifyControl::kContinue;
            switch(event->status) {
            case CellAttachDetachEvent::kPreDetach: {
                // Runs inside the engine's cell detach, before any of the cell's 3D is torn down.
                detachingCells.insert(id); eligibleCells.erase(id);
                if(const auto removed=RemoveChunks([id](uint32_t cell) { return cell==id; }))
                    logger::info("Cell {:08X} detaching: restored {} chunks",id,removed);
                break;
            }
            case CellAttachDetachEvent::kPostAttach: {
                detachingCells.erase(id);
                // References load their 3D after the cell attaches; start this cell's attempts afresh.
                const auto retry=Clock::now()+std::chrono::milliseconds(500);
                for(auto index:byCell.at(id)) if(!slots[index].active) slots[index]=Slot{false,0,retry};
                break;
            }
            case CellAttachDetachEvent::kPostDetach:
                detachingCells.erase(id);
                DropPreloaded(&byCell.at(id));  // parsed again when the cell attaches again
                break;
            default:
                break;
            }
            return RE::BSEventNotifyControl::kContinue;
        }

        void TickBody() {
            WriteFrameRow(Clock::now());
            WriteAbRows();
            if(suspended.load()) return;
            RegisterCellSink();
            auto* ui=RE::UI::GetSingleton();
            const bool menu=!ui || ui->IsMenuOpen("Main Menu") || ui->IsMenuOpen("Loading Menu");
            if(runtimeBake.load(relaxed)) {
                LoadPendingRuntimeParts();
                if(!menu) RequestAttachedCells();
            }
            if(!enabled || menu) {
                if(!active.empty() || !eligibleCells.empty()) Clear(); return;
            }
            UpdateEligibleCells();
            // Fallback for a missed event: a chunk never outlives its cell's attachment.
            if(const auto removed=RemoveChunks([this](uint32_t cell) { return !eligibleCells.contains(cell); })) {
                rejections["cell no longer attached"]+=removed;
            }
            if(eligibleCells.empty()) {
                // Reference previs also covers cells without precombined chunks.
                if(!referencePrevisByCell.empty()) ApplyPrevis();
                return;
            }
            const auto now=Clock::now();
            validation.BeginTick();
            const auto epoch=validation.Epoch();
            // Do not restore while sampling: restoration changes AppCulled and
            // must not contaminate another chunk's checks in this task.
            const auto referenceBegin=Clock::now();
            bool anyInvalid=!referenceSchedule.Visit(epoch,[&](auto& ref) { return CheckReference(ref,epoch); });
            // Enable-parented sources are checked every task, so a chunk follows its parent within one task.
            std::erase_if(parentedChecks,[&](const auto& weak) {
                const auto check=weak.lock();
                if(check) anyInvalid|=!CheckReference(*check,epoch);
                return !check;
            });
            const auto sceneBegin=Clock::now();
            for(auto& chunk:active) {
                const auto& def=definitions[chunk.definition];
                bool valid=chunk.mesh && chunk.mesh->parent==chunk.parent.get() &&
                    chunk.parentWorldCheck.Get(chunk.parent->world,[&](const auto& world) { return closeEnough(world,chunk.parentWorld); });
                if(valid && validation.Due(chunk.definition)) {
                    for(size_t i=0;valid && i<chunk.sources.size();++i) valid=StillValid(chunk.sources[i],def.sources[i],epoch,chunk.materialFlags);
                }
                chunk.valid=valid;
                anyInvalid|=!valid;
            }
            for(auto it=active.begin();anyInvalid && it!=active.end();) {
                const auto& def=definitions[it->definition];
                bool valid=it->valid;
                // A changed reference or ancestor dissolves every dependent
                // chunk in this task, including chunks outside today's slice.
                for(const auto& binding:it->sources) if(DependencyFailed(binding,epoch)) { valid=false; break; }
                if(valid) { ++it; continue; }
                const auto index=it->definition;
                Restore(*it); it=active.erase(it);
                Defer(index,now); ++dissolved; statDissolves.fetch_add(1,relaxed);
                logger::warn("Dissolved {}; originals restored, retry in {} ms",def.mesh,
                    std::chrono::duration_cast<std::chrono::milliseconds>(slots[index].retryAt-now).count());
            }
            // Bounded activation work across every eligible cell's due definitions.
            const auto activationBegin=Clock::now();
            candidates.clear();
            for(auto id:eligibleCells) { const auto& list=byCell.at(id); candidates.insert(candidates.end(),list.begin(),list.end()); }
            size_t attempted=0;
            // At most attachesPerTick per task, and nothing after activationBudget (the first candidate always is visited):
            // activating a chunk runs on a game thread (6 frames over 33 ms while 731 runtime-baked chunks
            // activated). Chunks still waiting for their meshes count against the time too, and the cursor carries the rest
            // to the next task (while a cell's meshes were prepared nothing was attempted, so every due chunk was
            // visited in every task).
            for(size_t visited=0;visited<candidates.size() && attempted<attachesPerTick &&
                                 (visited==0 || Clock::now()-activationBegin<activationBudget);++visited) {
                if(cursor>=candidates.size()) cursor=0;
                const auto index=candidates[cursor++];
                if(slots[index].active || slots[index].retryAt>now) continue;
                meshPending=false;
                if(Attach(index)) { ++attempted; continue; }
                // Every other task or so: at 50 ms a waiting chunk was due again in every 100 ms task.
                if(meshPending) { slots[index].retryAt=now+std::chrono::milliseconds(200); continue; }
                ++attempted;
                Defer(index,now);
            }
            ApplyPrevis();
            PublishCounts();
            const auto stageEnd=Clock::now();
            ++stageTicks;
            referenceNs+=std::chrono::duration_cast<std::chrono::nanoseconds>(sceneBegin-referenceBegin).count();
            sceneNs+=std::chrono::duration_cast<std::chrono::nanoseconds>(activationBegin-sceneBegin).count();
            activationNs+=std::chrono::duration_cast<std::chrono::nanoseconds>(stageEnd-activationBegin).count();
            if(developerMode && now-lastReport>std::chrono::seconds(10)) Report(now);
        }

        void Tick() {
            const auto begin=Clock::now();
            std::lock_guard lock(stateMutex);
            TickBody();
            const auto ns=static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-begin).count());
            statTicks.fetch_add(1,relaxed); statTickNs.fetch_add(ns,relaxed);
            for(auto prior=statTickMaxNs.load(relaxed);ns>prior && !statTickMaxNs.compare_exchange_weak(prior,ns,relaxed););
            const auto thread=REX::W32::GetCurrentThreadId();
            statThread.store(thread,relaxed);
            if(!eligibleCells.empty() && tickMicros.size()<4096) { tickMicros.push_back(static_cast<uint32_t>(ns/1000)); tickThreads.insert(thread); }
        }

        RE::BSEventNotifyControl ProcessEvent(RE::InputEvent* const* events,RE::BSTEventSource<RE::InputEvent*>*) override {
            if(!events) return RE::BSEventNotifyControl::kContinue;
            for(auto* event=*events;event;event=event->next) {
                auto* button=event->AsButtonEvent();
                if(!button || button->GetDevice()!=RE::INPUT_DEVICE::kKeyboard || !button->IsDown()) continue;
                const auto code=button->GetIDCode();
                // The toggles are a developer feature: bDeveloperShortcuts (live, the in-game menu) or developer_mode.
                if(!Settings::ShortcutsOn()) continue;
                if((toggleKey==0 || code!=toggleKey) && (previsToggleKey==0 || code!=previsToggleKey) && (hizToggleKey==0 || code!=hizToggleKey)
                   && (hizGeometryToggleKey==0 || code!=hizGeometryToggleKey) && (boxToggleKey==0 || code!=boxToggleKey)
                   && (probeKey==0 || code!=probeKey)) continue;
                // Typing a toggle's scan code into the console or a text field must not toggle, nor may a key pressed with
                // Ctrl, Alt or Shift held: Alt+F10 and the like belong to other programs (NVIDIA's overlay). A paused game is
                // no reason: a game on a hidden desktop counts as paused, and a toggle in the journal is harmless.
                // Each refusal is named in the log (the first 16).
                auto* ui=RE::UI::GetSingleton();
                auto* controls=RE::ControlMap::GetSingleton();
                const char* refused=ui && ui->IsMenuOpen(RE::Console::MENU_NAME) ? "the console is open" :
                                    controls && controls->GetRuntimeData().textEntryCount>0 ? "a text field is open" :
                                    ModifierHeld() ? "Ctrl, Alt or Shift is held" : nullptr;
                if(refused) {
                    static std::atomic<uint32_t> named{0};
                    if(named.fetch_add(1,std::memory_order_relaxed)<16) logger::info("Shortcut key {} ignored: {}",code,refused);
                    continue;
                }
                // Without the Hi-Z hooks (bHiZ=0 at startup with the developer features off, main.cpp, also when the shortcut
                // keys are turned on in play; VR without hiz_vr) only the precombines key works: the Hi-Z, per-object, box and
                // probe keys have nothing to switch, and previs would fall back to AppCulled, which also hides chunks from
                // water reflections (ApplyPrevis). The menu says the others work after a restart (Settings::Status).
                if((toggleKey==0 || code!=toggleKey) && !Occlusion::Installed()) {
                    logger::info("Hi-Z occlusion hooks are not installed this session; scan code {} does nothing",code);
                    continue;
                }
                if(probeKey!=0 && code==probeKey) {
                    const bool armed=Occlusion::Probe();
                    logger::info("Hi-Z probe {}",armed ? "armed: the next frame names what covers the crosshair" : "needs \"hiz_diagnostics\"");
                    continue;
                }
                if(boxToggleKey!=0 && code==boxToggleKey) {
                    const bool on=Occlusion::ToggleBoxes();
                    logger::info("Box occlusion {}",on ? "ON":"OFF");
                    SKSE::GetTaskInterface()->AddTask([on]{
                        if(auto* console=RE::ConsoleLog::GetSingleton()) console->Print("Precombine box occlusion %s",on ? "ON":"OFF");
                    });
                    continue;
                }
                if(hizGeometryToggleKey!=0 && code==hizGeometryToggleKey) {
                    const bool on=Occlusion::ToggleGeometry();
                    logger::info("Hi-Z per-geometry tests {}",on ? "ON":"OFF");
                    continue;
                }
                if(hizToggleKey!=0 && code==hizToggleKey) {
                    // Hi-Z occlusion only, for this session (the menu shows it; SkyrimPrecombines.ini keeps its value);
                    // precombines and previs stay as they are. Occlusion::SetEnabled logs the change.
                    const bool on=!Settings::GetBool("bHiZ");
                    Settings::SetBool("bHiZ",on,Settings::Save::kNo);
                    SKSE::GetTaskInterface()->AddTask([on]{
                        if(auto* console=RE::ConsoleLog::GetSingleton()) console->Print("Precombine Hi-Z occlusion %s",on ? "ON":"OFF");
                    });
                    continue;
                }
                if(code!=toggleKey) {
                    // Previs only: precombines stay as they are; turning it off shows every active chunk.
                    SKSE::GetTaskInterface()->AddTask([this]{
                        std::lock_guard lock(stateMutex);
                        previs=!previs; ApplyPrevis(); PublishCounts();
                        logger::info("Previs {}",previs ? "ON":"OFF");
                        if(auto* console=RE::ConsoleLog::GetSingleton()) console->Print("Precombine previs %s",previs ? "ON":"OFF");
                    });
                    continue;
                }
                // Precombines on or off for this session (the menu shows it; SkyrimPrecombines.ini keeps its value); after an
                // error (failed) the key turns them on, as before. SetEnabled's task switches and logs, and (announce)
                // reports the state it reached on the console and the HUD.
                const bool on=!Settings::GetBool("bEnabled") || failed.load(relaxed);
                announce=true;
                if(!Settings::SetBool("bEnabled",on,Settings::Save::kNo)) announce=false;
            }
            return RE::BSEventNotifyControl::kContinue;
        }

        // Ctrl, Shift or Alt held on the keyboard, left or right. Reads the device's key state as IsPressed does, without
        // calling it: CommonLib's BSWin32KeyboardDevice.obj also defines its constructor, whose vtable needs virtuals the
        // library never defines (unresolved at link).
        static bool ModifierHeld() {
            auto* devices=RE::BSInputDeviceManager::GetSingleton();
            auto* keyboard=devices ? devices->GetKeyboard() : nullptr;
            if(!keyboard) return false;
            const auto& keys=keyboard->GetRuntimeData().curState;
            using Key=RE::BSKeyboardDevice::Key;
            for(const auto key:{Key::kLeftControl,Key::kRightControl,Key::kLeftShift,Key::kRightShift,Key::kLeftAlt,Key::kRightAlt})
                if(key<sizeof keys && (keys[key]&0x80)!=0) return true;
            return false;
        }

        void Start() {
            std::lock_guard lock(stateMutex);
            if(started) return;
            if(!SupportedRuntime()) {
                logger::error("Skyrim {} is not a supported version (SE 1.5.97, AE 1.6.1170, 1.6.1179 GOG, 1.7.104, VR 1.4.15); activation disabled",
                    REL::Module::get().version().string());
                return;
            }
            ReadFiles(); ReadRuntimeParts(); started=true; suspended=false;
            RegisterCellSink();
            logger::info("Source validation sweep: {} tasks ({} ms nominal); references use independent stable groups, ancestors cached per task",
                ValidationSchedule::ticksPerSweep,ValidationSchedule::ticksPerSweep*100);
            RE::BSInputDeviceManager::GetSingleton()->AddEventSink(this);
            // Deliberately uses public SKSE tasks, not guessed game hooks. On SE 1.5.97
            // these run on the engine's task workers, serialized here by stateMutex.
            // Invalidation is polled (100 ms), not frame-synchronous.
            std::thread([this] {
                for(;;) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    if(suspended.load() || queued.exchange(true)) continue;
                    SKSE::GetTaskInterface()->AddTask([this] {
                        try { Tick(); }
                        catch(const std::exception& e) { std::lock_guard lock(stateMutex); logger::error("Precombines disabled after error: {}",e.what()); enabled=false; failed=true; Clear(); }
                        queued=false;
                    });
                }
            }).detach();
        }
    };
}
void Start() { try { Manager::Get().Start(); } catch(const std::exception& e) { logger::error("Precombines initialization failed: {}",e.what()); } }
void BeforeLoad() { auto& m=Manager::Get(); m.suspended=true; std::lock_guard lock(m.stateMutex); m.Clear(); m.DropPreloaded(nullptr); }
void AfterLoad() { auto& m=Manager::Get(); std::lock_guard lock(m.stateMutex); m.Clear(); m.suspended=false; }
// Called by Settings while it orders its Sets (Settings.cpp Store, lock order), from the menu and the precombines key: only
// queues the task, so no lock is held while waiting for stateMutex, and the task never sets Settings. Before Start the task
// does nothing: ReadSettings takes the value from Settings.
void SetEnabled(bool on) {
    auto* tasks=SKSE::GetTaskInterface();
    if(!tasks) return;
    tasks->AddTask([on]{
        auto& m=Manager::Get();
        std::lock_guard lock(m.stateMutex);
        const bool announce=m.announce.exchange(false);
        if(!m.started) return;
        if(m.enabled!=on) {
            m.enabled=on;
            if(on) { m.failed=false; m.ResumeRuntimeHandOff(); }
            else m.Clear();
            logger::info("Precombines {}",on ? "ON":"OFF");
        }
        if(!announce) return;
        if(auto* console=RE::ConsoleLog::GetSingleton()) console->Print("Precombines %s",m.enabled ? "ON (experimental)":"OFF: originals restored");
        RE::SendHUDMessage::ShowHUDMessage(m.enabled ? "Precombines enabled":"Precombines disabled");
    });
}
bool Failed() { return Manager::Get().failed.load(relaxed); }
}

extern "C" DLLEXPORT bool SkyrimPrecombines_GetStats(PrecombinesStats* out, bool resetMax) {
    if(!out || out->size<kPrecombinesStatsV3Size) return false;
    auto& m=Precombines::Manager::Get();
    out->version=out->size>=sizeof(PrecombinesStats) ? 4:3;
    out->ticks=m.statTicks.load(Precombines::relaxed);
    out->tickNanoseconds=m.statTickNs.load(Precombines::relaxed);
    out->tickMaxNanoseconds=resetMax ? m.statTickMaxNs.exchange(0,Precombines::relaxed) : m.statTickMaxNs.load(Precombines::relaxed);
    out->attaches=m.statAttaches.load(Precombines::relaxed);
    out->dissolves=m.statDissolves.load(Precombines::relaxed);
    out->activeChunks=m.statActive.load(Precombines::relaxed);
    out->hiddenSources=m.statHidden.load(Precombines::relaxed);
    out->lastTickThread=m.statThread.load(Precombines::relaxed);
    out->culledRoots=m.statCulled.load(Precombines::relaxed);
    out->previsHidden=m.statPrevis.load(Precombines::relaxed);
    out->reserved=0;
    if(out->version>=4) {
        const auto hiz=Occlusion::Read();
        out->hizEnabled=hiz.enabled; out->hizGridAge=hiz.gridAge;
        out->hizCaptures=hiz.captures; out->hizBuilds=hiz.builds;
        out->hizFadeTested=hiz.fadeTested; out->hizFadeCulled=hiz.fadeCulled;
        out->hizGeometryTested=hiz.geometryTested; out->hizGeometryCulled=hiz.geometryCulled;
        out->hizPrepareMicros=hiz.prepareMicros; out->hizLanded=hiz.landed;
        out->previsMainCulled=hiz.previsCulled; out->previsReferencesCulled=hiz.referencePrevisCulled;
    }
    return true;
}

// Runtime-bake hand-off: the SkyrimPrecombinesBake plugin offers each part it publishes (a file name in
// Data/SKSE/Plugins/Precombines/RuntimeBake/parts) from its worker thread; the next task loads it.
// Refuses every part with bRuntimeBake=0 in SkyrimPrecombines.ini, or bEnabled=0 outside developer mode.
// The bake plugin offers a cell's previs table (a file name in RuntimeBake/previs) once the table is written; the next
// task loads it after any parts still pending.
extern "C" DLLEXPORT bool SkyrimPrecombines_RuntimeBakePrevis(const char* name) {
    auto& m=Precombines::Manager::Get();
    if(!name || !m.runtimeBake.load() || !Precombines::runtimePrevisName(name)) return false;
    std::lock_guard lock(m.runtimePartsMutex);
    if(m.runtimePrevisPending.size()>=1024) return false;
    m.runtimePrevisPending.emplace_back(name);
    return true;
}
extern "C" DLLEXPORT bool SkyrimPrecombines_RuntimeBakePart(const char* name) {
    auto& m=Precombines::Manager::Get();
    if(!name || !m.runtimeBake.load() || !Precombines::runtimePartName(name)) return false;
    // Parsed here, on the bake plugin's worker thread, so the task that adds it only registers its chunks.
    Precombines::Manager::PendingPart part{name,nullptr,{}};
    try { part.manifest=std::make_shared<const nlohmann::json>(Precombines::Manager::ReadRuntimePart(name)); }
    catch(const std::exception& e) { part.error=e.what(); }
    std::lock_guard lock(m.runtimePartsMutex);
    if(m.runtimePartsPending.size()>=4096) return false;
    m.runtimePartsPending.push_back(std::move(part));
    return true;
}
