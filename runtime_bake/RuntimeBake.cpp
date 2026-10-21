#include "RuntimeBake.h"
#include "Config.h"
#include "ExteriorPrevis.h"
#include "PrecombineMeshCore.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <format>
#include <fstream>
#include <malloc.h>  // _resetstkoflw
#include <map>
#include <mutex>
#include <set>
#include <string_view>
#include <thread>
#include <tuple>

namespace RuntimeBake {
    namespace fs = std::filesystem;
    using json = nlohmann::json;
    using Clock = std::chrono::steady_clock;

    namespace {
        constexpr auto kManifestDir = "Data/SKSE/Plugins/Precombines";
        // The cache (Config::BakeDir and MeshRoot: under Data, or in the folder "sCacheFolder" names).
        fs::path WorkDir() { return Config::BakeDir(); }
        // Part manifests: a subfolder, so the precombine plugin's manifest-folder reader (not recursive) never takes
        // one for a cell manifest. Only a SkyrimPrecombines.dll with the hand-off export reads them.
        fs::path PartsDir() { return WorkDir()/"parts"; }
        fs::path RoomsDir() { return WorkDir()/"rooms"; }    // observed room groups per cell
        fs::path AssetDir() { return WorkDir()/"assets"; }   // this session's meshes, by hash
        fs::path PrevisDir() { return WorkDir()/"previs"; }  // <cell key>-<bake>.pvs.json
        fs::path MeshRoot() { return Config::MeshRoot(); }
        constexpr auto kNoMerges = "No eligible merges in this cell";  // PrecombineMesh: an interior part without chunks
        // As the offline generator: deleted, persistent, initially disabled, ignored, 0x2000000; bases also 0x800000.
        constexpr uint32_t kBadFlags = 0x20 | 0x400 | 0x800 | 0x1000 | 0x2000000;
        constexpr uint32_t kVisibleWhenDistant = 0x8000;  // exteriors: LOD-visible references stay original
        constexpr int kMaxPollsWaitingFor3D = 15;
        constexpr int kMaxPollsSettling = 30;  // 2 s polls: a minute at most for a cell's reference set to stop growing
        // Shader flags runtime material edits set: kProjectedUV, kMultiIndexSnow, kSnow (Seasons of Skyrim's
        // single-pass snow sets kProjectedUV and kSnow through NiAVObject::SetProjectedUVData).
        constexpr uint64_t kRuntimeMaterialFlags = (1ull<<23) | (1ull<<41) | (1ull<<60);

        Settings settings;
        int RunTool(const fs::path& tool,const fs::path& input,const fs::path& output,const fs::path& report,DWORD priority,
                    uint32_t timeoutSeconds,std::string& error);
        std::string ReadText(const fs::path& path);
        void GatherNavmesh(RE::TESObjectCELL* cell,Job& job);
        void GatherTerrain(RE::TESObjectCELL* cell,Job& job);
        struct ExteriorCell;
        void OfferExterior(const Job& job,const std::string& cellKey,const std::string& key,const std::string& bake,json chunks);
        std::mutex queueMutex;
        std::condition_variable queueReady;
        std::vector<Job> queue;             // the worker takes the nearest job first
        std::set<uint32_t> queuedOrDone;    // this session; guarded by queueMutex
        std::set<uint32_t> prebaked;        // door destinations gathered this session; guarded by queueMutex
        std::map<uint32_t,int> waitingFor3D;      // Poll only (pollMutex)
        // Poll only: a cell's eligible-reference count and how many polls it has held. A large interior loads its
        // references over several frames, so its first poll can see a fraction of them, all with 3D (68 of
        // Ustengrav02's references were baked 4 s after entry).
        std::map<uint32_t,std::pair<size_t,int>> settling;
        std::map<uint32_t,size_t> destinationSizes;  // Poll only: a door destination's eligible references last poll
        std::set<uint32_t> shippedCells;    // cells with a manifest in the manifest folder not merged by this plugin
        std::atomic_bool started{false};
        // SkyrimPrecombines_RuntimeBakePart(fileName) of a SkyrimPrecombines.dll with the hand-off; null otherwise.
        using OfferPart = bool (*)(const char*);
        OfferPart offerPart = nullptr;
        OfferPart offerPrevis = nullptr;  // SkyrimPrecombines_RuntimeBakePrevis(fileName)
        // Without the hand-off a release SkyrimPrecombines.dll reads one merged manifest per cell at the next data load,
        // and it accepts Skyrim.esm records only, so only those are gathered.
        bool legacyForms() { return offerPart==nullptr; }
        // Meshes read through the resource system, by model path. Worker thread only. Resources do not change during
        // a session (the precombine plugin's own fingerprint cache assumes the same), so each model is read once.
        struct Asset { std::string hash; uint64_t bytes=0; fs::path file; bool ok=false; };
        std::map<std::string,Asset> assetStore;

        std::string Lower(std::string s) { std::transform(s.begin(),s.end(),s.begin(),[](unsigned char c){ return static_cast<char>(std::tolower(c)); }); return s; }
        // A path as UTF-8, for JSON and the log. path::string() gives the ANSI code page's bytes, which nlohmann's dump()
        // refuses (type_error 316), and throws for characters that page lacks: one accented letter in the game or cache
        // folder failed every cell. The baker and VisibilityEstimate decode these fields as UTF-8.
        std::string Utf8(const fs::path& path) { const auto text=path.u8string(); return std::string(text.begin(),text.end()); }
        // Plugin text (model paths, plugin names, editor IDs) is in its author's code page. It goes into the scene and the
        // parts as JSON, so it must be valid UTF-8 (RFC 3629, as strictly as nlohmann's dump() checks: no overlong form,
        // surrogate or code point past U+10FFFF), which ASCII is: such text passes through JSON byte for byte and still
        // matches the game's strings. Other bytes made dump() throw for the whole cell (type_error 316), and a
        // converted string would no longer match.
        bool ValidUtf8(std::string_view text) {
            for(size_t i=0;i<text.size();) {
                const auto c=static_cast<unsigned char>(text[i++]);
                if(c<0x80) continue;
                // The lead byte gives the length and the range of the first continuation byte.
                size_t more=0; unsigned char lo=0x80, hi=0xBF;
                if(c>=0xC2 && c<=0xDF) more=1;
                else if(c>=0xE0 && c<=0xEF) { more=2; if(c==0xE0) lo=0xA0; if(c==0xED) hi=0x9F; }
                else if(c>=0xF0 && c<=0xF4) { more=3; if(c==0xF0) lo=0x90; if(c==0xF4) hi=0x8F; }
                else return false;
                if(text.size()-i<more) return false;
                for(size_t k=0;k<more;++k,lo=0x80,hi=0xBF)
                    if(const auto b=static_cast<unsigned char>(text[i++]);b<lo || b>hi) return false;
            }
            return true;
        }
        // An editor ID only names a cell in the log and the scene: as it is when valid UTF-8, else its ASCII characters.
        std::string EditorID(const char* text) {
            if(!text) return {};
            if(ValidUtf8(text)) return text;
            std::string out;
            for(const char* c=text;*c;++c) if(static_cast<unsigned char>(*c)<0x80) out+=*c;
            return out;
        }

        FormKey KeyOf(const RE::TESForm* form) {
            const auto* file=form ? form->GetFile(0) : nullptr;  // the plugin that defines the form
            if(!file) return {};                                  // created at runtime: no stable identity
            return {std::string(file->GetFilename()),form->GetLocalFormID()};
        }
        bool IsSkyrimEsm(const FormKey& key) { return Lower(key.plugin)=="skyrim.esm"; }
        bool WinsInSkyrimEsm(const RE::TESForm* form) {
            const auto* file=form ? form->GetFile() : nullptr;  // the winning override
            return file && Lower(std::string(file->GetFilename()))=="skyrim.esm";
        }
        // Old (Skyrim.esm-only) manifests name forms by number; parts name them by plugin and local ID, which the
        // precombine plugin resolves for the actual load order and checks field by field against the winning record.
        json FormJson(const FormKey& key) {
            if(legacyForms()) return key.local;
            return json{{"plugin",key.plugin},{"id",key.local}};
        }
        std::string KeyText(const FormKey& key) { return std::format("{}|{:06x}",Lower(key.plugin),key.local); }

        // The loaded root's parent chain up to the cell root, by name (index where unnamed): stable across sessions,
        // and equal exactly for references the engine placed under one room node.
        std::string RoomGroup(RE::NiAVObject* root) {
            std::string chain;
            for(auto* node=root ? root->parent : nullptr;node;node=node->parent) {
                std::string name=node->name.c_str();
                if(name.empty() && node->parent) {
                    const auto& children=node->parent->GetChildren();
                    for(uint32_t i=0;i<children.size();++i) if(children[i].get()==node) { name=std::format("#{}",i); break; }
                }
                chain+=name+'/';
            }
            return "p"+HashHex(chain.data(),chain.size());
        }

        // The live 3D differs from its NIF in a way the baker cannot see: Seasons of Skyrim's single-pass winter snow
        // tags the root with NiBooleanExtraData "SOS_SNOW_SHADER" and sets kProjectedUV and kSnow on its lighting
        // shaders. The baker merges NIF materials, so such a reference stays original (and the precombine plugin
        // refuses a chunk whose source shows this at attach). Also true for NIFs that carry these flags themselves.
        bool RuntimeMaterial(RE::NiAVObject* root) {
            static const RE::BSFixedString snowTag("SOS_SNOW_SHADER");
            if(root->GetExtraData(snowTag)) return true;
            bool changed=false;
            RE::BSVisit::TraverseScenegraphGeometries(root,[&](RE::BSGeometry* geometry) {
                auto* property=geometry->GetGeometryRuntimeData().shaderProperty.get();
                if(property && (property->flags.underlying()&kRuntimeMaterialFlags)) { changed=true; return RE::BSVisit::BSVisitControl::kStop; }
                return RE::BSVisit::BSVisitControl::kContinue;
            });
            return changed;
        }

        // Why references were left out of the latest gather, by reason (Poll only; the game's gather found 68
        // eligible references in Ustengrav02 where the offline generator merges 1,891).
        std::map<std::string,uint32_t> rejected;
        bool Reject(const char* reason) { ++rejected[reason]; return false; }
        const char* ExtraName(RE::ExtraDataType type) {
            switch(type) {
            case RE::ExtraDataType::kLinkedRef: return "linked reference";
            case RE::ExtraDataType::kLinkedRefChildren: return "linked children";
            case RE::ExtraDataType::kActivateRef: return "activate parent";
            case RE::ExtraDataType::kActivateRefChildren: return "activate children";
            case RE::ExtraDataType::kOwnership: return "ownership";
            case RE::ExtraDataType::kLock: return "lock";
            case RE::ExtraDataType::kTeleport: return "teleport";
            case RE::ExtraDataType::kLocationRefType: return "location reference type";
            case RE::ExtraDataType::kMultiBoundRef: return "multibound";
            case RE::ExtraDataType::kEmittanceSource: return "emittance";
            case RE::ExtraDataType::kPrimitive: return "primitive";
            default: return "other extra data";
            }
        }
        bool Eligible(RE::TESObjectREFR* ref,bool interior,Reference& out) {
            if(!ref) return Reject("no reference");
            if(ref->GetFormFlags()&kBadFlags) return Reject("reference flags");
            if(ref->IsDisabled() || ref->IsDeleted() || ref->IsMarkedForDeletion()) return Reject("disabled or deleted");
            if(!interior && ref->GetFormFlags()&kVisibleWhenDistant) return Reject("visible when distant");
            auto* base=ref->GetBaseObject();
            if(!base || !base->Is(RE::FormType::Static)) return Reject("not a static");
            if(base->GetFormFlags()&(kBadFlags|0x800000)) return Reject("base flags");
            out.refKey=KeyOf(ref); out.baseKey=KeyOf(base);
            if(out.refKey.plugin.empty() || out.baseKey.plugin.empty()) return Reject("no plugin key");
            // Plugin names and model paths go into the scene as JSON, which must be UTF-8 (ValidUtf8).
            if(!ValidUtf8(out.refKey.plugin) || !ValidUtf8(out.baseKey.plugin)) return Reject("plugin name not UTF-8");
            if(legacyForms() && !(IsSkyrimEsm(out.refKey) && IsSkyrimEsm(out.baseKey) && WinsInSkyrimEsm(ref) && WinsInSkyrimEsm(base))) return Reject("not Skyrim.esm (legacy)");
            auto* stat=base->As<RE::TESObjectSTAT>();
            if(!stat || stat->data.materialObj || stat->numAlternateTextures) return Reject("material or alternate textures");  // directional material, MODS
            const char* model=stat->GetModel();
            if(!model || !*model) return Reject("no model");
            if(!ValidUtf8(model)) return Reject("model path not UTF-8");
            // The generator accepts only NAME/DATA/XSCL/EDID: anything that links, owns, locks or teleports stays original.
            for(const auto type:{RE::ExtraDataType::kLinkedRef,RE::ExtraDataType::kLinkedRefChildren,
                                 RE::ExtraDataType::kActivateRef,RE::ExtraDataType::kActivateRefChildren,RE::ExtraDataType::kOwnership,
                                 RE::ExtraDataType::kLock,RE::ExtraDataType::kTeleport,RE::ExtraDataType::kLocationRefType,
                                 RE::ExtraDataType::kEmittanceSource,RE::ExtraDataType::kPrimitive})
                if(ref->extraList.HasType(type)) return Reject(ExtraName(type));
            // No multibound exclusion: in room-based interiors the engine gives nearly every reference a multibound (room)
            // reference at load (2,320 of Ustengrav02's references), which the records do not carry. The offline
            // generator merges those references and room groups keep each chunk inside one room.
            // Enable-parented references merge only with references of the same parent and flag (their room group
            // carries both), and the precombine plugin checks the parent's state every task.
            out.enableParent={}; out.enableOpposite=false;
            if(auto* extra=ref->extraList.GetByType<RE::ExtraEnableStateParent>()) {
                const auto parent=extra->parent.get();
                if(legacyForms() || !parent) return Reject("enable parent");
                out.enableParent=KeyOf(parent.get());
                out.enableOpposite=(extra->flags&1)!=0;
                if(out.enableParent.plugin.empty()) return Reject("enable parent");
                if(!ValidUtf8(out.enableParent.plugin)) return Reject("plugin name not UTF-8");
            }
            const float scale=ref->GetScale();
            if(!std::isfinite(scale) || scale<=0.f || scale>10.f) return Reject("scale");
            out.ref=ref->GetFormID(); out.base=base->GetFormID(); out.model=NormalizeModel(model);
            const auto position=ref->GetPosition(); const auto angle=ref->GetAngle();
            out.position={position.x,position.y,position.z}; out.rotation={angle.x,angle.y,angle.z}; out.scale=scale;
            return true;
        }
        // Exterior previs occluders beside the eligible references: any static with a model and 3D that cannot vanish.
        // Visible-when-distant references (cliffs, large buildings: rejected for merging outdoors), material and
        // alternate-texture statics and linked ones occlude like the rest. Enable-parented and persistent references
        // stay out (a script can only disable a persistent reference), so a hidden chunk never shows a gap.
        bool Occluder(RE::TESObjectREFR* ref,Reference& out) {
            constexpr uint32_t kPersistent=1u<<10;
            if(!ref || (ref->GetFormFlags()&(kBadFlags|kPersistent)) || ref->IsDisabled() || ref->IsDeleted() || ref->IsMarkedForDeletion()) return false;
            if(ref->extraList.HasType(RE::ExtraDataType::kEnableStateParent) || !ref->Get3D()) return false;
            auto* base=ref->GetBaseObject();
            auto* stat=base ? base->As<RE::TESObjectSTAT>() : nullptr;
            if(!stat || (base->GetFormFlags()&(kBadFlags|0x800000))) return false;
            const char* model=stat->GetModel();
            const float scale=ref->GetScale();
            if(!model || !*model || !ValidUtf8(model) || !std::isfinite(scale) || scale<=0.f || scale>10.f) return false;
            out.refKey=KeyOf(ref); out.baseKey=KeyOf(base);
            out.ref=ref->GetFormID(); out.base=base->GetFormID(); out.model=NormalizeModel(model);
            const auto position=ref->GetPosition(); const auto angle=ref->GetAngle();
            out.position={position.x,position.y,position.z}; out.rotation={angle.x,angle.y,angle.z}; out.scale=scale;
            return true;
        }
        void AddEnableParentGroup(Reference& r) {
            if(!r.enableParent.plugin.empty()) r.roomGroup+=std::format("{}ep:{}:{}",r.roomGroup.empty() ? "" : "|",KeyText(r.enableParent),int(r.enableOpposite));
        }

        // A placed light's sphere, by reference: each light once however many cells list it. Runtime-created lights
        // (dropped torches, spawned ones) come and go, so they are left out of the budget and the cache key.
        using Lights=std::map<uint32_t,std::array<double,4>>;
        void LightOf(RE::TESObjectREFR* ref,Lights& out) {
            auto* base=ref ? ref->GetBaseObject() : nullptr;
            auto* light=base ? base->As<RE::TESObjectLIGH>() : nullptr;
            if(!light || ref->IsDeleted() || ref->IsDynamicForm()) return;
            double radius=light->data.radius;
            if(auto* extra=ref->extraList.GetByType<RE::ExtraRadius>()) radius+=extra->radius;  // XRDS is a signed adjustment
            const auto p=ref->GetPosition();
            if(radius>0 && std::isfinite(radius)) out.emplace(ref->GetFormID(),std::array<double,4>{p.x,p.y,p.z,radius});
        }
        // The spheres in value order. TESObjectCELL::references is a BSTSet hashed by pointer, so the order the game lists
        // them in follows heap addresses and changed every session, and with it the cache key: cells with two lights or
        // more were baked again in every session.
        std::vector<std::array<double,4>> SortedLights(const Lights& lights) {
            std::vector<std::array<double,4>> out;
            out.reserve(lights.size());
            for(const auto& [form,sphere]:lights) out.push_back(sphere);
            std::sort(out.begin(),out.end());
            return out;
        }

        // A cell this plugin may bake: a stable identity, and a Skyrim.esm cell when only old manifests are read.
        bool BakeableCell(RE::TESObjectCELL* cell,FormKey& key) {
            key=KeyOf(cell);
            return !key.plugin.empty() && ValidUtf8(key.plugin) && (!legacyForms() || (IsSkyrimEsm(key) && cell->GetFormID()<0x1000000));
        }

        std::vector<RE::TESObjectCELL*> AttachedCells() {
            std::vector<RE::TESObjectCELL*> cells;
            auto* tes=RE::TES::GetSingleton();
            if(!tes) return cells;
            if(tes->interiorCell && tes->interiorCell->IsAttached()) cells.push_back(tes->interiorCell);
            if(auto* grid=tes->gridCells)
                for(uint32_t x=0;x<grid->length;++x) for(uint32_t y=0;y<grid->length;++y)
                    if(auto* cell=grid->GetCell(x,y);cell && cell->IsAttached()) cells.push_back(cell);
            return cells;
        }
        void Enqueue(Job&& job) {
            queue.push_back(std::move(job));
            queueReady.notify_one();
        }

        // Load doors near the player, or under the crosshair, whose destination is an unattached interior.
        // The engine does not load such a cell before the player enters unless bPreloadLinkedInteriors:General is set
        // (SE 1.5.97: Setting at RVA 1DB5A18, default 0; when set, a loading cell queues a CellLoaderTask for each
        // door's linked interior, RVA 268E50/273810), or the cell is still in the interior cell buffer. So its
        // reference set is usually empty and there is nothing to gather. When the references are there, they are
        // gathered without 3D once two polls agree on their number (a loading cell's set grows), and the worker reads
        // their meshes ahead of entry and bakes the cell if its room groups were observed in an earlier session.
        void PrebakeDoors(const std::vector<RE::TESObjectCELL*>& cells,const RE::NiPoint3& at) {
            std::vector<RE::NiPointer<RE::TESObjectREFR>> doors;
            std::set<RE::TESObjectREFR*> seenDoors;
            const auto addDoor=[&](RE::NiPointer<RE::TESObjectREFR> door) {
                auto* base=door ? door->GetBaseObject() : nullptr;
                if(base && base->Is(RE::FormType::Door) && seenDoors.insert(door.get()).second) doors.push_back(std::move(door));
            };
            if(auto* pick=RE::CrosshairPickData::GetSingleton()) addDoor(pick->GetActiveTarget().get());
            for(auto* cell:cells) cell->ForEachReference([&](RE::TESObjectREFR* ref) {
                if(ref && ref->GetPosition().GetDistance(at)<=settings.doorDistance) addDoor(RE::NiPointer<RE::TESObjectREFR>(ref));
                return RE::BSContainer::ForEachResult::kContinue;
            });
            std::set<uint32_t> destinations;
            for(const auto& door:doors) {
                auto* teleport=door->extraList.GetByType<RE::ExtraTeleport>();
                auto* data=teleport ? teleport->teleportData : nullptr;
                const auto linked=data ? data->linkedDoor.get() : RE::NiPointer<RE::TESObjectREFR>();
                auto* cell=linked ? linked->GetParentCell() : nullptr;
                FormKey key;
                if(!cell || !cell->IsInteriorCell() || cell->IsAttached() || !BakeableCell(cell,key)) continue;
                const auto id=cell->GetFormID();
                if(!destinations.insert(id).second) continue;
                {
                    std::lock_guard lock(queueMutex);
                    if(queuedOrDone.contains(id) || prebaked.contains(id)) continue;
                }
                Job job; job.cell=id; job.cellKey=key; job.editorID=EditorID(cell->GetFormEditorID()); job.interior=true; job.prebake=true;
                job.origin={data->position.x,data->position.y,data->position.z};  // where this door puts the player
                Lights lights;
                cell->ForEachReference([&](RE::TESObjectREFR* ref) {
                    LightOf(ref,lights);
                    Reference r;
                    if(Eligible(ref,true,r)) job.references.push_back(std::move(r));
                    return RE::BSContainer::ForEachResult::kContinue;
                });
                job.lights=SortedLights(lights);
                job.keyLights=job.lights;
                auto& previous=destinationSizes[id];
                const bool stable=job.references.size()>=2 && previous==job.references.size();
                previous=job.references.size();
                if(!stable) continue;
                destinationSizes.erase(id);
                std::lock_guard lock(queueMutex);
                prebaked.insert(id);
                Enqueue(std::move(job));
            }
        }

        // SKSE task: copy everything the worker needs out of the game.
        std::mutex pollMutex;  // SKSE tasks run on several game threads; one poll at a time

        void Poll() {
            std::unique_lock busy(pollMutex,std::try_to_lock);
            if(!busy) return;
            auto* ui=RE::UI::GetSingleton();
            auto* player=RE::PlayerCharacter::GetSingleton();
            if(!ui || !player || ui->IsMenuOpen("Main Menu") || ui->IsMenuOpen("Loading Menu")) return;
            const auto at=player->GetPosition();
            const auto cells=AttachedCells();
            if(settings.prebakeDoors) PrebakeDoors(cells,at);
            std::vector<RE::TESObjectCELL*> pending;
            {
                std::lock_guard lock(queueMutex);
                for(auto* cell:cells) if(!queuedOrDone.contains(cell->GetFormID())) pending.push_back(cell);
            }
            if(pending.empty()) return;
            // Exteriors: the lights of each attached exterior cell, by grid position.
            std::map<std::pair<int32_t,int32_t>,Lights> exteriorLights;
            if(std::any_of(pending.begin(),pending.end(),[](auto* cell) { return !cell->IsInteriorCell(); }))
                for(auto* cell:cells) {
                    auto* coordinates=cell->IsInteriorCell() ? nullptr : cell->GetCoordinates();
                    if(!coordinates) continue;
                    auto& lights=exteriorLights[{coordinates->cellX,coordinates->cellY}];
                    cell->ForEachReference([&](RE::TESObjectREFR* ref) { LightOf(ref,lights); return RE::BSContainer::ForEachResult::kContinue; });
                }
            for(auto* cell:pending) {
                const auto id=cell->GetFormID();
                Job job; job.cell=id; job.editorID=EditorID(cell->GetFormEditorID()); job.interior=cell->IsInteriorCell();
                if(!BakeableCell(cell,job.cellKey)) {
                    std::lock_guard lock(queueMutex);
                    queuedOrDone.insert(id);
                    continue;
                }
                job.origin={at.x,at.y,at.z};
                if(!job.interior) {
                    auto* coordinates=cell->GetCoordinates();
                    auto* world=cell->GetRuntimeData().worldSpace;
                    if(!coordinates || !world) continue;
                    job.worldspaceKey=KeyOf(world);
                    if(job.worldspaceKey.plugin.empty() || !ValidUtf8(job.worldspaceKey.plugin) || (legacyForms() && world->GetFormID()>=0x1000000)) continue;
                    job.grid={coordinates->cellX,coordinates->cellY}; job.worldspace=world->GetFormID(); job.worldspaceEditorID=EditorID(world->GetFormEditorID());
                    // The lights of this cell and of its attached neighbours that can reach its chunks (LightReachesCell). A
                    // cell gathered at the loaded grid's edge is baked at once and counts only the neighbours loaded then:
                    // waiting until all eight are attached left about 40% of the loaded cells along a road unmerged
                    // (the draw calls matter more than a darker merged mesh at a cell border, never seen in
                    // game). The cache key covers only the cell's own lights (keyLights), so the approach direction does not
                    // rebake it. A light of a cell two away is left out even when its radius reaches.
                    Lights reach,own;
                    for(int32_t dx=-1;dx<=1;++dx) for(int32_t dy=-1;dy<=1;++dy)
                        if(const auto found=exteriorLights.find({job.grid[0]+dx,job.grid[1]+dy});found!=exteriorLights.end())
                            for(const auto& [form,sphere]:found->second) if(Exterior::LightReachesCell(sphere,job.grid[0],job.grid[1])) {
                                reach.emplace(form,sphere);
                                if(dx==0 && dy==0) own.emplace(form,sphere);
                            }
                    job.lights=SortedLights(reach);
                    job.keyLights=SortedLights(own);
                    // Nearest cell first: distance from the player to the cell's centre when it was gathered.
                    job.distance=std::hypot((job.grid[0]+0.5)*4096.0-at.x,(job.grid[1]+0.5)*4096.0-at.y);
                }
                uint32_t missing3D=0;
                rejected.clear();
                Lights cellLights;
                cell->ForEachReference([&](RE::TESObjectREFR* ref) {
                    if(job.interior) LightOf(ref,cellLights);
                    Reference r;
                    if(!Eligible(ref,job.interior,r)) {
                        Reference o;
                        if(!job.interior && settings.previs && settings.previsExterior && Occluder(ref,o)) job.occluders.push_back(std::move(o));
                        return RE::BSContainer::ForEachResult::kContinue;
                    }
                    auto* root=ref->Get3D();
                    if(!root) { ++missing3D; return RE::BSContainer::ForEachResult::kContinue; }
                    if(RuntimeMaterial(root)) { ++job.skippedRuntimeMaterial; return RE::BSContainer::ForEachResult::kContinue; }
                    if(job.interior) r.roomGroup=RoomGroup(root);
                    AddEnableParentGroup(r);
                    job.references.push_back(std::move(r));
                    return RE::BSContainer::ForEachResult::kContinue;
                });
                if(job.interior) job.keyLights=job.lights=SortedLights(cellLights);
                // Wait until the cell's reference set stops growing (two polls in a row with the same count, at most
                // kMaxPollsSettling polls), then for its 3D to finish loading; after that, bake what is there.
                {
                    auto& [count,stable]=settling[id];
                    const size_t total=job.references.size()+job.skippedRuntimeMaterial+missing3D;
                    if(total!=count) { count=total; stable=0; if(++waitingFor3D[id]<kMaxPollsSettling) continue; }
                    else if(++stable<2 && ++waitingFor3D[id]<kMaxPollsSettling) continue;
                }
                if(missing3D && ++waitingFor3D[id]<kMaxPollsWaitingFor3D+kMaxPollsSettling) continue;
                job.skippedWithout3D=missing3D;
                waitingFor3D.erase(id); settling.erase(id);
                if(settings.previs && (job.interior || settings.previsExterior)) GatherNavmesh(cell,job);
                if(settings.previs && settings.previsExterior && !job.interior) GatherTerrain(cell,job);
                std::lock_guard lock(queueMutex);
                queuedOrDone.insert(id);
                {
                    std::string reasons;
                    for(const auto& [reason,count]:rejected) reasons+=std::format("{}{} {}",reasons.empty() ? "" : ", ",count,reason);
                    logger::info("Runtime bake: cell {} gathered {} eligible references ({} without 3D); left out: {}",KeyText(job.cellKey),
                        job.references.size(),job.skippedWithout3D,reasons.empty() ? "none" : reasons);
                }
                if(job.references.size()<2) {
                    if(!job.interior && settings.previs && settings.previsExterior && !settings.previsTool.empty()) {
                        job.previsOnly=true;
                        Enqueue(std::move(job));
                    }
                    continue;
                }
                Enqueue(std::move(job));
            }
        }

        bool ReadMesh(const std::string& model,std::vector<char>& bytes) {
            RE::BSResourceNiBinaryStream stream("meshes/"+model);
            if(!stream.good() || !stream.stream) return false;
            const uint32_t size=stream.stream->totalSize;
            if(!size || size>64u*1024*1024) return false;
            bytes.resize(size);
            return stream.read(bytes.data(),size);
        }

        // The model's bytes as the game loads them, stored once per session under their hash.
        const Asset& StoreMesh(const std::string& model) {
            auto [it,inserted]=assetStore.try_emplace(model);
            if(!inserted) return it->second;
            std::vector<char> bytes;
            if(!ReadMesh(model,bytes)) return it->second;
            Asset asset; asset.hash=HashHex(bytes.data(),bytes.size()); asset.bytes=bytes.size();
            asset.file=fs::absolute(AssetDir()/(asset.hash+".nif"));
            std::error_code ec;
            if(fs::file_size(asset.file,ec)!=bytes.size() || ec) {
                std::ofstream out(asset.file,std::ios::binary|std::ios::trunc);
                out.write(bytes.data(),static_cast<std::streamsize>(bytes.size()));
                if(!out) return it->second;
            }
            asset.ok=true;
            return it->second=std::move(asset);
        }

        // In-process bake: precombine::Bake on the calling worker thread. C++ exceptions come back as errors; a fault (a
        // mesh the baker cannot handle) is caught by structured exception handling, so it fails that part and the cell
        // stays original instead of ending the game. BakeSeh holds no objects that need unwinding.
        int BakeGuarded(const json* scene,const fs::path* output,json* result,std::string* message) {
            try { *result=precombine::Bake(*scene,*output); return 0; }
            catch(const std::exception& e) { *message=e.what(); return 1; }
            catch(...) { *message="unknown error"; return 1; }
        }
        // Only the faults a malformed mesh can raise in the baker are handled; anything else (a breakpoint, a
        // fail-fast, another module's exception) goes on to the next handler as if the guard were not there.
        struct Fault { DWORD code=0; const void* address=nullptr; };
        int BakeFilter(const EXCEPTION_POINTERS* info,Fault* fault) {
            switch(const DWORD code=info->ExceptionRecord->ExceptionCode) {
            case EXCEPTION_ACCESS_VIOLATION: case EXCEPTION_IN_PAGE_ERROR: case EXCEPTION_INT_DIVIDE_BY_ZERO:
            case EXCEPTION_ARRAY_BOUNDS_EXCEEDED: case EXCEPTION_STACK_OVERFLOW:
                fault->code=code; fault->address=info->ExceptionRecord->ExceptionAddress;
                return EXCEPTION_EXECUTE_HANDLER;
            default: return EXCEPTION_CONTINUE_SEARCH;
            }
        }
        int BakeSeh(const json* scene,const fs::path* output,json* result,std::string* message,Fault* fault) {
            __try { return BakeGuarded(scene,output,result,message); }
            __except(BakeFilter(GetExceptionInformation(),fault)) {
                // The stack has unwound to this frame: put back the guard page an overflow used up, or the next overflow
                // on this thread would end the process without any handler.
                if(fault->code==EXCEPTION_STACK_OVERFLOW) _resetstkoflw();
                return -1;
            }
        }
        // The first fault of the in-process baker this session (worker thread only). After it nothing more is baked
        // in-process until the game restarts: the fault may have left the baker's heap inconsistent.
        Fault bakerFault;
        std::string FaultText(const Fault& fault) {
            const char* name=fault.code==EXCEPTION_ACCESS_VIOLATION ? "access violation" : fault.code==EXCEPTION_IN_PAGE_ERROR ? "in-page error" :
                fault.code==EXCEPTION_INT_DIVIDE_BY_ZERO ? "integer division by zero" : fault.code==EXCEPTION_ARRAY_BOUNDS_EXCEEDED ? "array bounds exceeded" :
                fault.code==EXCEPTION_STACK_OVERFLOW ? "stack overflow" : "exception";
            HMODULE module=nullptr;
            wchar_t file[MAX_PATH]{};
            if(fault.address && GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                                   static_cast<LPCWSTR>(fault.address),&module) && GetModuleFileNameW(module,file,MAX_PATH))
                return std::format("{} {:08X} at {}+{:X}",name,fault.code,Utf8(fs::path(file).filename()),
                                   reinterpret_cast<uintptr_t>(fault.address)-reinterpret_cast<uintptr_t>(module));
            return std::format("{} {:08X} at {}",name,fault.code,fault.address);
        }
        // As PrecombineMesh.exe did: 0 with the generation report in `report`, or nonzero with "PrecombineMesh: <error>"
        // there (callers look for kNoMerges in it); -1 when it could not run, timed out or crashed. "runtime_bake_tool"
        // runs an external baker instead.
        int RunBaker(const fs::path& scene,const fs::path& output,const fs::path& report,std::string& error) {
            if(!settings.tool.empty()) return RunTool(settings.tool,scene,output,report,BELOW_NORMAL_PRIORITY_CLASS,settings.timeoutSeconds,error);
            json input,result;
            std::string message;
            int code;
            try { input=json::parse(ReadText(scene)); }
            catch(const std::exception& e) { error=std::string("unreadable scene: ")+e.what(); return -1; }
            Fault fault;
            code=BakeSeh(&input,&output,&result,&message,&fault);
            std::ofstream out(report,std::ios::binary);
            if(code==0) out<<result.dump(2)<<'\n';
            else if(code>0) { out<<"PrecombineMesh: "<<message<<'\n'; error="the baker: "+message; }
            else {
                // The scene is this frame's and the baker only reads it, so its model list is still sound to log.
                bakerFault=fault;
                std::string models;
                if(const auto assets=input.find("assets");assets!=input.end() && assets->is_object())
                    for(const auto& [model,asset]:assets->items()) models+=(models.empty() ? "" : ", ")+model;
                out<<"PrecombineMesh: "<<FaultText(fault)<<'\n';
                error=std::format("the baker faulted ({}); this part stays original and nothing more is baked in-process this session. Models: {}",
                    FaultText(fault),models);
            }
            return code;
        }
        int RunTool(const fs::path& tool,const fs::path& input,const fs::path& output,const fs::path& report,DWORD priority,
                    uint32_t timeoutSeconds,std::string& error) {
            std::wstring command=L"\""+tool.wstring()+L"\" \""+input.wstring()+L"\" \""+output.wstring()+L"\"";
            SECURITY_ATTRIBUTES inherit{sizeof(inherit),nullptr,TRUE};
            HANDLE out=CreateFileW(report.wstring().c_str(),GENERIC_WRITE,FILE_SHARE_READ,&inherit,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
            if(out==INVALID_HANDLE_VALUE) { error="cannot create the report file"; return -1; }
            STARTUPINFOW startup{}; startup.cb=sizeof(startup); startup.dwFlags=STARTF_USESTDHANDLES;
            startup.hStdOutput=out; startup.hStdError=out; startup.hStdInput=nullptr;
            PROCESS_INFORMATION process{};
            const BOOL ok=CreateProcessW(tool.wstring().c_str(),command.data(),nullptr,nullptr,TRUE,
                                         CREATE_NO_WINDOW|priority,nullptr,nullptr,&startup,&process);
            CloseHandle(out);
            if(!ok) { error=std::format("cannot start {} (error {})",Utf8(tool),GetLastError()); return -1; }
            const DWORD waited=WaitForSingleObject(process.hProcess,timeoutSeconds*1000);
            DWORD code=1;
            if(waited==WAIT_TIMEOUT) { TerminateProcess(process.hProcess,2); error=Utf8(tool.filename())+" timed out"; }
            else GetExitCodeProcess(process.hProcess,&code);
            CloseHandle(process.hThread); CloseHandle(process.hProcess);
            if(waited==WAIT_TIMEOUT) return -1;
            if(code!=0) error=std::format("{} exited with code {}",Utf8(tool.filename()),code);
            return static_cast<int>(code);
        }

        std::string ReadText(const fs::path& path) {
            std::ifstream in(path,std::ios::binary);
            return std::string(std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>());
        }
        // Readers see either the old or the new file, never a partial one.
        void WriteAtomically(const fs::path& path,const std::string& text) {
            auto temp=path;
            temp+=L".tmp";  // not through path::string(), which throws for characters outside the ANSI code page
            {
                std::ofstream out(temp,std::ios::binary|std::ios::trunc);
                out<<text;
                if(!out) throw std::runtime_error("cannot write "+Utf8(temp));
            }
            fs::rename(temp,path);
        }

        // index.json: the worker and the interior previs thread each rewrite only their cell's entry, from a fresh read,
        // under indexMutex (a reader holding the file open would also make the replacing rename fail).
        std::mutex indexMutex;
        json ReadIndexUnlocked() {
            std::ifstream in(WorkDir()/"index.json");
            if(!in) return json::object();
            try { return json::parse(in); } catch(...) { return json::object(); }
        }
        json ReadIndex() { std::lock_guard lock(indexMutex); return ReadIndexUnlocked(); }
        void WriteIndexEntry(const std::string& cellKey,const json& entry) {
            std::lock_guard lock(indexMutex);
            auto index=ReadIndexUnlocked();
            index[cellKey]=entry;
            WriteAtomically(WorkDir()/"index.json",index.dump(1));
        }

        // Room groups observed when the player was in the cell ("<plugin>|<local>" -> group), for a later door pre-bake.
        std::map<std::string,std::string> ReadRooms(const std::string& cellKey) {
            std::map<std::string,std::string> rooms;
            try {
                std::ifstream in(RoomsDir()/(cellKey+".json"));
                if(!in) return rooms;
                for(const auto& [ref,group]:json::parse(in).at("references").items()) rooms.emplace(ref,group.get<std::string>());
            } catch(...) { rooms.clear(); }
            return rooms;
        }
        void WriteRooms(const std::string& cellKey,const Job& job) {
            json references=json::object();
            for(const auto& r:job.references) references[KeyText(r.refKey)]=r.roomGroup;
            WriteAtomically(RoomsDir()/(cellKey+".json"),json{{"schema",2},{"cell",KeyText(job.cellKey)},{"references",references}}.dump());
        }

        // The previous bake's files: the merged manifest first (the loader reads manifests, and a manifest never names
        // a missing mesh), then each part manifest, then its meshes.
        void RemoveOutput(const std::string& cellKey,const json& previous) {
            std::error_code ec;
            if(previous.value("merged",false)) fs::remove(fs::path(kManifestDir)/(cellKey+".json"),ec);
            if(previous.contains("previs")) fs::remove(PrevisDir()/previous.at("previs").get<std::string>(),ec);
            // Exterior previs tables of the old bake ("<cell key>-<bake>-from-<camera cell>.pvs.json").
            if(previous.contains("bake")) {
                const auto prefix=std::format("{}-{}-from-",cellKey,previous.at("bake").get<std::string>());
                std::vector<fs::path> stale;
                for(const auto& file:fs::directory_iterator(PrevisDir(),ec)) if(Utf8(file.path().filename()).starts_with(prefix)) stale.push_back(file.path());
                for(const auto& path:stale) fs::remove(path,ec);
            }
            for(const auto& name:previous.value("parts",json::array())) {
                const auto path=PartsDir()/name.get<std::string>();
                std::vector<std::string> meshes;
                try { for(const auto& chunk:json::parse(ReadText(path)).at("chunks")) meshes.push_back(chunk.at("mesh")); } catch(...) {}
                fs::remove(path,ec);
                for(const auto& mesh:meshes) fs::remove(MeshRoot()/mesh,ec);
            }
        }

        // The cell field of a manifest, read from its start (the baker writes keys in sorted order, so "cell" comes
        // before "chunks"); the whole file is parsed only when that fails.
        json ManifestCell(const fs::path& path) {
            std::ifstream in(path,std::ios::binary);
            std::string head(65536,'\0');
            in.read(head.data(),static_cast<std::streamsize>(head.size()));
            head.resize(static_cast<size_t>(in.gcount()));
            if(const auto at=head.find("\"cell\"");at!=std::string::npos) {
                auto begin=head.find(':',at);
                if(begin!=std::string::npos) {
                    begin=head.find_first_not_of(" \t\r\n",begin+1);
                    const auto end=begin==std::string::npos ? begin : head[begin]=='{' ? head.find('}',begin)+1 : head.find_first_of(",}\r\n",begin);
                    if(end!=std::string::npos && end>begin) { try { return json::parse(head.substr(begin,end-begin)); } catch(...) {} }
                }
            }
            return json::parse(ReadText(path)).at("cell");
        }
        uint32_t ResolveForm(const json& value) {
            if(value.is_number_integer()) return value.get<uint32_t>();
            auto* handler=RE::TESDataHandler::GetSingleton();
            const auto plugin=value.at("plugin").get<std::string>();
            return handler ? handler->LookupFormID(value.at("id").get<RE::FormID>(),plugin) : 0;
        }
        // Cells whose manifest the release package (or another tool) ships: never baked or mixed with parts.
        void FindShippedCells() {
            const auto index=ReadIndex();
            std::set<std::string> merged;
            for(const auto& [key,entry]:index.items()) if(entry.is_object() && entry.value("merged",false)) merged.insert(key+".json");
            std::error_code ec;
            for(const auto& entry:fs::directory_iterator(kManifestDir,ec)) {
                const auto name=Lower(Utf8(entry.path().filename()));
                if(!entry.is_regular_file(ec) || !name.ends_with(".json") || merged.contains(name)) continue;
                try { if(const auto cell=ResolveForm(ManifestCell(entry.path()))) shippedCells.insert(cell); }
                catch(const std::exception& e) { logger::warn("Runtime bake: cannot read the cell of {}: {}",Utf8(entry.path()),e.what()); }
            }
        }

        // Community Shaders' Light Limit Fix shades point lights through clustered lists instead of each geometry's
        // first seven, so the light budget only adds chunks. Active when CommunityShaders.dll is loaded, its feature
        // file Data/Shaders/Features/LightLimitFix.ini exists (CS loads a feature only then), and "Disable at Boot"
        // in CommunityShaders/SettingsUser.json (else SettingsDefault.json) does not list "LightLimitFix"; or when
        // the standalone LightLimitFix.dll that preceded Community Shaders is loaded.
        std::string LightLimitFix() {
            if(GetModuleHandleW(L"LightLimitFix.dll")) return "LightLimitFix.dll";
            std::error_code ec;
            if(!GetModuleHandleW(L"CommunityShaders.dll") || !fs::exists("Data/Shaders/Features/LightLimitFix.ini",ec)) return {};
            for(const auto* file:{"Data/SKSE/Plugins/CommunityShaders/SettingsUser.json","Data/SKSE/Plugins/CommunityShaders/SettingsDefault.json"}) {
                try {
                    std::ifstream in(file);
                    if(!in) continue;
                    const auto settingsFile=json::parse(in,nullptr,true,true);
                    const auto boot=settingsFile.find("Disable at Boot");
                    if(boot==settingsFile.end() || !boot->is_object() || !boot->contains("LightLimitFix")) continue;
                    if(boot->at("LightLimitFix").get<bool>()) return {};
                    break;
                } catch(...) {}
            }
            return "Community Shaders Light Limit Fix";
        }

        json SceneHeader(const Job& job,const std::string& key) {
            // The baker names its output by "cell" (a number) and checks worldspace IDs below 0x1000000, so the scene
            // carries this load order's cell ID and the worldspace's local ID; each part then names both by plugin.
            json header={{"plugin","Skyrim.esm"},{"cell",job.cell},{"editor_id",job.editorID},{"chunk_size",settings.chunkSize},
                         {"source","runtime_bake"},{"cache_key",key},{"master_sha256","runtime"},{"record_skips",json::object()}};
            if(job.interior) { header["schema"]=1; header["interior"]=true; }
            else {
                header["schema"]=2; header["interior"]=false;
                header["worldspace"]={{"form_id",job.worldspaceKey.local},{"editor_id",job.worldspaceEditorID}};
                header["grid"]={job.grid[0],job.grid[1]};
            }
            return header;
        }
        // The forms of a published manifest, by plugin and local ID (or by number for old readers).
        void NameForms(json& manifest,const Job& job) {
            manifest["cell"]=FormJson(job.cellKey);
            if(!job.interior) manifest["worldspace"]["form_id"]=FormJson(job.worldspaceKey);
        }

        // ---- Previs (setting "runtime_previs"; interiors) ----
        // The live navmeshes give the view cells (grid boxes the camera can occupy), as an offline tool does
        // from the plugin records: every 256-unit box a triangle sample touches, from the floor up to eye height. After the
        // cell's last part, VisibilityEstimate renders cube maps from nine points in each view cell, on the GPU at idle
        // priority, and keeps the chunks a point sees (depth_probe), every chunk within 1,024 units, and every chunk inside
        // a light that reaches the cell or a visible chunk. Occluders are the cell's eligible references only, so the table
        // can only keep more than a full bake would.
        static constexpr double kViewCellSize=256.0, kEye=192.0, kNavSample=64.0;
        void GatherNavmesh(RE::TESObjectCELL* cell,Job& job) {
            auto* navs=cell->GetRuntimeData().navMeshes;
            if(!navs) return;
            for(const auto& nav:navs->navMeshes) {
                if(!nav) continue;
                const auto& vertices=nav->vertices;
                for(const auto& triangle:nav->triangles) {
                    std::array<float,9> t{};
                    bool valid=true;
                    for(int k=0;k<3 && valid;++k) {
                        const auto v=triangle.vertices[k];
                        valid=v<vertices.size();
                        if(valid) { const auto& p=vertices[v].location; t[3*k]=p.x; t[3*k+1]=p.y; t[3*k+2]=p.z; }
                    }
                    if(valid) job.navTriangles.push_back(t);
                }
            }
        }
        static std::vector<std::array<int64_t,3>> ViewCells(const std::vector<std::array<float,9>>& triangles,double size=kViewCellSize) {
            std::set<std::array<int64_t,3>> cells;
            const auto distance=[](const double* a,const double* b) { return std::sqrt((a[0]-b[0])*(a[0]-b[0])+(a[1]-b[1])*(a[1]-b[1])+(a[2]-b[2])*(a[2]-b[2])); };
            for(const auto& t:triangles) {
                const double a[3]={t[0],t[1],t[2]}, b[3]={t[3],t[4],t[5]}, c[3]={t[6],t[7],t[8]};
                const double edge=std::max({distance(a,b),distance(b,c),distance(c,a)});
                if(!std::isfinite(edge) || edge>65536.0) continue;
                const int steps=std::max(1,static_cast<int>(std::ceil(edge/kNavSample)));
                for(int i=0;i<=steps;++i) for(int j=0;j<=steps-i;++j) {
                    const double u=double(i)/steps, w=double(j)/steps;
                    double p[3];
                    for(int k=0;k<3;++k) p[k]=a[k]+u*(b[k]-a[k])+w*(c[k]-a[k]);
                    const auto ix=static_cast<int64_t>(std::floor(p[0]/size)), iy=static_cast<int64_t>(std::floor(p[1]/size));
                    const auto top=static_cast<int64_t>(std::floor((p[2]+kEye)/size));
                    for(auto iz=static_cast<int64_t>(std::floor(p[2]/size));iz<=top;++iz) cells.insert({ix,iy,iz});
                }
            }
            return {cells.begin(),cells.end()};
        }
        // Per view cell, plausible camera positions as fractions of the cell: navmesh samples every `step` units at camera
        // heights 64, 120.5 and 184, at most one per `box`-unit box and `limit` per cell.
        // A 512-unit exterior view cell's fixed corner points stand up
        // to 435 units above the ground and see over every roof and hill (nothing hidden at Whiterun's marker).
        static std::map<std::array<int64_t,3>,std::vector<std::array<double,3>>> CameraPoints(const std::vector<std::array<float,9>>& triangles,
                                                                                               double size,double step=32.0,double box=128.0,size_t limit=8) {
            std::map<std::array<int64_t,3>,std::map<std::array<int64_t,3>,std::array<double,3>>> found;
            const auto distance=[](const double* a,const double* b) { return std::sqrt((a[0]-b[0])*(a[0]-b[0])+(a[1]-b[1])*(a[1]-b[1])+(a[2]-b[2])*(a[2]-b[2])); };
            for(const auto& t:triangles) {
                const double a[3]={t[0],t[1],t[2]}, b[3]={t[3],t[4],t[5]}, c[3]={t[6],t[7],t[8]};
                const double edge=std::max({distance(a,b),distance(b,c),distance(c,a)});
                if(!std::isfinite(edge) || edge>65536.0) continue;
                const int steps=std::max(1,static_cast<int>(std::ceil(edge/step)));
                for(int i=0;i<=steps;++i) for(int j=0;j<=steps-i;++j) {
                    const double u=double(i)/steps, w=double(j)/steps;
                    double p[3];
                    for(int k=0;k<3;++k) p[k]=a[k]+u*(b[k]-a[k])+w*(c[k]-a[k]);
                    for(const double h:{64.0,120.5,184.0}) {
                        const double z=p[2]+h;
                        const std::array<int64_t,3> key{static_cast<int64_t>(std::floor(p[0]/size)),static_cast<int64_t>(std::floor(p[1]/size)),
                                                        static_cast<int64_t>(std::floor(z/size))};
                        auto& bucket=found[key];
                        const std::array<int64_t,3> slot{static_cast<int64_t>(std::floor(p[0]/box)),static_cast<int64_t>(std::floor(p[1]/box)),
                                                         static_cast<int64_t>(std::floor(z/box))};
                        if(bucket.size()<limit && !bucket.contains(slot))
                            bucket.emplace(slot,std::array<double,3>{p[0]/size-double(key[0]),p[1]/size-double(key[1]),z/size-double(key[2])});
                    }
                }
            }
            std::map<std::array<int64_t,3>,std::vector<std::array<double,3>>> out;
            for(const auto& [key,bucket]:found) for(const auto& [slot,point]:bucket) out[key].push_back(point);
            return out;
        }
        // ---- Interior previs (setting "runtime_previs"), on a thread of its own ----
        // A render takes minutes (Ustengrav02: 10 on an idle GPU) and far longer while the game keeps the GPU busy
        // (a large mod list uncapped at 300 fps: over 46 minutes), so it does not run on the worker: the next
        // cell's chunks bake meanwhile (Ustengrav01's waited behind the render the whole time).
        struct InteriorPrevis {
            Job job;
            std::string cellKey, bake;
            json chunks;
            std::map<std::string,std::string> assets;  // model -> extracted file (assetStore is the worker's)
        };
        void BakePrevis(const InteriorPrevis& task) {
            const auto& job=task.job;
            const auto& cellKey=task.cellKey;
            const auto& chunks=task.chunks;
            const auto begin=Clock::now();
            const auto cells=ViewCells(job.navTriangles);
            if(cells.empty()) { logger::info("Runtime previs: cell {} has no navmesh view cells; no table",cellKey); return; }
            json assets=json::object(), instances=json::array();
            std::map<std::string,uint32_t> runtimeRef;  // a part's {id, plugin} source reference -> this load order's form ID
            for(const auto& r:job.references) {
                const auto found=task.assets.find(r.model);
                if(found==task.assets.end()) continue;
                assets[r.model]=found->second;
                instances.push_back({{"ref",r.ref},{"base_type","STAT"},{"model",r.model},{"position",r.position},{"rotation",r.rotation},{"scale",r.scale}});
                runtimeRef.emplace(FormJson(r.refKey).dump(),r.ref);
            }
            json grouped=json::array(), objects=json::array();
            for(const auto& chunk:chunks) {
                json sources=json::array();
                for(const auto& source:chunk.at("sources")) {
                    const auto found=runtimeRef.find(source.at("ref").dump());
                    if(found==runtimeRef.end()) throw std::runtime_error("a chunk source is not among the cell's references");
                    sources.push_back({{"ref",found->second},{"path",source.at("path")}});
                }
                grouped.push_back({{"file",Utf8(fs::absolute(MeshRoot()/chunk.at("mesh").get<std::string>()))},
                                   {"origin",chunk.at("origin")},{"sources",std::move(sources)}});
                objects.push_back(chunk.at("mesh"));
            }
            json lights=json::array();
            for(const auto& l:job.lights) if(l[3]+128.0>0) lights.push_back({l[0],l[1],l[2],l[3]+128.0});  // + the attach-node pad
            json points=json::array({json::array({0.5,0.5,0.5})});
            for(const double x:{0.15,0.85}) for(const double y:{0.15,0.85}) for(const double z:{0.15,0.85}) points.push_back({x,y,z});
            json viewCells=json::array();
            for(const auto& c:cells) viewCells.push_back({c[0],c[1],c[2]});
            json request={{"schema",1},{"threads",2},{"near",15.0},{"depth_epsilon",1e-4},{"assets",std::move(assets)},
                          {"instances",std::move(instances)},{"groupings",json::array({json{{"name","runtime"},{"chunks",std::move(grouped)}}})},
                          {"views",json::array()},{"frames",json::array()},
                          {"pvs",{{"view_cell_size",kViewCellSize},{"view_cells",std::move(viewCells)},{"points",std::move(points)},
                                  {"face_size",settings.previsFaceSize},{"margin",1024.0},{"rule_mask",36},{"shadow_lights",std::move(lights)}}}};
            if(!settings.previsGpu.empty()) { request["pvs"]["gpu"]=settings.previsGpu; request["pvs"]["gpu_duty"]=settings.previsGpuDuty; }
            if(settings.previsMethod=="portals") { request["pvs"]["method"]="portals"; request["threads"]=settings.previsThreads; }
            const auto work=WorkDir()/"work"/(cellKey+"-previs");
            std::error_code ec;
            fs::remove_all(work,ec); fs::create_directories(work);
            { std::ofstream out(work/"request.json",std::ios::binary); out<<request.dump(); }
            std::string error;
            const int code=RunTool(settings.previsTool,fs::absolute(work/"request.json"),fs::absolute(work/"output.json"),fs::absolute(work/"report.txt"),
                                   IDLE_PRIORITY_CLASS,settings.previsTimeoutSeconds,error);
            if(code!=0) throw std::runtime_error(error.empty() ? "VisibilityEstimate failed" : error);
            const auto output=json::parse(ReadText(work/"output.json"));
            const auto& grouping=output.at("groupings").at(0);
            // A chunk's visibility is the union of its matched sources: an unmatched or duplicated source would under-report.
            if(grouping.value("unmatched_sources",0)!=0 || grouping.value("duplicate_members",0)!=0 || grouping.value("count_mismatches",0)!=0 ||
               output.at("pvs").at("chunks").get<size_t>()!=chunks.size())
                throw std::runtime_error(std::format("grouping problems (unmatched {}, duplicates {}, count mismatches {})",
                    grouping.value("unmatched_sources",0),grouping.value("duplicate_members",0),grouping.value("count_mismatches",0)));
            const auto name=std::format("{}-{}.pvs.json",cellKey,task.bake);
            const auto seconds=std::chrono::duration<double>(Clock::now()-begin).count();
            json table={{"schema",1},{"kind","runtime_pvs"},{"cell",FormJson(job.cellKey)},{"cell_key",cellKey},{"bake",task.bake},
                        {"view_cell_size",kViewCellSize},{"objects",std::move(objects)},{"view_cells",output.at("pvs").at("view_cells")},
                        {"stats",output.at("pvs").at("stats")},{"seconds",seconds}};
            fs::create_directories(PrevisDir());
            WriteAtomically(PrevisDir()/name,table.dump());
            {
                std::lock_guard lock(indexMutex);
                auto index=ReadIndexUnlocked();
                if(index.contains(cellKey) && index[cellKey].value("bake","")==task.bake) {
                    index[cellKey]["previs"]=name;
                    WriteAtomically(WorkDir()/"index.json",index.dump(1));
                }
            }
            fs::remove_all(work,ec);
            const bool accepted=offerPrevis && offerPrevis(name.c_str());
            logger::info("Runtime previs: cell {} ({}): {} view cells, {} chunks in {:.1f} s ({}){}",cellKey,job.editorID,cells.size(),chunks.size(),
                seconds,(settings.previsMethod=="portals" ? std::string("portals, points rendered on ") : std::string())+
                    (settings.previsGpu.empty() ? "CPU" : "GPU "+settings.previsGpu),accepted ? ", handed to the precombine plugin" : "");
        }
        std::mutex previsMutex;
        std::condition_variable previsReady;
        std::deque<InteriorPrevis> previsArrivals;  // guarded by previsMutex
        // One VisibilityEstimate render at a time: the interior and exterior previs threads take turns on the GPU.
        std::mutex renderMutex;
        // Worker thread: a baked interior, copied with the files of its meshes.
        void OfferInteriorPrevis(const Job& job,const std::string& cellKey,const std::string& bake,json chunks) {
            InteriorPrevis task{job,cellKey,bake,std::move(chunks),{}};
            for(const auto& r:job.references)
                if(const auto found=assetStore.find(r.model);found!=assetStore.end() && found->second.ok) task.assets[r.model]=Utf8(found->second.file);
            { std::lock_guard lock(previsMutex); previsArrivals.push_back(std::move(task)); }
            previsReady.notify_one();
        }
        void InteriorPrevisWorker() {
            while(true) {
                InteriorPrevis task;
                {
                    std::unique_lock lock(previsMutex);
                    previsReady.wait(lock,[]{ return !previsArrivals.empty(); });
                    task=std::move(previsArrivals.front());
                    previsArrivals.pop_front();
                }
                try { std::lock_guard render(renderMutex); BakePrevis(task); }
                catch(const std::exception& e) { logger::error("Runtime previs: cell {} has no table: {}",task.cellKey,e.what()); }
            }
        }

        // ---- Exterior previs (settings "runtime_previs" and "runtime_previs_exterior") ----
        // Outdoors the chunks worth hiding are mostly other cells' (behind hills and buildings), so each chunk cell's
        // table is filled camera cell by camera cell. A camera cell's view cells (512 units, from its live navmesh, eye
        // height 192) are rendered against the chunks of every cell within two of it, with those cells' eligible
        // references and terrain (the loaded heightfields, lowered 8 units) as occluders. The bits are split per chunk
        // cell into <cell>-<bake>-from-<camera
        // cell>.pvs.json; the precombine plugin merges them into the chunk cell's table and hides nothing near a camera
        // cell no table covers. A thread of its own at idle priority, so the worker keeps baking chunks while the player
        // travels; always the cube-map renderer (a portal graph cannot cull outdoors: the open sky merges into huge cells).
        void GatherTerrain(RE::TESObjectCELL* cell,Job& job) {
            auto* land=cell->GetRuntimeData().cellLand;
            if(!land || !land->loadedData) return;
            // TESObjectLAND::GetNiPointHeight adds the float at LoadedLandData+0x49C0 (the land's base height).
            const float base=*reinterpret_cast<const float*>(reinterpret_cast<const std::byte*>(land->loadedData)+0x49C0);
            job.terrain=Exterior::StitchTerrain(land->loadedData->heights,base);
            // The first cells' corners and centre, compared offline with the heights from the records.
            static std::atomic<int> logged{0};
            if(logged.fetch_add(1)<4) {
                const auto& t=job.terrain;
                logger::info("Runtime exterior previs: terrain of cell {},{}: base {:.1f}; south-west {:.1f}, south-east {:.1f}, north-west {:.1f}, north-east {:.1f}, centre {:.1f}",
                    job.grid[0],job.grid[1],base,t[0],t[32],t[32*33],t[33*33-1],t[16*33+16]);
            }
        }
        struct ExteriorCell {
            std::string cellKey;
            FormKey key;
            uint32_t world=0;
            std::array<int32_t,2> grid{};
            std::vector<Reference> references, occluders;
            std::map<std::string,std::string> assets;  // model -> extracted file
            std::vector<float> terrain;
            std::vector<std::array<float,9>> navTriangles;
            std::vector<std::array<double,4>> lights;
            std::string bakeKey, bake;
            json chunks=json::array();
        };
        constexpr double kExteriorViewCell=512.0;
        constexpr auto kExteriorPartialWait=std::chrono::seconds(120);
        std::mutex exteriorMutex;
        std::condition_variable exteriorReady;
        std::vector<ExteriorCell> exteriorArrivals;  // guarded by exteriorMutex
        // Worker thread: a finished exterior cell, copied with the files of its meshes (assetStore is the worker's).
        void OfferExterior(const Job& job,const std::string& cellKey,const std::string& key,const std::string& bake,json chunks) {
            ExteriorCell cell;
            cell.cellKey=cellKey; cell.key=job.cellKey; cell.world=job.worldspace; cell.grid=job.grid;
            cell.references=job.references; cell.terrain=job.terrain; cell.navTriangles=job.navTriangles; cell.lights=job.lights;
            cell.bakeKey=key; cell.bake=bake; cell.chunks=std::move(chunks);
            cell.occluders=job.occluders;
            for(const auto* list:{&job.references,&job.occluders})
                for(const auto& r:*list)
                    if(const auto& asset=StoreMesh(r.model);asset.ok) cell.assets[r.model]=Utf8(asset.file);
            { std::lock_guard lock(exteriorMutex); exteriorArrivals.push_back(std::move(cell)); }
            exteriorReady.notify_one();
        }
        using ExteriorGrid=std::map<std::tuple<uint32_t,int32_t,int32_t>,ExteriorCell>;
        // Renders camera cell `a`'s view cells against the chunks of the cells within two of it, unless the same inputs
        // were rendered before (exterior-previs.json), and hands the tables to the precombine plugin.
        void BakeExteriorPrevis(const ExteriorGrid& cells,const ExteriorCell& a,json& signatures) {
            const auto begin=Clock::now();
            std::vector<const ExteriorCell*> region;
            for(int dy=-2;dy<=2;++dy) for(int dx=-2;dx<=2;++dx)
                if(const auto found=cells.find({a.world,a.grid[0]+dx,a.grid[1]+dy});found!=cells.end()) region.push_back(&found->second);
            // Shadow lights: every light that reaches a region cell. Each cell keeps only the lights that reach it,
            // so the camera cell's own list would miss lights of a visible chunk two cells away. Hashed into the
            // signature, since the bake key carries no lights without a light budget.
            std::vector<std::array<double,4>> shadowLights;
            for(const auto* r:region) shadowLights.insert(shadowLights.end(),r->lights.begin(),r->lights.end());
            Exterior::MergeLights(shadowLights);
            std::string text=std::format("camera-points|{}|{}|{}|{}|{}|",a.cellKey,a.navTriangles.size(),settings.previsFaceSize,settings.previsGpu,
                HashHex(shadowLights.data(),shadowLights.size()*sizeof(shadowLights[0])));
            for(const auto* r:region) text+=std::format("{}:{}:{}:{}:{}:{};",r->cellKey,r->bakeKey,r->bake,r->terrain.size(),r->references.size(),r->occluders.size());
            const auto signature=HashHex(text.data(),text.size());
            const auto before=signatures.value(a.cellKey,json::object());
            if(before.value("signature","")==signature) {
                bool all=true;
                for(const auto& name:before.value("tables",json::array())) all&=fs::exists(PrevisDir()/name.get<std::string>());
                if(all) {
                    for(const auto& name:before.value("tables",json::array())) offerPrevis(name.get<std::string>().c_str());
                    return;
                }
            }
            const auto viewCells=ViewCells(a.navTriangles,kExteriorViewCell);
            if(viewCells.empty()) return;
            json assets=json::object(), instances=json::array(), terrain=json::array();
            std::map<std::string,uint32_t> runtimeRef;  // a part's {id, plugin} source reference -> this load order's form ID
            for(const auto* r:region) {
                for(const auto& ref:r->references) {
                    const auto file=r->assets.find(ref.model);
                    if(file==r->assets.end()) continue;
                    assets[ref.model]=file->second;
                    instances.push_back({{"ref",ref.ref},{"base_type","STAT"},{"model",ref.model},{"position",ref.position},{"rotation",ref.rotation},
                                         {"scale",ref.scale}});
                    runtimeRef.emplace(FormJson(ref.refKey).dump(),ref.ref);
                }
                for(const auto& ref:r->occluders) {
                    const auto file=r->assets.find(ref.model);
                    if(file==r->assets.end()) continue;
                    assets[ref.model]=file->second;
                    instances.push_back({{"ref",ref.ref},{"base_type","STAT"},{"model",ref.model},{"position",ref.position},{"rotation",ref.rotation},
                                         {"scale",ref.scale}});
                }
                if(r->terrain.size()==size_t(Exterior::kSide)*Exterior::kSide) {
                    json heights=json::array();
                    for(const float h:r->terrain) heights.push_back(h-8.f);
                    terrain.push_back({{"land",0},{"origin",{r->grid[0]*Exterior::kCell,r->grid[1]*Exterior::kCell}},{"step",Exterior::kStep},
                                       {"side",Exterior::kSide},{"heights",std::move(heights)}});
                }
            }
            // Targets: each region cell's chunks, consecutive in one grouping. A chunk with a source not among the region's
            // references is left out of the table, so it is never hidden.
            struct Target { const ExteriorCell* cell; size_t start=0, count=0; json objects=json::array(); };
            std::vector<Target> targets;
            json grouped=json::array();
            for(const auto* r:region) {
                Target t{r,grouped.size()};
                for(const auto& chunk:r->chunks) {
                    json sources=json::array();
                    bool known=true;
                    for(const auto& source:chunk.at("sources")) {
                        const auto found=runtimeRef.find(source.at("ref").dump());
                        if(found==runtimeRef.end()) { known=false; break; }
                        sources.push_back({{"ref",found->second},{"path",source.at("path")}});
                    }
                    if(!known) continue;
                    grouped.push_back({{"file",Utf8(fs::absolute(MeshRoot()/chunk.at("mesh").get<std::string>()))},
                                       {"origin",chunk.at("origin")},{"sources",std::move(sources)}});
                    t.objects.push_back(chunk.at("mesh"));
                }
                t.count=grouped.size()-t.start;
                if(t.count) targets.push_back(std::move(t));
            }
            if(targets.empty()) return;
            const size_t total=grouped.size();
            json lights=json::array();
            for(const auto& l:shadowLights) if(l[3]+128.0>0) lights.push_back({l[0],l[1],l[2],l[3]+128.0});  // + the attach-node pad
            json points=json::array({json::array({0.5,0.5,0.5})});
            for(const double x:{0.15,0.85}) for(const double y:{0.15,0.85}) for(const double z:{0.15,0.85}) points.push_back({x,y,z});
            // Each view cell's own camera points where the navmesh gives some; the fixed points otherwise.
            const auto own=CameraPoints(a.navTriangles,kExteriorViewCell);
            json cellsJson=json::array();
            size_t ownCells=0;
            for(const auto& c:viewCells) {
                const auto found=own.find(c);
                if(found==own.end()) { cellsJson.push_back({c[0],c[1],c[2]}); continue; }
                json points=json::array();
                for(const auto& q:found->second) points.push_back({q[0],q[1],q[2]});
                cellsJson.push_back({c[0],c[1],c[2],std::move(points)});
                ++ownCells;
            }
            json request={{"schema",1},{"threads",2},{"near",15.0},{"depth_epsilon",1e-4},{"assets",std::move(assets)},
                          {"instances",std::move(instances)},{"terrain",std::move(terrain)},
                          {"groupings",json::array({json{{"name","runtime-exterior"},{"chunks",std::move(grouped)}}})},
                          {"views",json::array()},{"frames",json::array()},
                          {"pvs",{{"view_cell_size",kExteriorViewCell},{"view_cells",std::move(cellsJson)},{"points",std::move(points)},
                                  {"face_size",settings.previsFaceSize},{"margin",1024.0},{"rule_mask",36},{"shadow_lights",std::move(lights)}}}};
            if(!settings.previsGpu.empty()) { request["pvs"]["gpu"]=settings.previsGpu; request["pvs"]["gpu_duty"]=settings.previsGpuDuty; }
            const auto work=WorkDir()/"work"/(a.cellKey+"-exterior-previs");
            std::error_code ec;
            fs::remove_all(work,ec); fs::create_directories(work);
            { std::ofstream out(work/"request.json",std::ios::binary); out<<request.dump(); }
            std::string error;
            const int code=RunTool(settings.previsTool,fs::absolute(work/"request.json"),fs::absolute(work/"output.json"),fs::absolute(work/"report.txt"),
                                   IDLE_PRIORITY_CLASS,settings.previsTimeoutSeconds,error);
            if(code!=0) throw std::runtime_error(error.empty() ? "VisibilityEstimate failed" : error);
            const auto output=json::parse(ReadText(work/"output.json"));
            const auto& grouping=output.at("groupings").at(0);
            if(grouping.value("unmatched_sources",0)!=0 || grouping.value("duplicate_members",0)!=0 || grouping.value("count_mismatches",0)!=0 ||
               output.at("pvs").at("chunks").get<size_t>()!=total)
                throw std::runtime_error(std::format("grouping problems (unmatched {}, duplicates {}, count mismatches {})",
                    grouping.value("unmatched_sources",0),grouping.value("duplicate_members",0),grouping.value("count_mismatches",0)));
            const auto seconds=std::chrono::duration<double>(Clock::now()-begin).count();
            json names=json::array();
            fs::create_directories(PrevisDir());
            for(const auto& t:targets) {
                json split=json::array();
                for(const auto& v:output.at("pvs").at("view_cells"))
                    split.push_back({v.at(0),v.at(1),v.at(2),Exterior::SliceBits(v.at(3).get<std::string>(),t.start,t.count)});
                const auto name=std::format("{}-{}-from-{}.pvs.json",t.cell->cellKey,t.cell->bake,a.cellKey);
                json table={{"schema",1},{"kind","runtime_pvs_exterior"},{"cell",FormJson(t.cell->key)},{"cell_key",t.cell->cellKey},
                            {"bake",t.cell->bake},{"from_cell_key",a.cellKey},{"from_grid",{a.grid[0],a.grid[1]}},{"view_cell_size",kExteriorViewCell},
                            {"objects",t.objects},{"view_cells",std::move(split)},{"stats",output.at("pvs").at("stats")},{"seconds",seconds}};
                WriteAtomically(PrevisDir()/name,table.dump());
                names.push_back(name);
            }
            for(const auto& old:before.value("tables",json::array()))
                if(std::find(names.begin(),names.end(),old)==names.end()) fs::remove(PrevisDir()/old.get<std::string>(),ec);
            signatures[a.cellKey]={{"signature",signature},{"tables",names},{"region",region.size()},{"seconds",seconds}};
            WriteAtomically(WorkDir()/"exterior-previs.json",signatures.dump(1));
            fs::remove_all(work,ec);
            size_t offered=0;
            for(const auto& name:names) offered+=offerPrevis(name.get<std::string>().c_str());
            logger::info("Runtime exterior previs: camera cell {},{} ({}): {} view cells ({} with navmesh camera points) against {} chunks of {} of {} region cells, {} terrain cells, in {:.1f} s; {} tables, {} handed to the precombine plugin",
                a.grid[0],a.grid[1],a.cellKey,viewCells.size(),ownCells,total,targets.size(),region.size(),output.at("terrain").at("cells").get<size_t>(),
                seconds,names.size(),offered);
        }
        // A camera cell is rendered once the cells within two of it have arrived (their chunks, references and terrain), or
        // with what is there once its eight neighbours have been there for kExteriorPartialWait (cells beyond the loaded
        // grid may never arrive), and again when that region is complete.
        void ExteriorPrevisWorker() {
            ExteriorGrid cells;
            json signatures=json::object();
            try {
                const auto path=WorkDir()/"exterior-previs.json";
                if(fs::exists(path)) signatures=json::parse(ReadText(path));
            } catch(...) { signatures=json::object(); }
            using Key=std::tuple<uint32_t,int32_t,int32_t>;
            std::set<Key> pending, complete;
            std::map<Key,Clock::time_point> neighboursSince;
            while(true) {
                std::vector<ExteriorCell> arrivals;
                {
                    std::unique_lock lock(exteriorMutex);
                    exteriorReady.wait_for(lock,std::chrono::seconds(10),[]{ return !exteriorArrivals.empty(); });
                    arrivals.swap(exteriorArrivals);
                }
                for(auto& cell:arrivals) {
                    // A cell rebaked since it last arrived (its old bake's tables are gone): every camera cell within two
                    // of it is rendered again, including complete ones.
                    const auto known=cells.find(Key{cell.world,cell.grid[0],cell.grid[1]});
                    const bool rebaked=known!=cells.end() && known->second.bake!=cell.bake;
                    for(int dy=-2;dy<=2;++dy) for(int dx=-2;dx<=2;++dx) {
                        const Key around{cell.world,cell.grid[0]+dx,cell.grid[1]+dy};
                        pending.insert(around);
                        if(rebaked) complete.erase(around);
                    }
                    cells.insert_or_assign(Key{cell.world,cell.grid[0],cell.grid[1]},std::move(cell));
                }
                const auto now=Clock::now();
                for(auto it=pending.begin();it!=pending.end();) {
                    const auto [world,x,y]=*it;
                    const auto own=cells.find(*it);
                    const auto present=[&](int reach) {
                        for(int dy=-reach;dy<=reach;++dy) for(int dx=-reach;dx<=reach;++dx) if(!cells.contains({world,x+dx,y+dy})) return false;
                        return true;
                    };
                    if(own==cells.end() || complete.contains(*it) || !present(1)) { ++it; continue; }
                    const bool full=present(2);
                    if(!full) {
                        const auto since=neighboursSince.try_emplace(*it,now).first->second;
                        if(now-since<kExteriorPartialWait) { ++it; continue; }
                    }
                    try { if(!own->second.navTriangles.empty()) { std::lock_guard render(renderMutex); BakeExteriorPrevis(cells,own->second,signatures); } }
                    catch(const std::exception& e) { logger::error("Runtime exterior previs: camera cell {},{}: {}",x,y,e.what()); }
                    if(full) complete.insert(*it);
                    it=pending.erase(it);  // a partial bake comes back to pending when another region cell arrives
                }
            }
        }

        // The chunks of a complete earlier bake, read from its parts, or why they cannot all be used (empty: all there).
        // A part file must be there and of that bake, and every chunk mesh there at its recorded size: a cache cleaned
        // in part (meshes deleted, index kept) or cut short by a power loss after the index was marked complete otherwise
        // left the cell original in every later session. A file of the right size with wrong bytes is left
        // to the precombine plugin's hash check.
        std::string StoredChunks(const json& previous,json& chunks) {
            try {
                const auto bake=previous.at("bake").get<std::string>();
                for(const auto& entry:previous.value("parts",json::array())) {
                    const auto name=entry.get<std::string>();
                    const auto text=ReadText(PartsDir()/name);
                    if(text.empty()) return "part "+name+" is missing";
                    auto part=json::parse(text);
                    if(part.at("runtime_part").at("bake")!=bake) return "part "+name+" is of another bake";
                    for(auto& chunk:part.at("chunks")) {
                        const auto mesh=chunk.at("mesh").get<std::string>();
                        std::error_code ec;
                        const auto size=fs::file_size(MeshRoot()/mesh,ec);
                        if(ec) return "chunk mesh "+mesh+" is missing";
                        if(size!=chunk.at("mesh_bytes").get<uint64_t>()) return "chunk mesh "+mesh+" has another size";
                        chunks.push_back(std::move(chunk));
                    }
                }
                if(const auto listed=previous.value("chunks",chunks.size());listed!=chunks.size()) return std::format("its parts hold {} of its {} chunks",chunks.size(),listed);
                return {};
            } catch(const std::exception& e) { return e.what(); }
        }

        void Bake(Job& job) {
            const auto begin=Clock::now();
            const auto ms=[](auto a,auto b){ return std::chrono::duration_cast<std::chrono::milliseconds>(b-a).count(); };
            const auto cellKey=CellKey(job.cellKey);
            const auto manifestPath=fs::path(kManifestDir)/(cellKey+".json");
            const json previous=ReadIndex().value(cellKey,json::object());
            const bool exteriorPrevis=!job.interior && settings.previs && settings.previsExterior && !settings.previsTool.empty();
            if(job.previsOnly) {
                if(exteriorPrevis) OfferExterior(job,cellKey,"","",json::array());
                return;
            }
            // A manifest the release package ships is never replaced or mixed with parts.
            if(shippedCells.contains(job.cell)) {
                logger::info("Runtime bake: cell {} has a shipped manifest; skipped",cellKey);
                if(exteriorPrevis) OfferExterior(job,cellKey,"","",json::array());
                return;
            }
            // Meshes through the resource system: loose files and archives in the user's load order.
            std::set<std::string> models;
            for(const auto& r:job.references) models.insert(r.model);
            size_t read=0;
            std::vector<std::pair<std::string,std::string>> hashes;
            for(const auto& model:models) {
                read+=!assetStore.contains(model);
                if(const auto& asset=StoreMesh(model);asset.ok) hashes.emplace_back(model,asset.hash);
            }
            const auto readDone=Clock::now();
            if(job.prebake) {
                const auto rooms=ReadRooms(cellKey);
                if(rooms.empty()) {
                    logger::info("Runtime bake: door destination {} ({}) gathered before entry: {} references, {} meshes read ahead; room groups are unknown until the player enters",
                        cellKey,job.editorID,job.references.size(),read);
                    return;
                }
                // Groups observed in an earlier session; a reference not seen then merges only within its own root.
                for(auto& r:job.references) {
                    const auto found=rooms.find(KeyText(r.refKey));
                    if(found!=rooms.end()) r.roomGroup=found->second;
                    else { r.roomGroup="unobserved:"+KeyText(r.refKey); AddEnableParentGroup(r); }
                }
            } else if(job.interior) {
                std::error_code ec; fs::create_directories(RoomsDir(),ec);
                WriteRooms(cellKey,job);
            }
            const auto key=CacheKey(job,hashes,settings);
            const std::string mode=legacyForms() ? "merged" : "parts";
            // The same input faulted the in-process baker in an earlier session: it would fault again. A
            // changed reference, mesh or setting changes the key, and the cell is tried again.
            if(settings.tool.empty() && previous.value("faulted","")==key) {
                logger::info("Runtime bake: cell {} ({}) faulted the baker in an earlier session with the same references and meshes; its unbaked part stays original",
                    cellKey,job.editorID);
                if(exteriorPrevis) OfferExterior(job,cellKey,"","",json::array());
                return;
            }
            std::error_code ec;
            if(previous.value("key","")==key && previous.value("complete",false) && previous.value("mode","")==mode && (mode=="parts" || fs::exists(manifestPath,ec))) {
                json stored=json::array();
                const auto damaged=StoredChunks(previous,stored);
                if(damaged.empty()) {
                    logger::info("Runtime bake: cell {} unchanged since its last bake",cellKey);
                    // Its previs table may be missing: the setting was off then, or the renderer failed or timed out
                    // (a busy GPU took over 30 minutes). Render it now from the stored parts.
                    try {
                        if(mode=="parts" && settings.previs && job.interior && !settings.previsTool.empty() && !previous.contains("previs") && !stored.empty())
                            OfferInteriorPrevis(job,cellKey,previous.at("bake").get<std::string>(),std::move(stored));
                    } catch(const std::exception& e) { logger::error("Runtime previs: cell {} has no table: {}",cellKey,e.what()); }
                    try {
                        if(mode=="parts" && settings.previs && settings.previsExterior && !job.interior && !settings.previsTool.empty())
                            OfferExterior(job,cellKey,key,previous.at("bake").get<std::string>(),std::move(stored));
                    } catch(const std::exception& e) { logger::error("Runtime exterior previs: cell {} left out: {}",cellKey,e.what()); }
                    return;
                }
                logger::warn("Runtime bake: cell {} was baked before, but {}; baking it again",cellKey,damaged);
            }
            // After a fault the heap may be inconsistent, so nothing more is baked in-process this session.
            if(settings.tool.empty() && bakerFault.code) {
                static bool logged=false;  // worker thread only
                if(!std::exchange(logged,true))
                    logger::warn("Runtime bake: no more cells are baked this session after the baker's fault; cells baked earlier stay in use");
                return;
            }
            RemoveOutput(cellKey,previous);
            // A new name for every bake, so no path the precombine plugin may already have loaded is ever reused.
            const auto stamp=std::format("{}|{}|{}",key,cellKey,std::chrono::system_clock::now().time_since_epoch().count());
            const auto bake=HashHex(stamp.data(),stamp.size());
            json entry={{"key",key},{"bake",bake},{"baker",precombine::kOutputVersion},{"mode",mode},{"parts",json::array()},{"complete",false},
                       {"cell",KeyText(job.cellKey)},{"references",job.references.size()},{"skipped_without_3d",job.skippedWithout3D},
                       {"skipped_runtime_material",job.skippedRuntimeMaterial},{"light_budget",settings.lightBudget},
                       {"prebake",job.prebake},{"tool",settings.tool.empty() ? std::string("in-process") : Utf8(settings.tool)}};
            WriteIndexEntry(cellKey,entry);
            const auto parts=PlanParts(job,settings.firstPartReferences,settings.partReferences);
            const auto work=WorkDir()/"work"/cellKey;
            const auto producedName=std::format("{:08x}.json",job.cell);
            json mergedChunks=json::array(), previsChunks=json::array(), exteriorChunks=json::array();
            size_t chunks=0;
            long long firstPart=-1;
            for(size_t k=0;k<parts.size();++k) {
                const auto partBegin=Clock::now();
                fs::remove_all(work,ec); fs::create_directories(work);
                json references=json::array(), assets=json::object();
                for(const auto i:parts[k]) {
                    const auto& r=job.references[i];
                    const auto& asset=assetStore[r.model];
                    if(!asset.ok) continue;
                    json item={{"ref",FormJson(r.refKey)},{"base",FormJson(r.baseKey)},{"model",r.model},{"position",r.position},{"rotation",r.rotation},{"scale",r.scale}};
                    if(!r.roomGroup.empty()) item["room_group"]=r.roomGroup;
                    if(!r.enableParent.plugin.empty()) item["enable_parent"]={{"ref",FormJson(r.enableParent)},{"opposite",r.enableOpposite}};
                    references.push_back(std::move(item));
                    assets[r.model]={{"file",Utf8(asset.file)},{"bytes",asset.bytes},{"hash",asset.hash}};
                }
                auto scene=SceneHeader(job,key);
                scene["references"]=std::move(references); scene["assets"]=std::move(assets);
                if(settings.lightBudget>=0) { scene["light_budget"]=settings.lightBudget; scene["lights"]=job.lights; }
                { std::ofstream out(work/"scene.json",std::ios::binary); out<<scene.dump(); }
                std::string error;
                const int code=RunBaker(fs::absolute(work/"scene.json"),fs::absolute(work/"out"),fs::absolute(work/"report.json"),error);
                const auto produced=work/"out/SKSE/Plugins/Precombines"/producedName;
                json manifest;
                if(code==0 && fs::exists(produced)) manifest=json::parse(ReadText(produced));
                else if(code==0 || (code>0 && ReadText(work/"report.json").find(kNoMerges)!=std::string::npos)) {
                    manifest=SceneHeader(job,key); manifest["chunks"]=json::array();  // no merges among this part's groups
                } else {
                    // A fault is kept with this key in the index, so a later session does not bake the same input again. Not
                    // an in-page error: a page of this DLL or the page file that could not be read, not the scene's doing
                    // (the baker reads its assets through streams), so the next session tries the cell again.
                    if(code<0 && settings.tool.empty() && bakerFault.code && bakerFault.code!=EXCEPTION_IN_PAGE_ERROR) {
                        entry["faulted"]=key; entry["fault"]=std::format("{:08X}",bakerFault.code);
                        try { WriteIndexEntry(cellKey,entry); }
                        catch(const std::exception& e) { logger::error("Runtime bake: cell {}: the fault is not recorded: {}",cellKey,e.what()); }
                    }
                    throw std::runtime_error(std::format("part {} of {}: {}",k,parts.size(),error));
                }
                NameForms(manifest,job);
                size_t n=0;
                for(auto& chunk:manifest["chunks"]) {
                    const auto mesh=std::format("precombined/skyrim/rb/{}/{}_{}_{}.nif",cellKey,bake,k,n++);
                    const auto target=MeshRoot()/mesh;
                    fs::create_directories(target.parent_path());
                    fs::copy_file(work/"out/meshes"/chunk.at("mesh").get<std::string>(),target,fs::copy_options::overwrite_existing);
                    chunk["mesh"]=mesh;
                    if(mode=="merged") mergedChunks.push_back(chunk);
                    else if(settings.previs && job.interior) previsChunks.push_back(chunk);
                    else if(settings.previs && settings.previsExterior && !job.interior) exteriorChunks.push_back(chunk);
                }
                chunks+=n;
                manifest["runtime_part"]={{"cell_key",cellKey},{"bake",bake},{"key",key},{"index",k},{"parts",parts.size()},{"baker",precombine::kOutputVersion}};
                const auto name=std::format("{}-{}-{}.json",cellKey,bake,k);
                WriteAtomically(PartsDir()/name,manifest.dump(1));
                entry["parts"].push_back(name);
                WriteIndexEntry(cellKey,entry);
                const bool accepted=offerPart && offerPart(name.c_str());
                if(firstPart<0) firstPart=ms(begin,Clock::now());
                logger::info("Runtime bake: cell {} part {}/{}: {} references, {} chunks in {} ms{}",cellKey,k+1,parts.size(),parts[k].size(),n,
                    ms(partBegin,Clock::now()),accepted ? ", handed to the precombine plugin" : "");
            }
            if(mode=="merged" && chunks) {
                // Without the hand-off: one manifest for the next data load, as a release SkyrimPrecombines.dll reads it.
                auto merged=SceneHeader(job,key);
                NameForms(merged,job);
                merged["chunks"]=std::move(mergedChunks);
                WriteAtomically(manifestPath,merged.dump(1));
                entry["merged"]=true;
            }
            entry["complete"]=true; entry["chunks"]=chunks;
            WriteIndexEntry(cellKey,entry);
            fs::remove_all(work,ec);
            if(!previsChunks.empty() && !settings.previsTool.empty()) OfferInteriorPrevis(job,cellKey,bake,std::move(previsChunks));
            if(mode=="parts" && settings.previs && settings.previsExterior && !job.interior && !settings.previsTool.empty())
                OfferExterior(job,cellKey,key,bake,std::move(exteriorChunks));
            logger::info("Runtime bake: cell {} ({}) baked {} chunks from {} references in {} parts: first part after {} ms, all after {} ms (meshes {} ms, {} read; {} references left original for runtime material); {}",
                cellKey,job.editorID,chunks,job.references.size(),parts.size(),firstPart,ms(begin,Clock::now()),ms(begin,readDone),read,job.skippedRuntimeMaterial,
                mode=="parts" ? "parts activate during play" : "used from the next data load");
        }

        void Worker() {
            // The in-process baker runs here: priority 6, as PrecombineMesh.exe had in its below-normal priority class.
            if(settings.tool.empty()) SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_LOWEST);
            try { FindShippedCells(); }
            catch(const std::exception& e) { logger::error("Runtime bake: cannot list shipped manifests: {}",e.what()); }
            while(true) {
                Job job;
                {
                    std::unique_lock lock(queueMutex);
                    queueReady.wait(lock,[]{ return !queue.empty(); });
                    // The cell the player is in (distance 0) and the nearest exterior cells first; door destinations last.
                    const auto next=std::min_element(queue.begin(),queue.end(),[](const Job& a,const Job& b) {
                        return std::tie(a.prebake,a.distance)<std::tie(b.prebake,b.distance);
                    });
                    job=std::move(*next); queue.erase(next);
                }
                try { Bake(job); }
                catch(const std::exception& e) { logger::error("Runtime bake: cell {:08x}: {}",job.cell,e.what()); }
                catch(...) { logger::error("Runtime bake: cell {:08x}: unknown error",job.cell); }
            }
        }
    }

    std::string NormalizeModel(std::string model) {
        std::replace(model.begin(),model.end(),'\\','/');
        model=Lower(std::move(model));
        if(model.starts_with("meshes/")) model.erase(0,7);
        return model;
    }

    std::string HashHex(const void* data,size_t size,uint64_t seed) {
        uint64_t h=seed;
        const auto* bytes=static_cast<const unsigned char*>(data);
        for(size_t i=0;i<size;++i) h=(h^bytes[i])*1099511628211ull;
        return std::format("{:016x}",h);
    }

    std::string CellKey(const FormKey& cell) {
        if(IsSkyrimEsm(cell)) return std::format("{:08x}",cell.local);
        const auto text=KeyText(cell);
        return HashHex(text.data(),text.size());
    }

    std::string CacheKey(const Job& job,const std::vector<std::pair<std::string,std::string>>& meshHashes,const Settings& s) {
        std::vector<std::string> rows;
        for(const auto& r:job.references)
            rows.push_back(std::format("{}|{}|{}|{:.2f},{:.2f},{:.2f}|{:.4f},{:.4f},{:.4f}|{:.4f}|{}",KeyText(r.refKey),KeyText(r.baseKey),r.model,
                r.position[0],r.position[1],r.position[2],r.rotation[0],r.rotation[1],r.rotation[2],r.scale,r.roomGroup));
        std::sort(rows.begin(),rows.end());
        // The in-process baker is PrecombineMesh.exe's code (precombine::Bake), so both have one name; "b" is its output
        // version, so a baker fix bakes every cached cell again ("v3": that and the light rows below).
        std::string text=std::format("v3|b{}|{}|{}|{}|{}|",precombine::kOutputVersion,KeyText(job.cellKey),s.chunkSize,s.lightBudget,
                                     s.tool.empty() ? std::string("PrecombineMesh.exe") : Utf8(s.tool.filename()));
        for(const auto& row:rows) text+=row+'\n';
        for(const auto& [model,hash]:meshHashes) text+=model+'='+hash+'\n';
        // Lights reach the baker only with a budget (Light Limit Fix sets none), and count there in any order.
        if(s.lightBudget>=0) {
            auto lights=job.keyLights;
            std::sort(lights.begin(),lights.end());
            for(const auto& l:lights) text+=std::format("L{:.1f},{:.1f},{:.1f},{:.1f}\n",l[0],l[1],l[2],l[3]);
        }
        return HashHex(text.data(),text.size());
    }

    std::vector<std::vector<size_t>> PlanParts(const Job& job,size_t first,size_t later) {
        std::vector<std::vector<size_t>> parts;
        if(job.references.empty()) return parts;
        if(!job.interior || !later) {
            parts.emplace_back(job.references.size());
            for(size_t i=0;i<job.references.size();++i) parts.back()[i]=i;
            return parts;
        }
        std::map<std::string,std::vector<size_t>> groups;
        for(size_t i=0;i<job.references.size();++i) groups[job.references[i].roomGroup].push_back(i);
        const auto distance=[&](const std::vector<size_t>& members) {
            double nearest=INFINITY;
            for(const auto i:members) {
                const auto& p=job.references[i].position;
                nearest=std::min(nearest,std::hypot(double(p[0])-job.origin[0],double(p[1])-job.origin[1],double(p[2])-job.origin[2]));
            }
            return nearest;
        };
        std::vector<std::pair<double,const std::string*>> order;
        for(const auto& [name,members]:groups) order.emplace_back(distance(members),&name);
        std::sort(order.begin(),order.end(),[](const auto& a,const auto& b) { return std::tie(a.first,*a.second)<std::tie(b.first,*b.second); });
        std::vector<size_t> current;
        size_t target=std::max<size_t>(first,1);
        for(const auto& [d,name]:order) {
            const auto& members=groups.at(*name);
            current.insert(current.end(),members.begin(),members.end());
            if(current.size()>=target) { std::sort(current.begin(),current.end()); parts.push_back(std::move(current)); current.clear(); target=later; }
        }
        if(!current.empty()) { std::sort(current.begin(),current.end()); parts.push_back(std::move(current)); }
        return parts;
    }

    void Configure(const Settings& s) { settings=s; }

    void Start() {
        if(!settings.enabled || started.exchange(true)) return;
#ifndef PRECOMBINES_BAKE_PLUGIN
        // An older install's separate SkyrimPrecombinesBake.dll bakes on its own: two bakers would bake every cell twice.
        if(GetModuleHandleW(L"SkyrimPrecombinesBake.dll")) {
            logger::warn("Runtime bake: SkyrimPrecombinesBake.dll from an older version is loaded and bakes instead; remove it and "
                         "Precombines/PrecombineMesh.exe, which this version no longer needs");
            return;
        }
#endif
        // Every check below uses error codes: a game folder the game cannot write in (an install under Program Files
        // without elevation or MO2) threw out of kDataLoaded and ended the game. Each failure leaves the bake
        // off for the session, before any thread starts.
        std::error_code ec;
        if(!settings.tool.empty()) {
            const auto tool=fs::absolute(settings.tool,ec);
            if(ec || !fs::exists(tool,ec)) { logger::error("Runtime bake disabled: {} is missing",Utf8(ec ? settings.tool : tool)); return; }
            settings.tool=tool;
        }
        fs::remove_all(AssetDir(),ec);
        for(const auto& dir:{PartsDir(),RoomsDir(),AssetDir()})
            if(fs::create_directories(dir,ec);ec) {
                logger::error("Runtime bake disabled: cannot create {} ({}); set sCacheFolder in SkyrimPrecombines.ini to a folder the game can write in",
                    Utf8(dir),ec.message());
                return;
            }
        // kDataLoaded: every SKSE plugin is loaded, so the precombine plugin's export is resolvable now.
        if(const auto module=GetModuleHandleW(L"SkyrimPrecombines.dll")) {
            offerPart=reinterpret_cast<OfferPart>(GetProcAddress(module,"SkyrimPrecombines_RuntimeBakePart"));
            offerPrevis=reinterpret_cast<OfferPart>(GetProcAddress(module,"SkyrimPrecombines_RuntimeBakePrevis"));
        }
        if(settings.previs) {
            if(settings.previsTool.empty()) settings.previsTool=fs::path(kManifestDir)/"VisibilityEstimate.exe";
            if(const auto tool=fs::absolute(settings.previsTool,ec);!ec) settings.previsTool=tool;
            if(ec || !fs::exists(settings.previsTool,ec) || !offerPrevis) {
                logger::error("Runtime previs disabled: {}",!offerPrevis ? "SkyrimPrecombines.dll has no previs hand-off" : Utf8(settings.previsTool)+" is missing");
                settings.previsTool.clear();
            }
        }
        std::string lightLimitFix;
        if(settings.detectLightLimitFix && settings.lightBudget>=0 && !(lightLimitFix=LightLimitFix()).empty()) settings.lightBudget=-1;
        std::thread(Worker).detach();
        if(settings.previs && !settings.previsTool.empty()) std::thread(InteriorPrevisWorker).detach();
        if(settings.previs && settings.previsExterior && !settings.previsTool.empty()) std::thread(ExteriorPrevisWorker).detach();
        // Poll attached cells every two seconds on an SKSE task (game objects are only read there). No exception may
        // unwind into the game's task queue: a failed poll is logged once and the next one tries again.
        std::thread([] {
            while(true) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
                if(auto* tasks=SKSE::GetTaskInterface()) tasks->AddTask([] {
                    static std::atomic_bool logged{false};
                    try { Poll(); }
                    catch(const std::exception& e) { if(!logged.exchange(true)) logger::error("Runtime bake: a poll failed: {}",e.what()); }
                    catch(...) { if(!logged.exchange(true)) logger::error("Runtime bake: a poll failed: unknown error"); }
                });
            }
        }).detach();
        logger::info("Runtime bake enabled: {} (chunk size {}, light budget {}{}, parts of {}/{} references, door pre-bake {}); {}",
            settings.tool.empty() ? std::string("in-process baker") : Utf8(settings.tool),settings.chunkSize,settings.lightBudget,lightLimitFix.empty() ? "" : " because "+lightLimitFix+" is active",
            settings.firstPartReferences,settings.partReferences,settings.prebakeDoors,
            offerPart ? "parts are handed to SkyrimPrecombines.dll during play" : "SkyrimPrecombines.dll has no hand-off; baked Skyrim.esm cells are used from the next data load");
    }
}
