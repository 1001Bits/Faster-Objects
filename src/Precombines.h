#pragma once

#include <cstddef>
#include <cstdint>

namespace Precombines {
    void Start();
    void BeforeLoad();
    void AfterLoad();
    // Precombines on or off in play (the in-game menu, the precombines shortcut key; any thread): an SKSE task under the
    // manager's lock; off restores every original object at once, and on in a session that started with them off also
    // reads the runtime bake's cached cells. Never waits for the manager's lock on the caller's thread.
    void SetEnabled(bool on);
    // True after precombines switched themselves off for an error this session, until they are switched on again (the
    // menu's status line). Any thread.
    bool Failed();
}

// Read-only counters for diagnostics (exported as SkyrimPrecombines_GetStats).
// Times are cumulative wall time spent inside the periodic SKSE task.
struct PrecombinesStats {
    std::uint32_t size;            // caller sets sizeof(PrecombinesStats)
    std::uint32_t version;         // 4 (3: culledRoots in the former reserved slot, previsHidden; 4: Hi-Z occlusion)
    std::uint64_t ticks;
    std::uint64_t tickNanoseconds;
    std::uint64_t tickMaxNanoseconds;  // since the previous read with resetMax
    std::uint64_t attaches;
    std::uint64_t dissolves;
    std::uint32_t activeChunks;
    std::uint32_t hiddenSources;
    std::uint32_t lastTickThread;
    std::uint32_t culledRoots;     // reference roots culled because every shape is merged
    std::uint32_t previsHidden;    // active chunks currently culled by precomputed visibility
    std::uint32_t reserved;
    // Version 4. A caller built against version 3 passes the smaller size and gets version 3.
    std::uint32_t hizEnabled;
    std::uint32_t hizGridAge;      // frames between the depth capture and the camera it was reprojected to
    std::uint64_t hizCaptures;     // depth grids captured when the latest pyramid was built
    std::uint64_t hizBuilds;       // pyramids built (frames culled with Hi-Z)
    std::uint32_t hizFadeTested;   // last complete frame, main camera only
    std::uint32_t hizFadeCulled;
    std::uint32_t hizGeometryTested;
    std::uint32_t hizGeometryCulled;
    std::uint32_t hizPrepareMicros;
    std::uint32_t hizLanded;       // captured texels that landed in the current view
    std::uint32_t previsMainCulled;  // geometries previs kept out of the main view (main-camera previs only)
    std::uint32_t previsReferencesCulled;  // references reference previs kept out of the main view in the last frame
};
inline constexpr std::uint32_t kPrecombinesStatsV3Size=static_cast<std::uint32_t>(offsetof(PrecombinesStats,hizEnabled));
