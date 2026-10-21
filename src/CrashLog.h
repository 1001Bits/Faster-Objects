#pragma once

// Logs fatal-looking exceptions (module+offset, registers, stack return-address candidates) to the plugin log.
namespace CrashLog {
    void Install();
}
