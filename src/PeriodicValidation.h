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

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace Precombines {
    // Keep the last accepted representation for a value whose acceptance rule
    // is fixed for this object's lifetime. Identical bytes need no tolerance or
    // finiteness checks; changed bytes always run the complete acceptance rule.
    // This does not substitute a previous sample for reading the current value.
    template <class Value>
    class ValidatedSnapshot {
        static_assert(std::is_trivially_copyable_v<Value>);
    public:
        template <class Check>
        bool Get(const Value& value, Check&& check) {
            Value sampled;
            std::memcpy(&sampled, &value, sizeof(Value));
            if (accepted && std::memcmp(&snapshot, &sampled, sizeof(Value)) == 0) return true;
            accepted = false;
            if (!std::forward<Check>(check)(sampled)) return false;
            std::memcpy(&snapshot, &sampled, sizeof(Value));
            accepted = true;
            return true;
        }
    private:
        Value snapshot{};
        bool accepted = false;
    };

    // Definition IDs, rather than indices in the mutable active vector, prevent
    // removals and new attachments from skipping a chunk indefinitely.
    class ValidationSchedule {
    public:
        static constexpr std::uint64_t ticksPerSweep = 5;
        void BeginTick() { ++epoch; }
        [[nodiscard]] std::uint64_t Epoch() const { return epoch; }
        [[nodiscard]] bool Due(std::size_t definition) const {
            return definition % ticksPerSweep == epoch % ticksPerSweep;
        }
    private:
        std::uint64_t epoch = 0;
    };

    // A reference may contribute to chunks in several definition groups. Give
    // it its own stable group so it is sampled once per sweep, rather than once
    // for each chunk group that uses it. Weak entries never retain unloaded refs.
    template <class Check>
    class ValidationBuckets {
    public:
        void Add(std::size_t id, const std::shared_ptr<Check>& check) {
            buckets[id % ValidationSchedule::ticksPerSweep].emplace_back(check);
        }
        template <class Sample>
        bool Visit(std::uint64_t epoch, Sample&& sample) {
            bool valid = true;
            std::erase_if(buckets[epoch % ValidationSchedule::ticksPerSweep], [&](const auto& weak) {
                if (auto check = weak.lock()) {
                    valid = sample(*check) && valid;  // Visit all, even after a failure.
                    return false;
                }
                return true;
            });
            return valid;
        }
        void Clear() { for (auto& bucket : buckets) bucket.clear(); }
    private:
        std::array<std::vector<std::weak_ptr<Check>>, ValidationSchedule::ticksPerSweep> buckets;
    };

    // A result belongs to one task only. Even a failed check is sampled afresh
    // on the next task; retries never inherit an earlier task's stale result.
    class ValidationResult {
    public:
        template <class Check>
        bool Get(std::uint64_t task, Check&& check) {
            if (epoch != task) {
                valid = std::forward<Check>(check)();
                epoch = task;
            }
            return valid;
        }
        [[nodiscard]] bool Failed(std::uint64_t task) const {
            return epoch == task && !valid;
        }
    private:
        std::uint64_t epoch = 0;
        bool valid = true;
    };
}
