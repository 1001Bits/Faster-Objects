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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Engine-free parts of runtime exterior previs (tests/ExteriorPrevisTest.cpp): the loaded terrain as one heightfield,
// splitting a bake's per-view-cell chunk bits between the cells the chunks belong to, and which lights reach a cell.
namespace RuntimeBake::Exterior {
    inline constexpr int kSide = 33;           // terrain vertices per cell edge
    inline constexpr float kStep = 128.f;      // units between terrain vertices
    inline constexpr float kCell = 4096.f;     // units per exterior cell

    // TESObjectLAND::LoadedLandData holds heights[4][289]: 17 x 17 per quadrant, row-major from the quadrant's south-west
    // vertex, relative to the land's base height (the float at LoadedLandData+0x49C0). TESObjectLAND::GetNiPointHeight
    // (SE 1.5.97 RVA 25B830) places vertex i of quadrant q at x = (i % 17) * 128 + X[q], y = (i / 17) * 128 + Y[q] from
    // the cell centre, z = heights[q][i] + base, with X = {-2048, 0, -2048, 0} and Y = {-2048, -2048, 0, 0} (set in
    // FUN_14025BCA0): quadrants south-west, south-east, north-west, north-east. Returns the 33 x 33 grid from the cell's
    // south-west corner, row-major by y, as in the VHGT record.
    inline std::vector<float> StitchTerrain(const float (&heights)[4][289], float base) {
        std::vector<float> out(size_t(kSide) * kSide, 0.f);
        for (int q = 0; q < 4; ++q) {
            const int qx = q & 1, qy = q >> 1;
            for (int i = 0; i < 289; ++i) {
                const int col = i % 17 + 16 * qx, row = i / 17 + 16 * qy;
                out[size_t(row) * kSide + col] = heights[q][i] + base;
            }
        }
        return out;
    }

    inline std::string Base64Encode(const std::vector<uint8_t>& bytes) {
        static constexpr char kTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out;
        size_t i = 0;
        for (; i + 2 < bytes.size(); i += 3) {
            const uint32_t v = (uint32_t(bytes[i]) << 16) | (uint32_t(bytes[i + 1]) << 8) | bytes[i + 2];
            out += kTable[v >> 18]; out += kTable[(v >> 12) & 63]; out += kTable[(v >> 6) & 63]; out += kTable[v & 63];
        }
        if (i + 1 == bytes.size()) {
            const uint32_t v = uint32_t(bytes[i]) << 16;
            out += kTable[v >> 18]; out += kTable[(v >> 12) & 63]; out += "==";
        } else if (i + 2 == bytes.size()) {
            const uint32_t v = (uint32_t(bytes[i]) << 16) | (uint32_t(bytes[i + 1]) << 8);
            out += kTable[v >> 18]; out += kTable[(v >> 12) & 63]; out += kTable[(v >> 6) & 63]; out += '=';
        }
        return out;
    }

    inline std::vector<uint8_t> Base64Decode(const std::string& text) {
        std::vector<uint8_t> out;
        uint32_t buffer = 0;
        int bits = 0;
        for (const char c : text) {
            int v;
            if (c >= 'A' && c <= 'Z') v = c - 'A';
            else if (c >= 'a' && c <= 'z') v = c - 'a' + 26;
            else if (c >= '0' && c <= '9') v = c - '0' + 52;
            else if (c == '+') v = 62;
            else if (c == '/') v = 63;
            else if (c == '=') break;
            else throw std::runtime_error("invalid base64");
            buffer = (buffer << 6) | uint32_t(v);
            bits += 6;
            if (bits >= 8) { bits -= 8; out.push_back(uint8_t((buffer >> bits) & 0xFF)); }
        }
        return out;
    }

    // One view cell's chunk bits (base64, bit k = chunk k, least significant bit first) cut to chunks
    // [start, start + count), renumbered from 0. Bits past the decoded bytes read as 0.
    inline std::string SliceBits(const std::string& base64, size_t start, size_t count) {
        const auto in = Base64Decode(base64);
        std::vector<uint8_t> out((count + 7) / 8, 0);
        for (size_t k = 0; k < count; ++k) {
            const size_t from = start + k;
            if (from / 8 < in.size() && (in[from / 8] >> (from % 8)) & 1) out[k / 8] |= uint8_t(1u << (k % 8));
        }
        return Base64Encode(out);
    }

    // The exterior cell (grid x, y) a view cell of `size` units lies in.
    inline std::array<int32_t, 2> CellOfViewCell(int64_t ix, int64_t iy, double size) {
        const auto cell = [&](int64_t i) { return int32_t(std::floor(double(i) * size / double(kCell))); };
        return {cell(ix), cell(iy)};
    }

    // Exterior light budget: how far a chunk's bound can reach out of its cell. A full-cell chunk's sphere
    // reaches about 850 units past each edge's middle, plus 1.41 times as far as its meshes stick out of the cell (a shape
    // is binned by its bound's centre, so one near the edge can overhang by its radius; 900 units already reach past half a
    // cell), whatever the chunk size. A whole cell: every light inside the eight neighbours counts, and the runtime bake
    // offers no others (RuntimeBake.cpp Poll), so a chunk reaching more than a cell out of its own, or a light reaching in
    // from two cells away, can still exceed the budget.
    inline constexpr double kLightMargin = kCell;

    // Whether a light sphere (x, y, z, radius) can touch a chunk of exterior cell (gx, gy): its distance in x and y to the
    // cell square grown by `margin` is below its radius. Height is ignored, which only keeps more lights.
    inline bool LightReachesCell(const std::array<double, 4>& light, int32_t gx, int32_t gy, double margin = kLightMargin) {
        const auto outside = [&](double p, int32_t g) {
            const double lo = double(g) * kCell - margin, hi = double(g + 1) * kCell + margin;
            return p < lo ? lo - p : p > hi ? p - hi : 0.0;
        };
        return std::hypot(outside(light[0], gx), outside(light[1], gy)) < light[3];
    }

    // Light spheres in value order, each once: the cells of a previs region list a light near their border each.
    inline void MergeLights(std::vector<std::array<double, 4>>& lights) {
        std::sort(lights.begin(), lights.end());
        lights.erase(std::unique(lights.begin(), lights.end()), lights.end());
    }
}
