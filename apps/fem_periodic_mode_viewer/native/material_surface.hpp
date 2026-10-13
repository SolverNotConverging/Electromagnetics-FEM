#pragma once

#include "model.hpp"
#include <algorithm>
#include <cmath>
#include <map>

namespace femperiodic {

struct MaterialFace {
    std::array<std::int64_t, 3> vertices;
    std::size_t owner;
    bool boundary;
};

// Extract interfaces before surface rendering discards interior tetra faces.
// Internal single-owner faces bound excluded PEC/PMC geometry, which has no
// bulk material samples. Domain faces (including periodic z ends) are not metal.
inline std::vector<MaterialFace> materialFaces(
    const MeshData& mesh, const MaterialState& material) {
    if (mesh.points.empty() || mesh.cells.empty()
        || material.epsilonR.size() != mesh.cells.size()
        || material.muR.size() != mesh.cells.size()) return {};
    struct FaceOwners { std::size_t first; std::size_t second; int count; };
    std::map<std::array<std::int64_t, 3>, FaceOwners> faces;
    for (std::size_t cell = 0; cell < mesh.cells.size(); ++cell) {
        const auto& vertices = mesh.cells[cell];
        if (vertices.size() != 4) continue;
        for (std::size_t omitted = 0; omitted < 4; ++omitted) {
            std::array<std::int64_t, 3> key{};
            std::size_t offset = 0;
            for (std::size_t i = 0; i < 4; ++i)
                if (i != omitted) key[offset++] = vertices[i];
            std::sort(key.begin(), key.end());
            auto [entry, inserted] = faces.try_emplace(key, FaceOwners{cell, cell, 1});
            if (!inserted) { entry->second.second = cell; ++entry->second.count; }
        }
    }
    std::array<double, 3> lower = mesh.points.front(), upper = lower;
    for (const auto& point : mesh.points)
        for (std::size_t axis = 0; axis < 3; ++axis) {
            lower[axis] = std::min(lower[axis], point[axis]);
            upper[axis] = std::max(upper[axis], point[axis]);
        }
    auto contrast = [&](std::size_t cell) {
        double value = 0.0;
        for (std::size_t axis = 0; axis < 3; ++axis)
            value += std::abs(material.epsilonR[cell][axis] - Complex{1.0})
                + std::abs(material.muR[cell][axis] - Complex{1.0});
        return value;
    };
    auto sameMaterial = [&](std::size_t first, std::size_t second) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            for (const auto* tensor : {&material.epsilonR, &material.muR}) {
                const auto a = (*tensor)[first][axis], b = (*tensor)[second][axis];
                if (std::abs(a - b) > 1e-8 * std::max({1.0, std::abs(a), std::abs(b)}))
                    return false;
            }
        }
        return true;
    };
    bool uniform = true;
    for (std::size_t cell = 1; cell < mesh.cells.size(); ++cell)
        uniform &= sameMaterial(0, cell);
    std::vector<MaterialFace> result;
    for (const auto& [vertices, owners] : faces) {
        if (owners.count == 2) {
            if (!sameMaterial(owners.first, owners.second)) {
                const auto owner = contrast(owners.first) >= contrast(owners.second)
                    ? owners.first : owners.second;
                result.push_back({vertices, owner, false});
            }
            continue;
        }
        bool exterior = false;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const auto tolerance = 1e-8 * (upper[axis] - lower[axis]);
            for (const auto bound : {lower[axis], upper[axis]}) {
                exterior |= std::all_of(vertices.begin(), vertices.end(), [&](auto vertex) {
                    return std::abs(mesh.points[static_cast<std::size_t>(vertex)][axis] - bound)
                        <= tolerance;
                });
            }
        }
        if (!exterior || uniform || contrast(owners.first) > 1e-10)
            result.push_back({vertices, owners.first, !exterior});
    }
    return result;
}

} // namespace femperiodic
