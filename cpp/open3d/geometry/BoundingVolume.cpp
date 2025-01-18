// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include "open3d/geometry/BoundingVolume.h"

#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/Qhull.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace geometry {

OrientedBoundingBox& OrientedBoundingBox::Clear() {
    center_.setZero();
    extent_.setZero();
    R_ = Eigen::Matrix3d::Identity();
    color_.setOnes();
    return *this;
}

bool OrientedBoundingBox::IsEmpty() const { return Volume() <= 0; }

Eigen::Vector3d OrientedBoundingBox::GetMinBound() const {
    auto points = GetBoxPoints();
    return ComputeMinBound(points);
}

Eigen::Vector3d OrientedBoundingBox::GetMaxBound() const {
    auto points = GetBoxPoints();
    return ComputeMaxBound(points);
}

Eigen::Vector3d OrientedBoundingBox::GetCenter() const { return center_; }

AxisAlignedBoundingBox OrientedBoundingBox::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox::CreateFromPoints(GetBoxPoints());
}

OrientedBoundingBox OrientedBoundingBox::GetOrientedBoundingBox(bool) const {
    return *this;
}

OrientedBoundingBox OrientedBoundingBox::GetMinimalOrientedBoundingBox(
        bool) const {
    return *this;
}

OrientedBoundingBox& OrientedBoundingBox::Transform(
        const Eigen::Matrix4d& transformation) {
    utility::LogError(
            "A general transform of an OrientedBoundingBox is not implemented. "
            "Call Translate, Scale, and Rotate.");
    return *this;
}

OrientedBoundingBox& OrientedBoundingBox::Translate(
        const Eigen::Vector3d& translation, bool relative) {
    if (relative) {
        center_ += translation;
    } else {
        center_ = translation;
    }
    return *this;
}

OrientedBoundingBox& OrientedBoundingBox::Scale(const double scale,
                                                const Eigen::Vector3d& center) {
    extent_ *= scale;
    center_ = scale * (center_ - center) + center;
    return *this;
}

OrientedBoundingBox& OrientedBoundingBox::Rotate(
        const Eigen::Matrix3d& R, const Eigen::Vector3d& center) {
    R_ = R * R_;
    center_ = R * (center_ - center) + center;
    return *this;
}

double OrientedBoundingBox::Volume() const {
    return extent_(0) * extent_(1) * extent_(2);
}

std::vector<Eigen::Vector3d> OrientedBoundingBox::GetBoxPoints() const {
    Eigen::Vector3d x_axis = R_ * Eigen::Vector3d(extent_(0) / 2, 0, 0);
    Eigen::Vector3d y_axis = R_ * Eigen::Vector3d(0, extent_(1) / 2, 0);
    Eigen::Vector3d z_axis = R_ * Eigen::Vector3d(0, 0, extent_(2) / 2);
    std::vector<Eigen::Vector3d> points(8);
    points[0] = center_ - x_axis - y_axis - z_axis;
    points[1] = center_ + x_axis - y_axis - z_axis;
    points[2] = center_ - x_axis + y_axis - z_axis;
    points[3] = center_ - x_axis - y_axis + z_axis;
    points[4] = center_ + x_axis + y_axis + z_axis;
    points[5] = center_ - x_axis + y_axis + z_axis;
    points[6] = center_ + x_axis - y_axis + z_axis;
    points[7] = center_ + x_axis + y_axis - z_axis;
    return points;
}

std::vector<size_t> OrientedBoundingBox::GetPointIndicesWithinBoundingBox(
        const std::vector<Eigen::Vector3d>& points) const {
    std::vector<size_t> indices;
    Eigen::Vector3d dx = R_ * Eigen::Vector3d(1, 0, 0);
    Eigen::Vector3d dy = R_ * Eigen::Vector3d(0, 1, 0);
    Eigen::Vector3d dz = R_ * Eigen::Vector3d(0, 0, 1);
    for (size_t idx = 0; idx < points.size(); idx++) {
        Eigen::Vector3d d = points[idx] - center_;
        if (std::abs(d.dot(dx)) <= extent_(0) / 2 &&
            std::abs(d.dot(dy)) <= extent_(1) / 2 &&
            std::abs(d.dot(dz)) <= extent_(2) / 2) {
            indices.push_back(idx);
        }
    }
    return indices;
}

OrientedBoundingBox OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(
        const AxisAlignedBoundingBox& aabox) {
    OrientedBoundingBox obox;
    obox.center_ = aabox.GetCenter();
    obox.extent_ = aabox.GetExtent();
    obox.R_ = Eigen::Matrix3d::Identity();
    return obox;
}

OrientedBoundingBox OrientedBoundingBox::CreateFromPoints(
        const std::vector<Eigen::Vector3d>& points, bool robust) {
    PointCloud hull_pcd;
    std::vector<size_t> hull_point_indices;
    {
        std::shared_ptr<TriangleMesh> mesh;
        std::tie(mesh, hull_point_indices) =
                Qhull::ComputeConvexHull(points, robust);
        hull_pcd.points_ = mesh->vertices_;
    }

    Eigen::Vector3d mean;
    Eigen::Matrix3d cov;
    std::tie(mean, cov) = hull_pcd.ComputeMeanAndCovariance();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    Eigen::Vector3d evals = es.eigenvalues();
    Eigen::Matrix3d R = es.eigenvectors();

    if (evals(1) > evals(0)) {
        std::swap(evals(1), evals(0));
        Eigen::Vector3d tmp = R.col(1);
        R.col(1) = R.col(0);
        R.col(0) = tmp;
    }
    if (evals(2) > evals(0)) {
        std::swap(evals(2), evals(0));
        Eigen::Vector3d tmp = R.col(2);
        R.col(2) = R.col(0);
        R.col(0) = tmp;
    }
    if (evals(2) > evals(1)) {
        std::swap(evals(2), evals(1));
        Eigen::Vector3d tmp = R.col(2);
        R.col(2) = R.col(1);
        R.col(1) = tmp;
    }
    R.col(0) /= R.col(0).norm();
    R.col(1) /= R.col(1).norm();
    R.col(2) = R.col(0).cross(R.col(1));

    for (size_t i = 0; i < hull_point_indices.size(); ++i) {
        hull_pcd.points_[i] =
                R.transpose() * (points[hull_point_indices[i]] - mean);
    }

    const auto aabox = hull_pcd.GetAxisAlignedBoundingBox();

    OrientedBoundingBox obox;
    obox.center_ = R * aabox.GetCenter() + mean;
    obox.R_ = R;
    obox.extent_ = aabox.GetExtent();

    return obox;
}

OrientedBoundingBox OrientedBoundingBox::CreateFromPointsMinimalApprox(
        const std::vector<Eigen::Vector3d>& points, bool robust) {
    std::shared_ptr<TriangleMesh> mesh;
    std::tie(mesh, std::ignore) = Qhull::ComputeConvexHull(points, robust);
    double min_vol = -1;
    OrientedBoundingBox min_box;
    PointCloud hull_pcd;
    for (auto& tri : mesh->triangles_) {
        hull_pcd.points_ = mesh->vertices_;
        Eigen::Vector3d a = mesh->vertices_[tri(0)];
        Eigen::Vector3d b = mesh->vertices_[tri(1)];
        Eigen::Vector3d c = mesh->vertices_[tri(2)];
        Eigen::Vector3d u = b - a;
        Eigen::Vector3d v = c - a;
        Eigen::Vector3d w = u.cross(v);
        v = w.cross(u);
        u = u / u.norm();
        v = v / v.norm();
        w = w / w.norm();
        Eigen::Matrix3d m_rot;
        m_rot << u[0], v[0], w[0], u[1], v[1], w[1], u[2], v[2], w[2];
        hull_pcd.Rotate(m_rot.inverse(), a);

        const auto aabox = hull_pcd.GetAxisAlignedBoundingBox();
        double volume = aabox.Volume();
        if (min_vol == -1. || volume < min_vol) {
            min_vol = volume;
            min_box = aabox.GetOrientedBoundingBox();
            min_box.Rotate(m_rot, a);
        }
    }
    return min_box;
}

OrientedBoundingBox OrientedBoundingBox::CreateFromPointsMinimal(
        const std::vector<Eigen::Vector3d>& points, bool robust) {
    // ------------------------------------------------------------
    // 0) Compute the convex hull of the input point cloud
    // ------------------------------------------------------------
    if (points.empty()) {
        utility::LogError("CreateFromPointsMinimal: Input point set is empty.");
        return OrientedBoundingBox();
    }
    std::shared_ptr<TriangleMesh> hullMesh;
    std::tie(hullMesh, std::ignore) = Qhull::ComputeConvexHull(points, robust);
    if (!hullMesh) {
        utility::LogError("Failed to compute convex hull.");
        return OrientedBoundingBox();
    }

    // Get convex hull vertices and triangles
    const std::vector<Eigen::Vector3d>& hullV = hullMesh->vertices_;
    const std::vector<Eigen::Vector3i>& hullT = hullMesh->triangles_;
    int numVertices = static_cast<int>(hullV.size());
    int numTriangles = static_cast<int>(hullT.size());

    OrientedBoundingBox minOBB;
    double minVolume = std::numeric_limits<double>::max();

    // Handle degenerate planar cases up front.
    if (numVertices <= 3 || numTriangles < 1) {  // Handle degenerate case
        utility::LogError("Convex hull is degenerate.");
        return OrientedBoundingBox();
    }

    auto mapOBBToClosestIdentity = [&](OrientedBoundingBox& obb) {
        Eigen::Matrix3d& R = obb.R_;
        Eigen::Vector3d& extent = obb.extent_;
        Eigen::Vector3d col[3] = {R.col(0), R.col(1), R.col(2)};
        Eigen::Vector3d ext = extent;
        double best_score = -1e9;
        Eigen::Matrix3d best_R;
        Eigen::Vector3d best_extent;
        // Hard-coded permutations of indices [0,1,2]
        static const std::array<std::array<int, 3>, 6> permutations = {
                {{{0, 1, 2}},
                 {{0, 2, 1}},
                 {{1, 0, 2}},
                 {{1, 2, 0}},
                 {{2, 0, 1}},
                 {{2, 1, 0}}}};

        // Evaluate all 6 permutations × 8 sign flips = 48 candidates
        for (const auto& p : permutations) {
            for (int sign_bits = 0; sign_bits < 8; ++sign_bits) {
                // Derive the sign of each axis from bits (0 => -1, 1 => +1)
                // s0 is bit0, s1 is bit1, s2 is bit2 of sign_bits
                const int s0 = (sign_bits & 1) ? 1 : -1;
                const int s1 = (sign_bits & 2) ? 1 : -1;
                const int s2 = (sign_bits & 4) ? 1 : -1;

                // Construct candidate columns
                Eigen::Vector3d c0 = s0 * col[p[0]];
                Eigen::Vector3d c1 = s1 * col[p[1]];
                Eigen::Vector3d c2 = s2 * col[p[2]];

                // Score: how close are we to the identity?
                // Since e_x = (1,0,0), e_y = (0,1,0), e_z = (0,0,1),
                // we can skip dot products & do c0(0)+c1(1)+c2(2).
                double score = c0(0) + c1(1) + c2(2);

                // If this orientation is better, update the best.
                if (score > best_score) {
                    best_score = score;
                    best_R.col(0) = c0;
                    best_R.col(1) = c1;
                    best_R.col(2) = c2;

                    // Re-permute extents: if the axis p[0] in old frame
                    // now goes to new X, etc.
                    best_extent(0) = ext(p[0]);
                    best_extent(1) = ext(p[1]);
                    best_extent(2) = ext(p[2]);
                }
            }
        }

        // Update the OBB with the best orientation found
        obb.R_ = best_R;
        obb.extent_ = best_extent;
    };

    // --------------------------------------------------------------------
    // 1) Precompute vertex adjacency data, face normals, and edge data
    // --------------------------------------------------------------------
    std::vector<std::vector<int>> adjacencyData;
    adjacencyData.reserve(numVertices);
    adjacencyData.insert(adjacencyData.end(), numVertices, std::vector<int>());

    std::vector<Eigen::Vector3d> faceNormals;
    faceNormals.reserve(numTriangles);

    // Each edge is stored as (v0, v1).
    std::vector<std::pair<int, int>> edges;
    edges.reserve(numVertices * 2);

    // Each edge knows which two faces it belongs to: (f0, f1).
    std::vector<std::pair<int, int>> facesForEdge;
    facesForEdge.reserve(numVertices * 2);

    constexpr unsigned int emptyEdge = std::numeric_limits<unsigned int>::max();
    std::vector<unsigned int> vertexPairsToEdges(numVertices * numVertices,
                                                 emptyEdge);

    for (int i = 0; i < numTriangles; ++i) {
        const Eigen::Vector3i& tri = hullT[i];
        int t0 = tri(0), t1 = tri(1), t2 = tri(2);
        int v0 = t2, v1 = t0;

        for (int j = 0; j < 3; ++j) {
            v1 = tri(j);

            // Build Adjacency Data (vertex -> adjacent vertices)
            adjacencyData[v0].push_back(v1);

            // Register Edges (edge -> neighbouring faces)
            unsigned int& refIdx1 = vertexPairsToEdges[v0 * numVertices + v1];
            unsigned int& refIdx2 = vertexPairsToEdges[v1 * numVertices + v0];
            if (refIdx1 == emptyEdge) {
                // Not registered yet
                unsigned int newIdx = static_cast<unsigned int>(edges.size());
                refIdx1 = newIdx;
                refIdx2 = newIdx;
                edges.emplace_back(v0, v1);
                facesForEdge.emplace_back(i, -1);
            } else {
                // Already existing, update the second face index
                facesForEdge[refIdx1].second = i;
            }

            v0 = v1;
        }
        // Compute Face Normal
        auto n = (hullV[t1] - hullV[t0]).cross(hullV[t2] - hullV[t0]);
        faceNormals.push_back(n.normalized());
    }

    // ------------------------------------------------------------
    // 2) Precompute "antipodal vertices" for each edge of the hull
    // ------------------------------------------------------------

    // Throughout the algorithm, internal edges can all be discarded.
    auto isInternalEdge = [&](std::size_t iEdge) noexcept {
        return (faceNormals[facesForEdge[iEdge].first].dot(
                        faceNormals[facesForEdge[iEdge].second]) > 1.0 - 1e-4);
    };

    // Throughout the whole algorithm, this array stores an auxiliary structure
    // for performing graph searches on the vertices of the convex hull.
    // Conceptually each index of the array stores a boolean whether we have
    // visited that vertex or not during the current search. However storing
    // such booleans is slow, since we would have to perform a linear-time scan
    // through this array before next search to reset each boolean to unvisited
    // false state. Instead, store a number, called a "color" for each vertex to
    // specify whether that vertex has been visited, and manage a global color
    // counter floodFillVisitColor that represents the visited vertices. At any
    // given time, the vertices that have already been visited have the value
    // floodFillVisited[i] == floodFillVisitColor in them. This gives a win that
    // we can perform constant-time clears of the floodFillVisited array, by
    // simply incrementing the "color" counter to clear the array.

    int edgeSize = edges.size();
    std::vector<std::vector<int>> antipodalPointsForEdge(edgeSize);
    antipodalPointsForEdge.reserve(edgeSize);

    std::vector<unsigned int> floodFillVisited(numVertices, 0u);
    unsigned int floodFillVisitColor = 1u;

    auto markVertexVisited = [&](int v) {
        floodFillVisited[v] = floodFillVisitColor;
    };

    auto haveVisitedVertex = [&](int v) -> bool {
        return floodFillVisited[v] == floodFillVisitColor;
    };

    auto clearGraphSearch = [&]() { ++floodFillVisitColor; };

    auto isVertexAntipodalToEdge =
            [&](int vi, const std::vector<int>& neighbors,
                const Eigen::Vector3d& f1a,
                const Eigen::Vector3d& f1b) noexcept -> bool {
        constexpr double epsilon = 1e-4;
        constexpr double degenerateThreshold = -5e-2;
        double tMin = 0.0;
        double tMax = 1.0;

        // Precompute values outside the loop for efficiency.
        const auto& v = hullV[vi];
        Eigen::Vector3d f1b_f1a = f1b - f1a;

        // Iterate over each neighbor.
        for (int neighborIndex : neighbors) {
            const auto& neighbor = hullV[neighborIndex];

            // Compute edge vector e = neighbor - v.
            Eigen::Vector3d e = neighbor - v;

            // Compute dot products manually for efficiency.
            double s = f1b_f1a.dot(e);
            double n = f1b.dot(e);

            // Adjust tMin and tMax based on the value of s.
            if (s > epsilon) {
                tMax = std::min(tMax, n / s);
            } else if (s < -epsilon) {
                tMin = std::max(tMin, n / s);
            } else if (n < -epsilon) {
                // No feasible t if n is negative when s is nearly zero.
                return false;
            }

            // If the valid interval for t has degenerated, exit early.
            if ((tMax - tMin) < degenerateThreshold) {
                return false;
            }
        }
        return true;
    };

    auto extremeVertexConvex = [&](auto& self, const Eigen::Vector3d& direction,
                                   std::vector<unsigned int>& floodFillVisited,
                                   unsigned int floodFillVisitColor,
                                   double& mostExtremeDistance,
                                   int startingVertex) -> int {
        // Compute dot product for the starting vertex.
        double curD = direction.dot(hullV[startingVertex]);

        // Cache neighbor list for the starting vertex.
        const int* neighbors = &adjacencyData[startingVertex][0];
        const int* neighborsEnd =
                neighbors + adjacencyData[startingVertex].size();

        // Mark starting vertex as visited.
        floodFillVisited[startingVertex] = floodFillVisitColor;

        // Traverse neighbors to find more extreme vertices.
        int secondBest = -1;
        double secondBestD = curD - 1e-3;
        while (neighbors != neighborsEnd) {
            int n = *neighbors++;
            if (floodFillVisited[n] != floodFillVisitColor) {
                double d = direction.dot(hullV[n]);
                if (d > curD) {
                    // Found a new vertex with higher dot product.
                    startingVertex = n;
                    curD = d;
                    floodFillVisited[startingVertex] = floodFillVisitColor;
                    neighbors = &adjacencyData[startingVertex][0];
                    neighborsEnd =
                            neighbors + adjacencyData[startingVertex].size();
                    secondBest = -1;
                    secondBestD = curD - 1e-3;
                } else if (d > secondBestD) {
                    // Update second-best candidate.
                    secondBest = n;
                    secondBestD = d;
                }
            }
        }

        // Explore second-best neighbor recursively if valid.
        if (secondBest != -1 &&
            floodFillVisited[secondBest] != floodFillVisitColor) {
            double secondMostExtreme = -std::numeric_limits<double>::infinity();
            int secondTry =
                    self(self, direction, floodFillVisited, floodFillVisitColor,
                         secondMostExtreme, secondBest);

            if (secondMostExtreme > curD) {
                mostExtremeDistance = secondMostExtreme;
                return secondTry;
            }
        }

        mostExtremeDistance = curD;
        return startingVertex;
    };

    // The currently best variant for establishing a spatially coherent
    // traversal order.
    std::vector<int> spatialFaceOrder;
    spatialFaceOrder.reserve(numTriangles);
    std::vector<int> spatialEdgeOrder;
    spatialEdgeOrder.reserve(edgeSize);

    // Initialize random number generator
    std::random_device rd;   // Obtain a random number from hardware
    std::mt19937 rng(rd());  // Seed the generator
    {  // Explicit scope for variables that are not needed after this.

        std::vector<unsigned int> visitedEdges(edgeSize, 0u);
        std::vector<unsigned int> visitedFaces(numTriangles, 0u);

        std::vector<std::pair<int, int>> traverseStackEdges;
        traverseStackEdges.reserve(edgeSize);
        traverseStackEdges.emplace_back(0, adjacencyData[0].front());
        while (!traverseStackEdges.empty()) {
            auto e = traverseStackEdges.back();
            traverseStackEdges.pop_back();

            // Find edge index
            int edgeIdx = vertexPairsToEdges[e.first * numVertices + e.second];
            if (visitedEdges[edgeIdx]) continue;
            visitedEdges[edgeIdx] = 1;
            auto& ff = facesForEdge[edgeIdx];
            if (!visitedFaces[ff.first]) {
                visitedFaces[ff.first] = 1;
                spatialFaceOrder.push_back(ff.first);
            }
            if (!visitedFaces[ff.second]) {
                visitedFaces[ff.second] = 1;
                spatialFaceOrder.push_back(ff.second);
            }

            // If not an internal edge, keep it
            if (!isInternalEdge(edgeIdx)) {
                spatialEdgeOrder.push_back(edgeIdx);
            }

            int v0 = e.second;
            size_t sizeBefore = traverseStackEdges.size();
            for (int v1 : adjacencyData[v0]) {
                int e1 = vertexPairsToEdges[v0 * numVertices + v1];
                if (visitedEdges[e1]) continue;
                traverseStackEdges.push_back(std::make_pair(v0, v1));
            }

            // Randomly shuffle newly added edges
            int nNewEdges =
                    static_cast<int>(traverseStackEdges.size() - sizeBefore);
            if (nNewEdges > 0) {
                std::uniform_int_distribution<> distr(0, nNewEdges - 1);
                int r = distr(rng);
                std::swap(traverseStackEdges.back(),
                          traverseStackEdges[sizeBefore + r]);
            }
        }
    }

    // --------------------------------------------------------------------
    // 3) Precompute "sidepodal edges" for each edge of the hull
    // --------------------------------------------------------------------

    // Stores a memory of yet unvisited vertices for current graph search.
    std::vector<int> traverseStack;

    // Since we do several extreme vertex searches, and the search directions
    // have a lot of spatial locality, always start the search for the next
    // extreme vertex from the extreme vertex that was found during the previous
    // iteration for the previous edge. This has been profiled to improve
    // overall performance by as much as 15-25%.
    int startVertex = 0;

    // Precomputation: for each edge, we need to compute the list of potential
    // antipodal points (points on the opposing face of an enclosing OBB of the
    // face that is flush with the given edge of the polyhedron).
    for (int edgeI : spatialEdgeOrder) {
        auto [faceI_a, faceI_b] = facesForEdge[edgeI];
        const Eigen::Vector3d& f1a = faceNormals[faceI_a];
        const Eigen::Vector3d& f1b = faceNormals[faceI_b];

        double dummy;
        clearGraphSearch();
        startVertex =
                extremeVertexConvex(extremeVertexConvex, -f1a, floodFillVisited,
                                    floodFillVisitColor, dummy, startVertex);
        clearGraphSearch();

        traverseStack.push_back(startVertex);
        markVertexVisited(startVertex);
        while (!traverseStack.empty()) {
            int v = traverseStack.back();
            traverseStack.pop_back();
            const auto& neighbors = adjacencyData[v];
            if (isVertexAntipodalToEdge(v, neighbors, f1a, f1b)) {
                if (edges[edgeI].first == v || edges[edgeI].second == v) {
                    return OrientedBoundingBox();
                }
                antipodalPointsForEdge[edgeI].push_back(v);
                for (size_t j = 0; j < neighbors.size(); ++j) {
                    if (!haveVisitedVertex(neighbors[j])) {
                        traverseStack.push_back(neighbors[j]);
                        markVertexVisited(neighbors[j]);
                    }
                }
            }
        }

        // Robustness: If the above search did not find any antipodal points,
        // add the first found extreme point at least, since it is always an
        // antipodal point. This is known to occur very rarely due to numerical
        // imprecision in the above loop over adjacent edges.
        if (antipodalPointsForEdge[edgeI].empty()) {
            // Getting here is most likely a bug. Fall back to linear scan,
            // which is very slow.
            for (int j = 0; j < numVertices; ++j) {
                if (isVertexAntipodalToEdge(j, adjacencyData[j], f1a, f1b)) {
                    antipodalPointsForEdge[edgeI].push_back(j);
                }
            }
        }
    }

    // Data structure for sidepodal vertices.
    std::vector<unsigned char> sidepodalVertices(edgeSize * numVertices, 0);

    // Stores for each edge i the list of all sidepodal edge indices j that it
    // can form an OBB with.
    std::vector<std::vector<int>> compatibleEdges(edgeSize);
    compatibleEdges.reserve(edgeSize);

    // Compute all sidepodal edges for each edge by performing a graph search.
    // The set of sidepodal edges is connected in the graph, which lets us avoid
    // having to iterate over each edge pair of the convex hull.
    for (int edgeI : spatialEdgeOrder) {
        auto [faceI_a, faceI_b] = facesForEdge[edgeI];
        const Eigen::Vector3d& f1a = faceNormals[faceI_a];
        const Eigen::Vector3d& f1b = faceNormals[faceI_b];

        // Pixar orthonormal basis code:
        // https://graphics.pixar.com/library/OrthonormalB/paper.pdf
        Eigen::Vector3d deadDirection = (f1a + f1b) * 0.5;
        Eigen::Vector3d basis1, basis2;
        double sign = std::copysign(1.0, deadDirection.z());
        const double a = -1.0 / (sign + deadDirection.z());
        const double b = deadDirection.x() * deadDirection.y() * a;
        basis1 = Eigen::Vector3d(
                1.0 + sign * deadDirection.x() * deadDirection.x() * a,
                sign * b, -sign * deadDirection.x());
        basis2 = Eigen::Vector3d(
                b, sign + deadDirection.y() * deadDirection.y() * a,
                -deadDirection.y());

        double dummy;
        Eigen::Vector3d dir =
                (f1a.cross(Eigen::Vector3d(0, 1, 0))).normalized();
        if (dir.norm() < 1e-4) {
            dir = Eigen::Vector3d(0, 0, 1);  // If f1a is parallel to y-axis
        }
        clearGraphSearch();
        startVertex =
                extremeVertexConvex(extremeVertexConvex, dir, floodFillVisited,
                                    floodFillVisitColor, dummy, startVertex);
        clearGraphSearch();
        traverseStack.push_back(startVertex);
        while (!traverseStack.empty()) {
            int v = traverseStack.back();
            traverseStack.pop_back();

            if (haveVisitedVertex(v)) continue;
            markVertexVisited(v);

            // const auto& neighbors = adjacencyData[v];
            for (int vAdj : adjacencyData[v]) {
                if (haveVisitedVertex(vAdj)) continue;
                int edge = vertexPairsToEdges[v * numVertices + vAdj];
                auto [faceI_a, faceI_b] = facesForEdge[edge];
                Eigen::Vector3d f1a_f1b = f1a - f1b;
                Eigen::Vector3d f2a_f2b =
                        faceNormals[faceI_a] - faceNormals[faceI_b];

                double a2 = f1b.dot(faceNormals[faceI_b]);
                double b2 = f1a_f1b.dot(faceNormals[faceI_b]);
                double c2 = f2a_f2b.dot(f1b);
                double d2 = f1a_f1b.dot(f2a_f2b);
                double ab = a2 + b2;
                double ac = a2 + c2;
                double abcd = ab + c2 + d2;
                double minVal = std::min({a2, ab, ac, abcd});
                double maxVal = std::max({a2, ab, ac, abcd});
                bool are_edges_compatible_for_obb =
                        (minVal <= 0.0 && maxVal >= 0.0);

                if (are_edges_compatible_for_obb) {
                    if (edgeI <= edge) {
                        if (!isInternalEdge(edge)) {
                            compatibleEdges[edgeI].push_back(edge);
                        }

                        sidepodalVertices[edgeI * numVertices +
                                          edges[edge].first] = 1;
                        sidepodalVertices[edgeI * numVertices +
                                          edges[edge].second] = 1;
                        if (edgeI != edge) {
                            if (!isInternalEdge(edge)) {
                                compatibleEdges[edge].push_back(edgeI);
                            }
                            sidepodalVertices[edge * numVertices +
                                              edges[edgeI].first] = 1;
                            sidepodalVertices[edge * numVertices +
                                              edges[edgeI].second] = 1;
                        }
                    }
                    traverseStack.push_back(vAdj);
                }
            }
        }
    }

    // --------------------------------------------------------------------
    // 4) Test configurations where all three edges are on adjacent faces.
    // --------------------------------------------------------------------

    // Take advantage of spatial locality: start the search for the extreme
    // vertex from the extreme vertex that was found during the previous
    // iteration for the previous edge. This speeds up the search since edge
    // directions have some amount of spatial locality and the next extreme
    // vertex is often close to the previous one. Track two hint variables since
    // we are performing extreme vertex searches to two opposing directions at
    // the same time.
    int vHint1 = 0;
    int vHint2 = 0;
    int vHint3 = 0;
    int vHint4 = 0;
    int vHint1_b = 0;
    int vHint2_b = 0;
    int vHint3_b = 0;

    // Stores a memory of yet unvisited vertices that are common sidepodal
    // vertices to both currently chosen edges for current graph search.
    std::vector<int> traverseStackCommonSidepodals;
    traverseStackCommonSidepodals.reserve(numVertices);
    for (int edgeI : spatialEdgeOrder) {
        auto [faceI_a, faceI_b] = facesForEdge[edgeI];
        const Eigen::Vector3d& f1a = faceNormals[faceI_a];
        const Eigen::Vector3d& f1b = faceNormals[faceI_b];

        const auto& compatibleEdgesI = compatibleEdges[edgeI];
        Eigen::Vector3d baseDir = 0.5 * (f1a + f1b);

        for (int edgeJ : compatibleEdgesI) {
            if (edgeJ <= edgeI) continue;  // Remove symmetry.
            auto [faceJ_a, faceJ_b] = facesForEdge[edgeJ];
            const Eigen::Vector3d& f2a = faceNormals[faceJ_a];
            const Eigen::Vector3d& f2b = faceNormals[faceJ_b];

            // Compute search direction
            Eigen::Vector3d deadDir = 0.5 * (f2a + f2b);
            Eigen::Vector3d searchDir = baseDir.cross(deadDir);
            searchDir = searchDir.normalized();
            if (searchDir.norm() < 1e-9) {
                searchDir = f1a.cross(f2a);
                searchDir = searchDir.normalized();
                if (searchDir.norm() < 1e-9) {
                    searchDir =
                            (f1a.cross(Eigen::Vector3d(0, 1, 0))).normalized();
                }
            }

            double dummy;
            clearGraphSearch();
            vHint1 = extremeVertexConvex(extremeVertexConvex, searchDir,
                                         floodFillVisited, floodFillVisitColor,
                                         dummy, vHint1);
            clearGraphSearch();
            vHint2 = extremeVertexConvex(extremeVertexConvex, -searchDir,
                                         floodFillVisited, floodFillVisitColor,
                                         dummy, vHint2);

            int secondSearch = -1;
            if (sidepodalVertices[edgeJ * numVertices + vHint1]) {
                traverseStackCommonSidepodals.push_back(vHint1);
            } else {
                traverseStack.push_back(vHint1);
            }
            if (sidepodalVertices[edgeJ * numVertices + vHint2]) {
                traverseStackCommonSidepodals.push_back(vHint2);
            } else {
                secondSearch = vHint2;
            }

            // Bootstrap to a good vertex that is sidepodal to both edges.
            clearGraphSearch();
            while (!traverseStack.empty()) {
                int v = traverseStack.front();
                traverseStack.erase(traverseStack.begin());
                if (haveVisitedVertex(v)) continue;
                markVertexVisited(v);
                const auto& neighbors = adjacencyData[v];
                for (int vAdj : neighbors) {
                    if (!haveVisitedVertex(vAdj) &&
                        sidepodalVertices[edgeI * numVertices + vAdj]) {
                        if (sidepodalVertices[edgeJ * numVertices + vAdj]) {
                            traverseStack.clear();
                            if (secondSearch != -1) {
                                traverseStack.push_back(secondSearch);
                                secondSearch = -1;
                                markVertexVisited(vAdj);
                            }
                            traverseStackCommonSidepodals.push_back(vAdj);
                            break;
                        } else {
                            traverseStack.push_back(vAdj);
                        }
                    }
                }
            }

            clearGraphSearch();
            while (!traverseStackCommonSidepodals.empty()) {
                int v = traverseStackCommonSidepodals.back();
                traverseStackCommonSidepodals.pop_back();
                if (haveVisitedVertex(v)) continue;
                markVertexVisited(v);
                const auto& neighbors = adjacencyData[v];
                for (int vAdj : neighbors) {
                    int edgeK = vertexPairsToEdges[v * numVertices + vAdj];
                    int idxI = edgeI * numVertices + vAdj;
                    int idxJ = edgeJ * numVertices + vAdj;

                    if (isInternalEdge(edgeK)) continue;

                    if (sidepodalVertices[idxI] && sidepodalVertices[idxJ]) {
                        if (!haveVisitedVertex(vAdj)) {
                            traverseStackCommonSidepodals.push_back(vAdj);
                        }
                        if (edgeJ < edgeK) {
                            auto [faceK_a, faceK_b] = facesForEdge[edgeK];
                            const Eigen::Vector3d& f3a = faceNormals[faceK_a];
                            const Eigen::Vector3d& f3b = faceNormals[faceK_b];

                            std::vector<Eigen::Vector3d> n1 = {
                                    Eigen::Vector3d::Zero(),
                                    Eigen::Vector3d::Zero()};
                            std::vector<Eigen::Vector3d> n2 = {
                                    Eigen::Vector3d::Zero(),
                                    Eigen::Vector3d::Zero()};
                            std::vector<Eigen::Vector3d> n3 = {
                                    Eigen::Vector3d::Zero(),
                                    Eigen::Vector3d::Zero()};

                            constexpr double eps = 1e-4;
                            constexpr double angleEps = 1e-3;
                            int nSolutions = 0;

                            {
                                // Precompute intermediate vectors for
                                // polynomial coefficients.
                                Eigen::Vector3d a = f1b;
                                Eigen::Vector3d b = f1a - f1b;
                                Eigen::Vector3d c = f2b;
                                Eigen::Vector3d d = f2a - f2b;
                                Eigen::Vector3d e = f3b;
                                Eigen::Vector3d f = f3a - f3b;

                                // Compute polynomial coefficients.
                                double g = a.dot(c) * d.dot(e) -
                                           a.dot(d) * c.dot(e);
                                double h = a.dot(c) * d.dot(f) -
                                           a.dot(d) * c.dot(f);
                                double i = b.dot(c) * d.dot(e) -
                                           b.dot(d) * c.dot(e);
                                double j = b.dot(c) * d.dot(f) -
                                           b.dot(d) * c.dot(f);
                                double k = g * b.dot(e) - a.dot(e) * i;
                                double l = h * b.dot(e) + g * b.dot(f) -
                                           a.dot(f) * i - a.dot(e) * j;
                                double m = h * b.dot(f) - a.dot(f) * j;
                                double s = l * l - 4 * m * k;

                                // Handle degenerate or linear case.
                                if (std::abs(m) < 1e-5 || std::abs(s) < 1e-5) {
                                    double v = -k / l;
                                    double t = -(g + h * v) / (i + j * v);
                                    double u = -(c.dot(e) + c.dot(f) * v) /
                                               (d.dot(e) + d.dot(f) * v);
                                    nSolutions = 0;

                                    // If we happened to divide by zero above,
                                    // the following checks handle them.
                                    if (v >= -eps && t >= -eps && u >= -eps &&
                                        v <= 1.0 + eps && t <= 1.0 + eps &&
                                        u <= 1.0 + eps) {
                                        n1[0] = (a + b * t).normalized();
                                        n2[0] = (c + d * u).normalized();
                                        n3[0] = (e + f * v).normalized();
                                        if (std::abs(n1[0].dot(n2[0])) <
                                                    angleEps &&
                                            std::abs(n1[0].dot(n3[0])) <
                                                    angleEps &&
                                            std::abs(n2[0].dot(n3[0])) <
                                                    angleEps) {
                                            nSolutions = 1;
                                        } else {
                                            nSolutions = 0;
                                        }
                                    }
                                } else {
                                    // Discriminant negative: no solutions for v
                                    if (s < 0.0) {
                                        nSolutions = 0;
                                    } else {
                                        double sgnL = l < 0 ? -1.0 : 1.0;
                                        double V1 = -(l + sgnL * std::sqrt(s)) /
                                                    (2.0 * m);
                                        double V2 = k / (m * V1);
                                        double T1 =
                                                -(g + h * V1) / (i + j * V1);
                                        double T2 =
                                                -(g + h * V2) / (i + j * V2);
                                        double U1 =
                                                -(c.dot(e) + c.dot(f) * V1) /
                                                (d.dot(e) + d.dot(f) * V1);
                                        double U2 =
                                                -(c.dot(e) + c.dot(f) * V2) /
                                                (d.dot(e) + d.dot(f) * V2);

                                        if (V1 >= -eps && T1 >= -eps &&
                                            U1 >= -eps && V1 <= 1.0 + eps &&
                                            T1 <= 1.0 + eps &&
                                            U1 <= 1.0 + eps) {
                                            n1[nSolutions] =
                                                    (a + b * T1).normalized();
                                            n2[nSolutions] =
                                                    (c + d * U1).normalized();
                                            n3[nSolutions] =
                                                    (e + f * V1).normalized();

                                            if (std::abs(n1[nSolutions].dot(
                                                        n2[nSolutions])) <
                                                        angleEps &&
                                                std::abs(n1[nSolutions].dot(
                                                        n3[nSolutions])) <
                                                        angleEps &&
                                                std::abs(n2[nSolutions].dot(
                                                        n3[nSolutions])) <
                                                        angleEps)
                                                ++nSolutions;
                                        }
                                        if (V2 >= -eps && T2 >= -eps &&
                                            U2 >= -eps && V2 <= 1.0 + eps &&
                                            T2 <= 1.0 + eps &&
                                            U2 <= 1.0 + eps) {
                                            n1[nSolutions] =
                                                    (a + b * T2).normalized();
                                            n2[nSolutions] =
                                                    (c + d * U2).normalized();
                                            n3[nSolutions] =
                                                    (e + f * V2).normalized();
                                            if (std::abs(n1[nSolutions].dot(
                                                        n2[nSolutions])) <
                                                        angleEps &&
                                                std::abs(n1[nSolutions].dot(
                                                        n3[nSolutions])) <
                                                        angleEps &&
                                                std::abs(n2[nSolutions].dot(
                                                        n3[nSolutions])) <
                                                        angleEps)
                                                ++nSolutions;
                                        }
                                        if (s < 1e-4 && nSolutions == 2) {
                                            nSolutions = 1;
                                        }
                                    }
                                }
                            }

                            for (int s = 0; s < nSolutions; ++s) {
                                const auto& hullVi = hullV[edges[edgeI].first];
                                const auto& hullVj = hullV[edges[edgeJ].first];
                                const auto& hullVk = hullV[edges[edgeK].first];
                                const auto& n1_ = n1[s];
                                const auto& n2_ = n2[s];
                                const auto& n3_ = n3[s];

                                // Compute the most extreme points in each
                                // direction.
                                double maxN1 = n1_.dot(hullVi);
                                double maxN2 = n2_.dot(hullVj);
                                double maxN3 = n3_.dot(hullVk);
                                double minN1 =
                                        std::numeric_limits<double>::infinity();
                                double minN2 =
                                        std::numeric_limits<double>::infinity();
                                double minN3 =
                                        std::numeric_limits<double>::infinity();

                                const auto& antipodalI =
                                        antipodalPointsForEdge[edgeI];
                                const auto& antipodalJ =
                                        antipodalPointsForEdge[edgeJ];
                                const auto& antipodalK =
                                        antipodalPointsForEdge[edgeK];

                                // Determine the minimum projections along each
                                // axis over respective antipodal sets.
                                for (int vIdx : antipodalI) {
                                    minN1 = std::min(minN1,
                                                     n1_.dot(hullV[vIdx]));
                                }
                                for (int vIdx : antipodalJ) {
                                    minN2 = std::min(minN2,
                                                     n2_.dot(hullV[vIdx]));
                                }
                                for (int vIdx : antipodalK) {
                                    minN3 = std::min(minN3,
                                                     n3_.dot(hullV[vIdx]));
                                }

                                // Compute volume based on extents in the three
                                // principal directions.
                                double extent0 = maxN1 - minN1;
                                double extent1 = maxN2 - minN2;
                                double extent2 = maxN3 - minN3;
                                double volume = extent0 * extent1 * extent2;

                                // Update the minimum oriented bounding box if a
                                // smaller volume is found.
                                if (volume < minVolume) {
                                    // Update rotation matrix columns.
                                    minOBB.R_.col(0) = n1_;
                                    minOBB.R_.col(1) = n2_;
                                    minOBB.R_.col(2) = n3_;

                                    // Update extents.
                                    minOBB.extent_(0) = extent0;
                                    minOBB.extent_(1) = extent1;
                                    minOBB.extent_(2) = extent2;

                                    // Compute the center of the OBB using
                                    // midpoints along each axis.
                                    minOBB.center_ =
                                            (minN1 + 0.5 * extent0) * n1_ +
                                            (minN2 + 0.5 * extent1) * n2_ +
                                            (minN3 + 0.5 * extent2) * n3_;

                                    minVolume = volume;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // --------------------------------------------------------------------
    // 5) Test all configurations where two edges are on opposing faces,
    //    ,and the third one is on a face adjacent to the two.
    // --------------------------------------------------------------------

    {
        std::vector<int> antipodalEdges;
        antipodalEdges.reserve(128);
        std::vector<Eigen::Vector3d> antipodalEdgeNormals;
        antipodalEdgeNormals.reserve(128);

        // Iterate over each edgeI in spatialEdgeOrder.
        for (int edgeI : spatialEdgeOrder) {
            // Cache face indices and normals for edgeI.
            auto [faceI_a, faceI_b] = facesForEdge[edgeI];
            const Eigen::Vector3d& f1a = faceNormals[faceI_a];
            const Eigen::Vector3d& f1b = faceNormals[faceI_b];

            antipodalEdges.clear();
            antipodalEdgeNormals.clear();

            // Iterate over vertices antipodal to edgeI.
            const auto& antipodalsForI = antipodalPointsForEdge[edgeI];
            for (int antipodalVertex : antipodalsForI) {
                const auto& neighbors = adjacencyData[antipodalVertex];
                for (int vAdj : neighbors) {
                    if (vAdj < antipodalVertex) continue;

                    int edgeIndex = antipodalVertex * numVertices + vAdj;
                    int edge = vertexPairsToEdges[edgeIndex];

                    if (edgeI > edge) continue;
                    if (isInternalEdge(edge)) continue;

                    auto [faceJ_a, faceJ_b] = facesForEdge[edge];
                    const Eigen::Vector3d& f2a = faceNormals[faceJ_a];
                    const Eigen::Vector3d& f2b = faceNormals[faceJ_b];

                    Eigen::Vector3d n;

                    bool areCompatibleOpposingEdges = false;
                    constexpr double tooCloseToFaceEpsilon = 1e-4;

                    Eigen::Matrix3d A;
                    A.col(0) = f2b;
                    A.col(1) = f1a - f1b;
                    A.col(2) = f2a - f2b;
                    Eigen::ColPivHouseholderQR<Eigen::Matrix3d> solver(A);
                    Eigen::Vector3d x = solver.solve(-f1b);
                    double c = x(0);
                    double t = x(1);
                    double cu = x(2);

                    if (c <= 0.0 || t < 0.0 || t > 1.0) {
                        areCompatibleOpposingEdges = false;
                    } else {
                        double u = cu / c;
                        if (t < tooCloseToFaceEpsilon ||
                            t > 1.0 - tooCloseToFaceEpsilon ||
                            u < tooCloseToFaceEpsilon ||
                            u > 1.0 - tooCloseToFaceEpsilon) {
                            areCompatibleOpposingEdges = false;
                        } else {
                            if (cu < 0.0 || cu > c) {
                                areCompatibleOpposingEdges = false;
                            } else {
                                n = f1b + (f1a - f1b) * t;
                                areCompatibleOpposingEdges = true;
                            }
                        }
                    }

                    if (areCompatibleOpposingEdges) {
                        antipodalEdges.push_back(edge);
                        antipodalEdgeNormals.push_back(n.normalized());
                    }
                }
            }

            auto moveSign = [](double& dst, double& src) {
                if (src < 0.0) {
                    dst = -dst;
                    src = -src;
                }
            };

            const auto& compatibleEdgesI = compatibleEdges[edgeI];
            for (int edgeJ : compatibleEdgesI) {
                for (size_t k = 0; k < antipodalEdges.size(); ++k) {
                    int edgeK = antipodalEdges[k];

                    const Eigen::Vector3d& n1 = antipodalEdgeNormals[k];
                    double minN1 = n1.dot(hullV[edges[edgeK].first]);
                    double maxN1 = n1.dot(hullV[edges[edgeI].first]);

                    // Test all mutual compatible edges.
                    auto [faceK_a, faceK_b] = facesForEdge[edgeJ];
                    const Eigen::Vector3d& f3a = faceNormals[faceK_a];
                    const Eigen::Vector3d& f3b = faceNormals[faceK_b];

                    double num = n1.dot(f3b);
                    double den = n1.dot(f3b - f3a);
                    moveSign(num, den);

                    constexpr double epsilon = 1e-4;
                    if (den < epsilon) {
                        num = (std::abs(num) < 1e-4) ? 0.0 : -1.0;
                        den = 1.0;
                    }

                    if (num >= den * -epsilon && num <= den * (1.0 + epsilon)) {
                        double v = num / den;
                        Eigen::Vector3d n3 =
                                (f3b + (f3a - f3b) * v).normalized();
                        Eigen::Vector3d n2 = n3.cross(n1).normalized();

                        double maxN2, minN2;
                        clearGraphSearch();
                        int hint = extremeVertexConvex(
                                extremeVertexConvex, n2, floodFillVisited,
                                floodFillVisitColor, maxN2,
                                (k == 0) ? vHint1 : vHint1_b);
                        if (k == 0) {
                            vHint1 = vHint1_b = hint;
                        } else {
                            vHint1_b = hint;
                        }

                        clearGraphSearch();
                        hint = extremeVertexConvex(
                                extremeVertexConvex, -n2, floodFillVisited,
                                floodFillVisitColor, minN2,
                                (k == 0) ? vHint2 : vHint2_b);
                        if (k == 0) {
                            vHint2 = vHint2_b = hint;
                        } else {
                            vHint2_b = hint;
                        }

                        minN2 = -minN2;

                        double maxN3 = n3.dot(hullV[edges[edgeJ].first]);
                        double minN3 = std::numeric_limits<double>::infinity();

                        // If there are very few antipodal vertices, do a
                        // very tight loop and just iterate over each.
                        const auto& antipodalsEdge =
                                antipodalPointsForEdge[edgeJ];
                        if (antipodalsEdge.size() < 20) {
                            for (int vIdx : antipodalsEdge) {
                                minN3 = std::min(minN3, n3.dot(hullV[vIdx]));
                            }
                        } else {
                            // Otherwise perform a spatial locality
                            // exploiting graph search.
                            clearGraphSearch();
                            hint = extremeVertexConvex(
                                    extremeVertexConvex, -n3, floodFillVisited,
                                    floodFillVisitColor, minN3,
                                    (k == 0) ? vHint3 : vHint3_b);

                            if (k == 0) {
                                vHint3 = vHint3_b = hint;
                            } else {
                                vHint3_b = hint;
                            }

                            minN3 = -minN3;
                        }

                        double volume = (maxN1 - minN1) * (maxN2 - minN2) *
                                        (maxN3 - minN3);
                        if (volume < minVolume) {
                            minOBB.R_.col(0) = n1;
                            minOBB.R_.col(1) = n2;
                            minOBB.R_.col(2) = n3;
                            minOBB.extent_(0) = (maxN1 - minN1);
                            minOBB.extent_(1) = (maxN2 - minN2);
                            minOBB.extent_(2) = (maxN3 - minN3);
                            minOBB.center_ = 0.5 * ((minN1 + maxN1) * n1 +
                                                    (minN2 + maxN2) * n2 +
                                                    (minN3 + maxN3) * n3);
                            minVolume = volume;
                        }
                    }
                }
            }
        }
    }

    // --------------------------------------------------------------------
    // 6) Test all configurations where two edges are on the same face (OBB
    //    aligns with a face of the convex hull)
    // --------------------------------------------------------------------
    {
        // Preallocate vectors to avoid frequent reallocations.
        std::vector<int> antipodalEdges;
        antipodalEdges.reserve(128);
        std::vector<Eigen::Vector3d> antipodalEdgeNormals;
        antipodalEdgeNormals.reserve(128);

        for (int faceIdx : spatialFaceOrder) {
            const Eigen::Vector3d& n1 = faceNormals[faceIdx];

            // Find two edges on the face. Since we have flexibility to
            // choose from multiple edges of the same face, choose two that
            // are possibly most opposing to each other, in the hope that
            // their sets of sidepodal edges are most mutually exclusive as
            // possible, speeding up the search below.
            int e1 = -1;
            const auto& tri = hullT[faceIdx];
            int v0 = tri(2);
            for (int j = 0; j < 3; ++j) {
                int v1 = tri(j);
                int e = vertexPairsToEdges[v0 * numVertices + v1];
                if (!isInternalEdge(e)) {
                    e1 = e;
                    break;
                }
                v0 = v1;
            }

            if (e1 == -1) continue;

            // Compute minN1 either by scanning antipodal points or using
            // ExtremeVertexConvex.
            double maxN1 = n1.dot(hullV[edges[e1].first]);
            double minN1 = std::numeric_limits<double>::infinity();
            const auto& antipodals = antipodalPointsForEdge[e1];
            if (antipodals.size() < 20) {
                minN1 = std::numeric_limits<double>::infinity();
                for (int vIdx : antipodals) {
                    minN1 = std::min(minN1, n1.dot(hullV[vIdx]));
                }
            } else {
                clearGraphSearch();
                vHint4 = extremeVertexConvex(
                        extremeVertexConvex, -n1, floodFillVisited,
                        floodFillVisitColor, minN1, vHint4);
                minN1 = -minN1;
            }

            // Check edges compatible with e1.
            const auto& compatibleEdgesI = compatibleEdges[e1];
            for (int edgeK : compatibleEdgesI) {
                auto [faceK_a, faceK_b] = facesForEdge[edgeK];
                const Eigen::Vector3d& f3a = faceNormals[faceK_a];
                const Eigen::Vector3d& f3b = faceNormals[faceK_b];

                // Is edge3 compatible with direction n?
                double num = n1.dot(f3b);
                double den = n1.dot(f3b - f3a);
                double v;
                constexpr double epsilon = 1e-4;
                if (std::abs(den) >= epsilon) {
                    v = num / den;
                } else {
                    v = (std::abs(num) < epsilon) ? 0.0 : -1.0;
                }

                if (v >= -epsilon && v <= 1.0 + epsilon) {
                    Eigen::Vector3d n3 = (f3b + (f3a - f3b) * v).normalized();
                    Eigen::Vector3d n2 = n3.cross(n1).normalized();

                    double maxN2, minN2;
                    clearGraphSearch();
                    vHint1 = extremeVertexConvex(
                            extremeVertexConvex, n2, floodFillVisited,
                            floodFillVisitColor, maxN2, vHint1);
                    clearGraphSearch();
                    vHint2 = extremeVertexConvex(
                            extremeVertexConvex, -n2, floodFillVisited,
                            floodFillVisitColor, minN2, vHint2);
                    minN2 = -minN2;

                    double maxN3 = n3.dot(hullV[edges[edgeK].first]);
                    double minN3 = std::numeric_limits<double>::infinity();

                    // If there are very few antipodal vertices, do a very
                    // tight loop and just iterate over each.
                    const auto& antipodalsEdge = antipodalPointsForEdge[edgeK];
                    if (antipodalsEdge.size() < 20) {
                        for (int vIdx : antipodalsEdge) {
                            minN3 = std::min(minN3, n3.dot(hullV[vIdx]));
                        }
                    } else {
                        clearGraphSearch();
                        vHint3 = extremeVertexConvex(
                                extremeVertexConvex, -n3, floodFillVisited,
                                floodFillVisitColor, minN3, vHint3);
                        minN3 = -minN3;
                    }

                    double volume =
                            (maxN1 - minN1) * (maxN2 - minN2) * (maxN3 - minN3);
                    if (volume < minVolume) {
                        minOBB.R_.col(0) = n1;
                        minOBB.R_.col(1) = n2;
                        minOBB.R_.col(2) = n3;
                        minOBB.extent_(0) = (maxN1 - minN1);
                        minOBB.extent_(1) = (maxN2 - minN2);
                        minOBB.extent_(2) = (maxN3 - minN3);
                        minOBB.center_ = 0.5 * ((minN1 + maxN1) * n1 +
                                                (minN2 + maxN2) * n2 +
                                                (minN3 + maxN3) * n3);
                        assert(volume > 0.0);
                        minVolume = volume;
                    }
                }
            }
        }
    }

    // Final check to ensure right-handed coordinate frame
    if (minOBB.R_.col(0).cross(minOBB.R_.col(1)).dot(minOBB.R_.col(2)) < 0.0) {
        minOBB.R_.col(2) = -minOBB.R_.col(2);
    }
    mapOBBToClosestIdentity(minOBB);
    return minOBB;
}

void OrientedBoundingBox::computeR_n(double theta1,
                                     double phi,
                                     const Eigen::Matrix3d& R_x,
                                     const Eigen::RowVector3d& normal2,
                                     Eigen::Matrix3d& R_n) {
    double ct = std::cos(theta1);
    double st = std::sin(theta1);
    double cp = std::cos(phi);
    double sp = std::sin(phi);

    Eigen::RowVector3d n1_Rx(ct, st, 0);
    n1_Rx = n1_Rx * R_x;
    Eigen::RowVector3d n2_Rx(-st * sp, ct * sp, -ct * cp);
    n2_Rx = n2_Rx * R_x / std::sqrt(ct * ct + st * st * sp * sp);
    if (n2_Rx.dot(normal2) < 0) {
        n2_Rx = -n2_Rx;
    }
    Eigen::RowVector3d n3_Rx = n1_Rx.cross(n2_Rx);
    R_n.row(0) = n1_Rx;
    R_n.row(1) = n2_Rx;
    R_n.row(2) = n3_Rx;
}

std::vector<double> OrientedBoundingBox::findRealRoots(
        const Eigen::VectorXd& coeffs) {
    // Determine polynomial degree by ignoring leading zeros
    int n = coeffs.size();
    int first_nonzero = -1;
    for (int i = 0; i < n; ++i) {
        if (coeffs[i] != 0.0) {
            first_nonzero = i;
            break;
        }
    }

    if (first_nonzero == -1) {
        throw std::runtime_error(
                "All coefficients are zero. Polynomial is not defined.");
    }

    // Degree is size - first_nonzero - 1
    // For example, if coeffs.size() = 8 and first_nonzero = 0, degree = 7
    int degree = n - first_nonzero - 1;
    if (degree < 1) {
        throw std::runtime_error("Polynomial degree must be at least 1.");
    }

    // Extract the relevant coefficients of the actual polynomial
    // After removing leading zeros, the polynomial looks like:
    // coeffs[first_nonzero]*x^degree + ... + coeffs[n-1]*x^0
    Eigen::VectorXd poly = coeffs.segment(first_nonzero, n - first_nonzero);

    double leading = poly[0];
    if (leading == 0.0) {
        throw std::runtime_error(
                "Leading coefficient cannot be zero after trimming.");
    }

    // Make polynomial monic
    Eigen::VectorXd monic = poly / leading;
    // monic: x^degree + monic[1]*x^(degree-1) + ... + monic[degree]

    // Construct companion matrix of size degree x degree
    // The last row: [ -monic[degree], ..., -monic[1] ]
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(degree, degree);

    // Fill the superdiagonal with ones
    for (int i = 0; i < degree - 1; ++i) {
        C(i, i + 1) = 1.0;
    }

    // Last row
    for (int i = 0; i < degree; ++i) {
        C(degree - 1, i) = -monic[degree - i];
    }

    // Compute eigenvalues
    Eigen::EigenSolver<Eigen::MatrixXd> es(C);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("Eigenvalue decomposition failed.");
    }

    Eigen::VectorXcd eigenvalues = es.eigenvalues();

    // Extract real parts of roots
    std::vector<double> realRoots;
    for (int i = 0; i < degree; ++i) {
        if (std::abs(eigenvalues[i].imag()) < 1e-10) {
            realRoots.push_back(eigenvalues[i].real());
        }
    }

    return realRoots;
}

std::vector<int> OrientedBoundingBox::findEquivalentExtrema(
        const Eigen::RowVector3d& normal,
        int todo,
        const Eigen::MatrixXi& allFaces,
        const std::vector<std::vector<int>>& allNodes,
        const Eigen::MatrixXd& X,
        double tol,
        int prev) {
    int numNodes = static_cast<int>(allNodes.size());
    std::vector<bool> done(numNodes, false);
    std::vector<bool> keep(numNodes, false);

    // Initialize the queue with the starting node 'todo'
    std::queue<int> nodeQueue;
    nodeQueue.push(todo);
    done[todo] = true;
    keep[todo] = true;

    if (prev >= 0 && prev < numNodes) {
        done[prev] = true;
    }

    // Compute the limit value for the dot product
    double dp_todo = X.row(todo).dot(normal);
    double limit = dp_todo - tol * std::max(std::abs(dp_todo), 1.0);

    // Breadth-first traversal to find equivalent extrema
    while (!nodeQueue.empty()) {
        int currentNode = nodeQueue.front();
        nodeQueue.pop();

        // Get the faces connected to the current node
        const std::vector<int>& facesToCheck = allNodes[currentNode];

        // Iterate over each face
        for (int faceIdx : facesToCheck) {
            // Get the nodes of the face
            const Eigen::Vector3i& face = allFaces.row(faceIdx);

            // Check each node in the face
            for (int i = 0; i < 3; ++i) {
                int nodeIdx = face[i];

                if (!done[nodeIdx]) {
                    double dp = X.row(nodeIdx).dot(normal);
                    if (dp >= limit) {
                        keep[nodeIdx] = true;
                        nodeQueue.push(nodeIdx);
                    }
                    done[nodeIdx] = true;
                }
            }
        }
    }

    // Collect the nodes that meet the condition
    std::vector<int> nodes;
    for (int i = 0; i < numNodes; ++i) {
        if (keep[i]) {
            nodes.push_back(i);
        }
    }

    return nodes;
}

void OrientedBoundingBox::findRow(const std::vector<Eigen::Vector2i>& M,
                                  const Eigen::Vector2i& row,
                                  int& idx,
                                  int& insert) {
    if (M.empty()) {
        idx = -1;
        insert = 0;
        return;
    }
    int m = 2;

    int low = 0;
    int high = static_cast<int>(M.size());

    // Binary search loop
    while (low < high) {                   // Changed condition to low < high
        int mid = low + (high - low) / 2;  // More efficient and avoids overflow
        bool isEqual = true;

        for (int i = 0; i < m; ++i) {  // 0-based indexing in C++
            if (row(i) < M[mid](i)) {  // Removed -1 to access correctly
                high = mid;
                isEqual = false;
                break;
            } else if (row(i) > M[mid](i)) {
                low = mid + 1;
                isEqual = false;
                break;
            }
            // If equal, continue to next column
        }
        // If all columns are equal
        if (isEqual) {
            idx = mid;
            insert = -1;
            return;  // Found the row, can return early
        }
    }

    idx = -1;      // Row not found
    insert = low;  // Position to insert
}

void OrientedBoundingBox::nodeToFaces(
        const Eigen::MatrixXi& hullFaces,
        const Eigen::MatrixXd& X,
        std::vector<std::vector<int>>& hullNodes) {
    int numFaces = (int)hullFaces.rows();
    int maxNode = hullFaces.maxCoeff();
    hullNodes.assign(maxNode + 1, std::vector<int>());

    auto normalToFaceFunc = [&](const Eigen::Vector3i& f) {
        return normalToFace(f, X);
    };

    auto normalizedCrossProduct = [&](const Eigen::Vector3d& u,
                                      const Eigen::Vector3d& v) {
        Eigen::Vector3d c = u.cross(v);
        double n = c.norm();
        if (n > 0.0) c /= n;
        return c;
    };

    auto computeTripleProduct = [&](int newNode, const Eigen::Vector3d& center,
                                    const Eigen::Vector3d& center_l,
                                    const Eigen::Vector3d& normal,
                                    const Eigen::Vector3d& normal_l) {
        Eigen::Vector3d mean_normal = (normal - normal_l) / 2.0;
        Eigen::Vector3d vec1 = center - X.row(newNode).transpose();
        Eigen::Vector3d vec2 = center_l - X.row(newNode).transpose();
        Eigen::Vector3d cross_vec = normalizedCrossProduct(vec1, vec2);
        return cross_vec.dot(mean_normal);
    };

    for (int k = 0; k < 3 * numFaces; ++k) {
        int faceIndex = k / 3;  // 0-based face index
        int localIdx = k % 3;

        static Eigen::Vector3i currentFace;
        static Eigen::Vector3d center;
        static Eigen::Vector3d normal;

        if (localIdx == 0) {
            currentFace = hullFaces.row(faceIndex);
            center = (X.row(currentFace(0)) + X.row(currentFace(1)) +
                      X.row(currentFace(2))) /
                     3.0;
            normal = -normalToFaceFunc(currentFace);
        }

        int newNode = currentFace(localIdx);
        std::vector<int>& faces = hullNodes[newNode];

        // Replicate MATLAB logic
        // l = [];
        // for l=1:length(faces)
        //    ...
        // end
        int forward_break_at = -1;  // If we break, store the iteration here
        double triple_product;
        for (int l = 0; l < (int)faces.size(); ++l) {
            int face_idx = faces[l];
            Eigen::Vector3i f = hullFaces.row(face_idx);
            Eigen::Vector3d center_l =
                    (X.row(f(0)) + X.row(f(1)) + X.row(f(2))) / 3.0;
            Eigen::Vector3d normal_l = normalToFaceFunc(f);

            triple_product = computeTripleProduct(newNode, center, center_l,
                                                  normal, normal_l);
            if (triple_product > 0.0) {
                forward_break_at = l;  // We break at iteration l (0-based)
                break;
            }
        }

        // Determine final l in MATLAB (1-based) to insert
        int l_matlab;
        if (faces.empty()) {
            // isempty(l) => l was never assigned since no loop ran
            l_matlab = 1;
        } else {
            if (forward_break_at == -1) {
                // No break in forward loop
                // After loop, l in MATLAB = length(faces)
                // Then no isempty(l), l != 0 => l = l+1 at end
                // final l_matlab = length(faces) + 1
                l_matlab = (int)faces.size() + 1;
            } else {
                // We broke at iteration forward_break_at (0-based)
                // MATLAB: l was (forward_break_at+1) when break triggered
                // then l = l - 1 => l_matlab = forward_break_at (1-based:
                // l_matlab = forward_break_at+1 - 1 = forward_break_at) Since
                // forward_break_at is 0-based, l_matlab = forward_break_at
                // (0-based) + 1 - 1 = forward_break_at Wait carefully: If break
                // at iteration l (0-based), MATLAB l= l (1-based) at break =>
                // l= l_break -1 afterwards 1-based l at break was
                // (forward_break_at+1) l = l -1 => l_matlab =
                // (forward_break_at+1)-1 = forward_break_at
                l_matlab =
                        forward_break_at;  // now forward_break_at is 0-based,
                                           // so l_matlab is 1 less than actual?
                // Actually, if forward_break_at = 0 (first iteration),
                // l_matlab=0 after l=l-1 in MATLAB is correct.

                if (l_matlab == 0) {
                    // reverse search
                    bool reverse_break = false;
                    for (int rl = (int)faces.size(); rl >= 1; rl--) {
                        int face_idx = faces[rl - 1];
                        Eigen::Vector3i f = hullFaces.row(face_idx);
                        Eigen::Vector3d center_l =
                                (X.row(f(0)) + X.row(f(1)) + X.row(f(2))) / 3.0;
                        Eigen::Vector3d normal_l = normalToFaceFunc(f);
                        triple_product = computeTripleProduct(
                                newNode, center, center_l, normal, normal_l);

                        if (triple_product < 0.0) {
                            // l = l+1 in MATLAB
                            // rl is 1-based here, so l_matlab = rl+1
                            l_matlab = rl + 1;
                            reverse_break = true;
                            break;
                        }
                    }
                    if (!reverse_break) {
                        // no break in reverse => l_matlab stays 0
                        // After reverse loop: l = l-1; l = l+1; cancel out =>
                        // still l=0 final step l=l+1 => l=1
                        l_matlab = 1;
                    } else {
                        // if we did break in reverse:
                        // after reverse search: l=l-1; l=l+1; cancel out, so
                        // l_matlab remains rl+1 final l=l+1 => l_matlab =
                        // (rl+1)+1 = rl+2 but we must follow code closely:
                        // Check original MATLAB code carefully. It does:
                        // if l==0
                        //   for l=length(faces):-1:1 ...
                        //       if triple_product < 0
                        //          l=l+1; break;
                        //       end
                        //   end
                        //   l=l-1;
                        // end
                        // l=l+1 at the end outside if
                        //
                        // Steps if break in reverse at rl:
                        // l after break in reverse = rl+1
                        // then l=l-1 => l=rl
                        // l=l+1 at end => l=rl+1
                        l_matlab = l_matlab - 1;  // l=l-1
                        l_matlab = l_matlab + 1;  // l=l+1 at end
                        // So final l_matlab = rl+1 still.
                    }
                } else {
                    // if l_matlab != 0
                    // just l=l+1 at the end
                    l_matlab = l_matlab + 1;
                }
            }
        }

        // Now l_matlab is the final 1-based insertion position
        // In C++ 0-based indexing:
        int insert_pos = l_matlab - 1;

        // Bound checks
        if (insert_pos < 0) insert_pos = 0;
        if (insert_pos > (int)faces.size()) insert_pos = (int)faces.size();

        faces.insert(faces.begin() + insert_pos, faceIndex);
    }
}

// Main function to list edges from hull faces
void OrientedBoundingBox::listEdges(const Eigen::MatrixXi& hullFaces,
                                    const Eigen::MatrixXd& X,
                                    Eigen::MatrixXd& hullEdges) {
    struct EdgeInfo {
        int origin;               // Origin node index
        int tip;                  // Tip node index
        int leftFace;             // Left face index (0 if none)
        int rightFace;            // Right face index (0 if none)
        Eigen::Vector3d edgeVec;  // Normalized edge vector
        // Comparator for sorting edges based on origin and tip
        bool operator<(const EdgeInfo& other) const {
            if (origin != other.origin) return origin < other.origin;
            return tip < other.tip;
        }
    };

    // Initialize an empty list of edges
    std::vector<EdgeInfo> edgeList;
    Eigen::Vector3d normal;
    Eigen::Vector3i face;
    // Iterate through each face and its three edges
    for (int k = 0; k < 3 * hullFaces.rows(); ++k) {
        if (k % 3 == 0) {
            face = hullFaces.row(int(k / 3));
            normal = -normalToFace(face, X);
        }
        // Determine the two nodes forming the current edge
        int node1 = face(k % 3);
        int node2 = face((k + 1) % 3);
        // Sort the node indices to ensure consistency
        Eigen::Vector2i newEdge;
        newEdge << std::min(node1, node2), std::max(node1, node2);
        // Prepare a vector of existing edges' origin and tip for findRow
        std::vector<Eigen::Vector2i> existingEdges;
        for (const auto& edge : edgeList) {
            existingEdges.emplace_back(Eigen::Vector2i(edge.origin, edge.tip));
        }
        // Find if the current edge already exists
        int idx = -1;
        int insertPos = 0;
        findRow(existingEdges, newEdge, idx, insertPos);
        if (idx > -1) {
            // Edge already exists
            bool isRight = (edgeList[idx].rightFace == -1);
            if (isRight) {
                edgeList[idx].rightFace = int(k / 3);
            } else {
                edgeList[idx].leftFace = int(k / 3);
            }

            // If both faces are already assigned, you might want to handle
            // duplicates
        } else {
            // Edge does not exist; insert a new edge
            // Compute the normalized edge vector
            Eigen::Vector3d vecEdge = X.row(newEdge(1)) - X.row(newEdge(0));
            double normEdge = vecEdge.norm();
            if (normEdge < 1e-12) {
                std::cerr << "Warning: Edge with zero length detected between "
                             "nodes "
                          << newEdge(0) << " and " << newEdge(1)
                          << ". Skipping.\n";
                continue;  // Skip zero-length edges
            }
            Eigen::Vector3d normalizedEdge = vecEdge / normEdge;
            // Determine if the face is on the right side
            // Compute cross product and scalar triple product
            // crossNorm(normal, vecEdge, (X(face(2), :) - X(face(0), :))')
            // Note: Adjust node indices for 0-based indexing
            Eigen::Vector3d vecA = normal.cross(vecEdge).normalized();
            Eigen::Vector3d vecB =
                    X.row(face((k + 2) % 3)) - X.row(face(k % 3));
            double tripleProduct = vecA.dot(vecB);
            bool isRight = (tripleProduct < 0);
            // Create a new EdgeInfo
            EdgeInfo newEdgeInfo;
            newEdgeInfo.origin = newEdge(0);
            newEdgeInfo.tip = newEdge(1);
            newEdgeInfo.leftFace = -1;
            newEdgeInfo.rightFace = -1;
            if (isRight) {
                newEdgeInfo.rightFace = int(k / 3);
            } else {
                newEdgeInfo.leftFace = int(k / 3);
            }
            // newEdgeInfo.leftFace =
            //         isRight ? 0 : int(k / 3);  // Left face if not right
            // newEdgeInfo.rightFace =
            //         isRight ? int(k / 3) : 0;  // Right face if isRight
            newEdgeInfo.edgeVec = normalizedEdge;
            // Insert the new edge at the determined position
            if (insertPos < 0 ||
                insertPos > static_cast<int>(edgeList.size())) {
                std::cerr << "Error: Invalid insertion position " << insertPos
                          << " for edge (" << newEdge(0) << ", " << newEdge(1)
                          << ").\n";
                assert(false);  // Handle according to your error policy
            }
            edgeList.insert(edgeList.begin() + insertPos, newEdgeInfo);
        }
    }
    // After processing all edges, convert the edgeList to Eigen::MatrixXd
    // Each edge has 7 components: origin, tip, leftFace, rightFace, edgeVec.x,
    // edgeVec.y, edgeVec.z
    hullEdges.resize(edgeList.size(), 7);
    for (size_t i = 0; i < edgeList.size(); ++i) {
        hullEdges(i, 0) = edgeList[i].origin;
        hullEdges(i, 1) = edgeList[i].tip;
        hullEdges(i, 2) = edgeList[i].leftFace;
        hullEdges(i, 3) = edgeList[i].rightFace;
        hullEdges(i, 4) = edgeList[i].edgeVec(0);
        hullEdges(i, 5) = edgeList[i].edgeVec(1);
        hullEdges(i, 6) = edgeList[i].edgeVec(2);
    }
}

void OrientedBoundingBox::computeTheta(
        const Eigen::RowVector3d& e1,
        const Eigen::RowVector3d& e2,
        double tol,
        const Eigen::Matrix<double, 2, 3>& normal1,
        const Eigen::Matrix<double, 2, 3>& normal2,
        const Eigen::RowVector3d& mean_normal1,
        double& phi,
        Eigen::Matrix3d& R_x,
        Eigen::VectorXd& theta1_min,
        Eigen::VectorXd& theta1_max,
        int& rc) {
    // Computation of the frame axes and angle
    Eigen::RowVector3d c = e2.cross(e1);  // Cross product

    if (c.norm() < tol * 1e-1) {
        c << 0, e1(2), -e1(1);
        if (c.norm() < tol * 1e-1) {
            c << e1(1), -e1(0), 0;
        }
    }

    Eigen::RowVector3d x = c.normalized();
    c = e1.cross(x);
    Eigen::RowVector3d y = c.normalized();
    Eigen::RowVector3d z = e1;

    // Construct R_x matrix
    R_x.row(0) = x;
    R_x.row(1) = y;
    R_x.row(2) = z;

    // Compute phi
    double arg = (y.cross(e2)).dot(x);
    // Ensure the argument of asin is within [-1, 1]
    arg = std::clamp(arg, -1.0, 1.0);
    phi = std::asin(arg);

    // Computation of the bound on theta1
    rc = 0;
    Eigen::MatrixXd n1(4, 3);
    Eigen::Vector2d sigma;
    if (std::abs(phi) > tol * 1e6) {
        n1.topRows<2>() = normal1;
        for (int i = 0; i < 2; ++i) {
            Eigen::RowVector3d temp = normal2.row(i);
            n1.row(2 + i) = temp.cross(e1);
        }

        // Compute sigma
        sigma = n1.bottomRows<2>() * mean_normal1.transpose();
        for (int i = 0; i < 2; ++i) {
            if (std::abs(sigma(i)) < tol) {
                sigma(i) = 1.0;
            }
        }

        Eigen::Vector2d norms;
        for (int i = 0; i < 2; ++i) {
            norms(i) = n1.row(2 + i).norm();
        }

        Eigen::Vector2d signs;
        for (int i = 0; i < 2; ++i) {
            signs(i) = (sigma(i) >= 0) ? 1.0 : -1.0;
        }
        sigma = signs.array() / norms.array();

        // Update n1
        Eigen::MatrixXd sigma_mat(2, 3);
        sigma_mat = sigma.replicate(1, 3);
        n1.bottomRows<2>() = n1.bottomRows<2>().array() * sigma_mat.array();

        // Compute theta1
        Eigen::MatrixXd cross_products(4, 3);
        for (int i = 0; i < 4; ++i) {
            Eigen::Vector3d temp = n1.row(i).transpose();
            cross_products.row(i) = x.cross(temp);
        }

        Eigen::Vector4d numerator = cross_products * e1.transpose();
        Eigen::Vector4d denominator = n1 * x.transpose();
        Eigen::VectorXd theta1(4);
        for (int i = 0; i < 4; ++i) {
            theta1(i) = std::atan2(numerator(i), denominator(i));
        }

        // Adjust theta1 if mean_normal1.dot(x) < 0
        if (mean_normal1.dot(x) < 0) {
            for (int i = 0; i < theta1.size(); ++i) {
                if (theta1(i) < 0) {
                    theta1(i) += 2 * M_PI;
                }
            }
        }

        // Compute theta1_min and theta1_max
        if (rc != 0) {
            theta1_min.resize(2);
            theta1_max.resize(2);
            theta1_min(0) = theta1(0);
            theta1_min(1) = rc * M_PI / 2;
            theta1_max(0) = rc * M_PI / 2;
            theta1_max(1) = theta1(1);
        } else {
            if (phi < 0) {
                // Swap theta1(2) and theta1(3)
                std::swap(theta1(2), theta1(3));
            }
            if (theta1(3) >= theta1(2) - tol) {
                if (theta1(1) - theta1(0) < -tol) {
                    throw std::runtime_error(
                            "Assertion failed: theta1(1) - theta1(0) < -tol");
                }
                theta1_min.resize(1);
                theta1_max.resize(1);
                theta1_min(0) = std::max(theta1(0), theta1(2));
                theta1_max(0) = std::min(theta1(1), theta1(3));
            } else {
                // Compute adjusted theta1_min and theta1_max
                Eigen::Matrix2d theta1_pairs_min;
                theta1_pairs_min << theta1(0), theta1(0), theta1(2), theta1(2);
                Eigen::Matrix2d adjust_min;
                adjust_min << 0, 0, M_PI, 0;
                theta1_pairs_min = theta1_pairs_min - adjust_min;

                Eigen::Matrix2d theta1_pairs_max;
                theta1_pairs_max << theta1(1), theta1(1), theta1(3), theta1(3);
                Eigen::Matrix2d adjust_max;
                adjust_max << 0, 0, 0, M_PI;
                theta1_pairs_max = theta1_pairs_max + adjust_max;

                theta1_min.resize(2);
                theta1_max.resize(2);
                theta1_min(0) = std::max(theta1_pairs_min(0, 0),
                                         theta1_pairs_min(1, 0));
                theta1_min(1) = std::max(theta1_pairs_min(0, 1),
                                         theta1_pairs_min(1, 1));
                theta1_max(0) = std::min(theta1_pairs_max(0, 0),
                                         theta1_pairs_max(1, 0));
                theta1_max(1) = std::min(theta1_pairs_max(0, 1),
                                         theta1_pairs_max(1, 1));
            }
        }
    } else {
        // Second branch
        Eigen::MatrixXd c(2, 3);
        for (int i = 0; i < 2; ++i) {
            Eigen::RowVector3d temp = normal2.row(i);
            c.row(i) = temp.cross(z);
        }

        double sigma;
        if (mean_normal1.dot(y) > 0) {
            sigma = 1;
        } else {
            if (mean_normal1.dot(x) < 0) {
                sigma = 3;
            } else {
                sigma = -1;
            }
        }

        double c_dot = c.row(0).dot(c.row(1));
        if (c_dot < -tol * 1e2) {
            // Compute theta1
            Eigen::MatrixXd cross_products(2, 3);
            for (int i = 0; i < 2; ++i) {
                Eigen::RowVector3d temp = normal1.row(i);
                cross_products.row(i) = x.cross(temp);
            }

            Eigen::Vector2d numerator = cross_products * e1.transpose();
            Eigen::Vector2d denominator = normal1 * x.transpose();
            Eigen::VectorXd theta1(2);
            for (int i = 0; i < 2; ++i) {
                theta1(i) = std::atan2(numerator(i), denominator(i));
            }

            // Compute c = cross(normal1, [y; y])
            Eigen::MatrixXd c_normal1(2, 3);
            for (int i = 0; i < 2; ++i) {
                Eigen::RowVector3d temp = normal1.row(i);
                c_normal1.row(i) = temp.cross(y);
            }

            double c_normal1_dot = c_normal1.row(0).dot(c_normal1.row(1));
            if (c_normal1_dot < -tol) {
                rc = static_cast<int>(sigma);
            } else {
                theta1.conservativeResize(4);
                theta1(2) = 0;
                theta1(3) = 2 * M_PI;
            }

            // Adjust theta1 if mean_normal1.dot(x) < 0
            if (mean_normal1.dot(x) < 0) {
                for (int i = 0; i < theta1.size(); ++i) {
                    if (theta1(i) < 0) {
                        theta1(i) += 2 * M_PI;
                    }
                }
            }

            // Compute theta1_min and theta1_max
            if (rc != 0) {
                theta1_min.resize(2);
                theta1_max.resize(2);
                theta1_min(0) = theta1(0);
                theta1_min(1) = rc * M_PI / 2;
                theta1_max(0) = rc * M_PI / 2;
                theta1_max(1) = theta1(1);
            } else {
                if (phi < 0) {
                    // Swap theta1(2) and theta1(3)
                    std::swap(theta1(2), theta1(3));
                }
                if (theta1(3) >= theta1(2) - tol) {
                    if (theta1(1) - theta1(0) < -tol) {
                        throw std::runtime_error(
                                "Assertion failed: theta1(2) - theta1(1) < "
                                "-tol");
                    }
                    theta1_min.resize(1);
                    theta1_max.resize(1);
                    theta1_min(0) = std::max(theta1(0), theta1(2));
                    theta1_max(0) = std::min(theta1(1), theta1(3));
                } else {
                    // Compute adjusted theta1_min and theta1_max
                    Eigen::Matrix2d theta1_pairs_min;
                    theta1_pairs_min << theta1(0), theta1(0), theta1(2),
                            theta1(2);
                    Eigen::Matrix2d adjust_min;
                    adjust_min << 0, 0, M_PI, 0;
                    theta1_pairs_min = theta1_pairs_min - adjust_min;

                    Eigen::Matrix2d theta1_pairs_max;
                    theta1_pairs_max << theta1(1), theta1(1), theta1(3),
                            theta1(3);
                    Eigen::Matrix2d adjust_max;
                    adjust_max << 0, 0, 0, M_PI;
                    theta1_pairs_max = theta1_pairs_max + adjust_max;

                    theta1_min.resize(2);
                    theta1_max.resize(2);
                    theta1_min(0) = std::max(theta1_pairs_min(0, 0),
                                             theta1_pairs_min(1, 0));
                    theta1_min(1) = std::max(theta1_pairs_min(0, 1),
                                             theta1_pairs_min(1, 1));
                    theta1_max(0) = std::min(theta1_pairs_max(0, 0),
                                             theta1_pairs_max(1, 0));
                    theta1_max(1) = std::min(theta1_pairs_max(0, 1),
                                             theta1_pairs_max(1, 1));
                }
            }
        } else {
            // Compute theta1
            Eigen::MatrixXd cross_products(2, 3);
            for (int i = 0; i < 2; ++i) {
                Eigen::RowVector3d temp = normal1.row(i);
                cross_products.row(i) = x.cross(temp);
            }

            Eigen::VectorXd numerator = cross_products * e1.transpose();
            Eigen::VectorXd denominator = normal1 * x.transpose();
            Eigen::VectorXd theta1(4);
            for (int i = 0; i < 2; ++i) {
                theta1(i) = std::atan2(numerator(i), denominator(i));
            }
            theta1(2) = sigma * M_PI / 2;
            theta1(3) = sigma * M_PI / 2;

            // Adjust theta1 if mean_normal1.dot(x) < 0
            if (mean_normal1.dot(x) < 0) {
                for (int i = 0; i < theta1.size(); ++i) {
                    if (theta1(i) < 0) {
                        theta1(i) += 2 * M_PI;
                    }
                }
            }

            // Compute theta1_min and theta1_max
            if (rc != 0) {
                theta1_min.resize(2);
                theta1_max.resize(2);
                theta1_min(0) = theta1(0);
                theta1_min(1) = rc * M_PI / 2;
                theta1_max(0) = rc * M_PI / 2;
                theta1_max(1) = theta1(1);
            } else {
                if (phi < 0) {
                    // Swap theta1(2) and theta1(3)
                    std::swap(theta1(2), theta1(3));
                }
                if (theta1(3) >= theta1(2) - tol) {
                    if (theta1(1) - theta1(0) < -tol) {
                        throw std::runtime_error(
                                "Assertion failed: theta1(2) - theta1(1) < "
                                "-tol");
                    }
                    theta1_min.resize(1);
                    theta1_max.resize(1);
                    theta1_min(0) = std::max(theta1(0), theta1(2));
                    theta1_max(0) = std::min(theta1(1), theta1(3));
                } else {
                    // Compute adjusted theta1_min and theta1_max
                    Eigen::MatrixXd theta1_pairs_min(2, 2);
                    theta1_pairs_min << theta1(0), theta1(0), theta1(2),
                            theta1(2);
                    Eigen::MatrixXd adjust_min(2, 2);
                    adjust_min << 0, 0, M_PI, 0;
                    theta1_pairs_min = theta1_pairs_min - adjust_min;

                    Eigen::MatrixXd theta1_pairs_max(2, 2);
                    theta1_pairs_max << theta1(1), theta1(1), theta1(3),
                            theta1(3);
                    Eigen::MatrixXd adjust_max(2, 2);
                    adjust_max << 0, 0, 0, M_PI;
                    theta1_pairs_max = theta1_pairs_max + adjust_max;

                    theta1_min.resize(2);
                    theta1_max.resize(2);
                    theta1_min(0) = std::max(theta1_pairs_min(0, 0),
                                             theta1_pairs_min(1, 0));
                    theta1_min(1) = std::max(theta1_pairs_min(0, 1),
                                             theta1_pairs_min(1, 1));
                    theta1_max(0) = std::min(theta1_pairs_max(0, 0),
                                             theta1_pairs_max(1, 0));
                    theta1_max(1) = std::min(theta1_pairs_max(0, 1),
                                             theta1_pairs_max(1, 1));
                }
            }
        }
    }

    // Final adjustments to theta1_min and theta1_max
    Eigen::VectorXd gud(theta1_min.size());
    for (int i = 0; i < theta1_min.size(); ++i) {
        gud(i) = (theta1_min(i) < theta1_max(i) - tol * 1e3);
    }

    std::vector<double> theta1_min_filtered;
    std::vector<double> theta1_max_filtered;

    for (int i = 0; i < gud.size(); ++i) {
        if (gud(i)) {
            theta1_min_filtered.push_back(theta1_min(i) + tol);
            theta1_max_filtered.push_back(theta1_max(i) - tol * 2e1);
        }
    }

    // Convert back to Eigen vectors
    if (!theta1_min_filtered.empty()) {
        theta1_min = Eigen::Map<Eigen::VectorXd>(theta1_min_filtered.data(),
                                                 theta1_min_filtered.size());
        theta1_max = Eigen::Map<Eigen::VectorXd>(theta1_max_filtered.data(),
                                                 theta1_max_filtered.size());
    } else {
        theta1_min.resize(0);
        theta1_max.resize(0);
    }
}
OrientedBoundingBox OrientedBoundingBox::CreateFromPointsMinimalRourke(
        const std::vector<Eigen::Vector3d>& points, bool robust) {

    // Check if input is valid
    if (points.empty()) {
        utility::LogError("The point set is empty.");
        return OrientedBoundingBox();
    }

    // Tolerance for numerical computations
    const double tol = 1e-9;

    // Normalize the points to improve numerical stability
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& pt : points) {
        mean += pt;
    }
    mean /= static_cast<double>(points.size());

    std::vector<Eigen::Vector3d> normalized_points;
    normalized_points.reserve(points.size());
    double normFactor = 0.0;
    for (const auto& pt : points) {
        Eigen::Vector3d centered = pt - mean;
        double norm = centered.norm();
        if (norm > normFactor) normFactor = norm;
        normalized_points.push_back(centered);
    }
    for (auto& pt : normalized_points) {
        pt /= normFactor;
    }

    // Compute the convex hull of the normalized points
    std::shared_ptr<TriangleMesh> convex_hull_mesh;
    std::tie(convex_hull_mesh, std::ignore) =
            Qhull::ComputeConvexHull(normalized_points, robust);

    if (!convex_hull_mesh) {
        utility::LogError("Failed to compute convex hull.");
        return OrientedBoundingBox();
    }

    // Get convex hull vertices and triangles
    const std::vector<Eigen::Vector3d>& hullV = convex_hull_mesh->vertices_;
    const std::vector<Eigen::Vector3i>& hullT = convex_hull_mesh->triangles_;
    int numVertices = static_cast<int>(hullV.size());
    int numTriangles = static_cast<int>(hullT.size());

    // Map to Eigen types
    Eigen::MatrixXd X(numVertices, 3);
    // X << -0.5774, -0.5774, -0.5774, 0.5774, 0.5774, -0.5774, 0.5774, -0.5774,
    //         0.5774, -0.5774, 0.5774, 0.5774;
    // X << 6.8956e-02, 7.9829e-01, -3.4341e-01, 7.3657e-02, 7.5508e-01,
    //         -3.5821e-01, 6.5277e-02, 7.1861e-01, -3.7886e-01, 7.7698e-02,
    //         8.2102e-01, -3.2310e-01, 7.1300e-02, 6.4208e-01, -3.9582e-01,
    //         -2.3525e-03, 7.5568e-01, -3.7787e-01, 7.3631e-02, 8.7308e-01,
    //         -2.9454e-01, 8.3708e-02, 8.7032e-01, -2.8927e-01, 6.3634e-02,
    //         6.1436e-01, -4.0364e-01, -2.3525e-03, 8.7290e-01, -3.1595e-01,
    //         6.1658e-02, 5.1298e-01, -4.0391e-01, 9.2000e-02, 7.0000e-01,
    //         -3.0281e-01, -2.3525e-03, 6.1491e-01, -4.0393e-01, -7.3662e-02,
    //         7.9829e-01, -3.4341e-01, 7.9589e-02, 9.3866e-01, -2.2563e-01,
    //         -2.3525e-03, 5.2181e-01, -4.1530e-01, 9.6215e-02, 7.9981e-01,
    //         -2.6012e-01, 8.7784e-02, 6.0019e-01, -3.4551e-01, -6.8340e-02,
    //         6.1436e-01, -4.0364e-01, -6.9983e-02, 7.1861e-01, -3.7886e-01,
    //         -7.8337e-02, 8.7308e-01, -2.9454e-01, 8.5512e-02, 9.2540e-01,
    //         -2.3040e-01, -2.3525e-03, 9.5224e-01, -2.3568e-01, -2.3525e-03,
    //         4.0164e-01, -3.9758e-01, 5.9026e-02, 3.4886e-01, -3.6807e-01,
    //         -7.8362e-02, 7.5508e-01, -3.5821e-01, -8.4294e-02, 9.3866e-01,
    //         -2.2563e-01, -8.2403e-02, 8.2102e-01, -3.2310e-01, -2.3525e-03,
    //         9.8520e-01, -1.7140e-01, -6.6364e-02, 5.1298e-01, -4.0391e-01,
    //         -7.6006e-02, 6.4208e-01, -3.9582e-01, -8.8414e-02, 8.7032e-01,
    //         -2.8927e-01, -2.3525e-03, 2.7876e-01, -3.5365e-01, -6.3733e-02,
    //         3.4886e-01, -3.6807e-01, -9.0217e-02, 9.2540e-01, -2.3040e-01,
    //         -8.7365e-02, 9.6956e-01, -1.6329e-01, -9.6706e-02, 7.0000e-01,
    //         -3.0281e-01, -1.0587e-01, 9.2333e-01, -1.4329e-01, -8.8952e-02,
    //         9.6313e-01, -1.5868e-01, 8.5590e-02, 9.7548e-01, -1.5620e-01,
    //         1.0578e-01, 9.1045e-01, -1.2461e-01, 7.9080e-02, 8.1491e-01,
    //         -8.4354e-02, -9.2490e-02, 6.0019e-01, -3.4551e-01, -1.0092e-01,
    //         7.9981e-01, -2.6012e-01, 9.5054e-02, 9.6784e-01, -1.5122e-01,
    //         -1.1067e-01, 8.8615e-01, -1.2631e-01, -9.4379e-02, 8.1322e-01,
    //         -9.2525e-02, -8.7752e-02, 8.0241e-01, -8.8124e-02, -9.8310e-02,
    //         -6.4334e-01, 3.9246e-01, -2.3525e-03, -6.4334e-01, 4.1817e-01,
    //         -2.3525e-03, -6.6075e-01, 4.2216e-01, 1.9355e-01, -6.6075e-01,
    //         2.2626e-01, 1.6731e-01, -6.6075e-01, 1.2831e-01, -1.0031e-01,
    //         -6.6075e-01, 3.9592e-01, 9.3220e-02, -6.8275e-01, 6.0721e-02,
    //         -2.3525e-03, -6.8275e-01, 3.5113e-02, -1.7201e-01, -6.6075e-01,
    //         1.2831e-01, -2.3525e-03, -6.8275e-01, 4.1740e-01, 1.1816e-01,
    //         -7.1066e-01, 4.3499e-01, 1.8879e-01, -6.8275e-01, 2.2626e-01,
    //         1.6318e-01, -6.8275e-01, 1.3069e-01, -9.7926e-02, -6.8275e-01,
    //         6.0721e-02, -9.7926e-02, -6.8275e-01, 3.9180e-01, -1.9826e-01,
    //         -6.6075e-01, 2.2626e-01, 2.0638e-01, -7.1066e-01, 3.4677e-01,
    //         -1.6789e-01, -6.8275e-01, 1.3069e-01, -1.9350e-01, -6.8275e-01,
    //         2.2626e-01, -2.1108e-01, -7.1066e-01, 3.4677e-01;

    Eigen::MatrixXi hullFaces(numTriangles, 3);

    // hullFaces << 65, 56, 68, 64, 46, 68, 46, 64, 44, 41, 52, 65, 52, 41, 17,
    // 63,
    //         51, 58, 59, 41, 65, 59, 51, 50, 51, 59, 58, 59, 65, 68, 63, 59,
    //         68, 59, 63, 58, 40, 36, 29, 42, 59, 50, 59, 42, 41, 34, 62, 56,
    //         56, 62, 68, 67, 64, 68, 55, 25, 56, 55, 56, 65, 33, 34, 56, 25,
    //         33, 56, 52, 60, 65, 51, 49, 50, 46, 49, 68, 38, 46, 44, 38, 39,
    //         46, 39, 38, 36, 40, 39, 36, 39, 45, 41, 45, 39, 40, 23, 40, 29,
    //         32, 38, 44, 64, 37, 44, 37, 32, 44, 34, 30, 57, 43, 64, 57, 30,
    //         43, 57, 43, 30, 31, 43, 37, 64, 37, 43, 32, 41, 8, 17, 45, 8, 41,
    //         12, 52, 17, 8, 12, 17, 11, 25, 53, 52, 18, 53, 18, 11, 53, 11,
    //         18, 5, 12, 18, 52, 18, 12, 8, 66, 34, 57, 66, 62, 34, 64, 66, 57,
    //         67, 66, 64, 62, 66, 68, 66, 67, 68, 25, 61, 53, 55, 61, 25, 61,
    //         52, 53, 61, 60, 52, 61, 55, 65, 60, 61, 65, 24, 33, 25, 24, 11,
    //         16, 11, 24, 25, 33, 24, 34, 30, 24, 16, 24, 30, 34, 54, 51, 63,
    //         54, 49, 51, 54, 63, 68, 49, 54, 68, 42, 47, 41, 47, 39, 41, 39,
    //         47, 46, 49, 47, 50, 47, 49, 46, 19, 13, 6, 13, 19, 16, 19, 30,
    //         16, 30, 19, 31, 13, 9, 6, 9, 13, 16, 11, 9, 16, 9, 11, 5, 36, 27,
    //         29, 27, 23, 29, 23, 15, 40, 15, 45, 40, 48, 42, 50, 47, 48, 50,
    //         48, 47, 42, 14, 20, 6, 20, 19, 6, 19, 20, 31, 26, 43, 31, 43, 26,
    //         32, 20, 26, 31, 26, 20, 14, 14, 21, 32, 27, 21, 23, 21, 27, 32,
    //         32, 35, 38, 27, 35, 32, 38, 35, 36, 35, 27, 36, 3, 1, 6, 9, 3, 6,
    //         3, 9, 5, 18, 2, 5, 2, 18, 8, 2, 3, 5, 3, 2, 1, 1, 10, 6, 21, 10,
    //         23, 10, 14, 6, 10, 21, 14, 22, 8, 45, 15, 22, 45, 22, 15, 8, 28,
    //         14, 32, 26, 28, 32, 28, 26, 14, 1, 4, 8, 4, 2, 8, 2, 4, 1, 7, 1,
    //         8, 7, 10, 1, 15, 7, 8, 7, 15, 23, 10, 7, 23;

    // hullFaces = hullFaces - Eigen::MatrixXi::Ones(numTriangles, 3);
    Eigen::MatrixXd hullNormals(numTriangles, 3);
    for (int i = 0; i < numVertices; ++i) {
        X.row(i) = hullV[i];
    }
    // hullFaces << 2, 1, 0, 1, 3, 0, 3, 2, 0, 2, 3, 1;
    for (int i = 0; i < numTriangles; ++i) {
        hullFaces.row(i) = hullT[i];
        hullNormals.row(i) = -normalToFace(hullFaces.row(i), X).transpose();
    }

    Eigen::MatrixXd hullEdges(6, 7);

    // hullEdges << 0, 1, 0, 1, 0.707107, 0.707107, 0, 0, 2, 2, 0, 0.707107, 0,
    //         0.707107, 0, 3, 1, 2, 0, 0.707107, 0.707107, 1, 2, 0, 3, 0,
    //         -0.707107, 0.707107, 1, 3, 3, 1, -0.707107, 0, 0.707107, 2, 3, 2,
    //         3, -0.707107, 0.707107, 0;
    listEdges(hullFaces, X, hullEdges);

    std::vector<std::vector<int>> hullNodes;
    nodeToFaces(hullFaces, X, hullNodes);
    // std::vector<int> F = {1, 0, 2};
    // hullNodes.push_back(F);
    // F = {0, 1, 3};
    // hullNodes.push_back(F);
    // F = {2, 0, 3};
    // hullNodes.push_back(F);
    // F = {1, 2, 3};
    // hullNodes.push_back(F);

    int i0 = 1;
    int j0 = 0;
    int nEdges = hullEdges.rows();
    double Vopt = std::numeric_limits<double>::max();

    // Variables to store minimal volume and corresponding OBB
    Eigen::Matrix3d min_R;
    Eigen::Vector3d min_extent;
    Eigen::Vector3d min_center;
    // double phi = 0;

    // Loop over edges to find the minimal bounding box
    for (int i = i0; i < std::max(nEdges, i0); ++i) {
        Eigen::Matrix<double, 2, 3> normal1 =
                hullNormals(hullEdges.row(i).segment<2>(2).array(),
                            Eigen::placeholders::all);
        Eigen::RowVector3d real_mean_normal1 = normal1.colwise().mean();

        Eigen::RowVector3d real_e1 =
                hullEdges.row(i).segment<3>(4).normalized();

        // Loop over edges j
        for (int j = j0; j < std::min(i, nEdges); ++j) {
            Eigen::Matrix<double, 2, 3> normal2 =
                    hullNormals(hullEdges.row(j).segment<2>(2).array(),
                                Eigen::placeholders::all);
            Eigen::RowVector3d real_mean_normal2 = normal2.colwise().mean();
            Eigen::RowVector3d real_e2 =
                    hullEdges.row(j).segment<3>(4).normalized();

            double phi;
            Eigen::Matrix3d R_x1;
            Eigen::VectorXd theta1_min;
            Eigen::VectorXd theta1_max;
            int rc1;
            computeTheta(real_e1, real_e2, tol, normal1, normal2,
                         real_mean_normal1, phi, R_x1, theta1_min, theta1_max,
                         rc1);

            if (std::abs(phi) < tol * 1e6) {
                continue;
            }

            for (int k = 0; k < theta1_min.size(); ++k) {
                Eigen::Matrix3d R_x = R_x1;
                Eigen::RowVector3d mean_normal1 = real_mean_normal1;
                Eigen::RowVector3d mean_normal2 = real_mean_normal2;
                Eigen::RowVector3d e1 = real_e1;
                Eigen::RowVector3d e2 = real_e2;
                Eigen::RowVector3d x = R_x.row(0);
                Eigen::RowVector3d z = R_x.row(2);

                // Declaration of some useful arrays
                Eigen::RowVector4i p1 = -Eigen::RowVector4i::Ones();
                Eigen::RowVector4i p2 = -Eigen::RowVector4i::Ones();
                Eigen::RowVector4d theta1_p2 =
                        Eigen::RowVector4d::Constant(theta1_max(k) + 2 * tol);

                std::array<std::vector<int>, 4> nodes_p2;
                std::array<std::vector<int>, 4> nodeExt;

                Eigen::RowVector4i nextNodeExt = -Eigen::RowVector4i::Ones();

                // Compute the smallest bounding box oriented along the axis
                // determined by the first theta1
                double theta1 = theta1_min(k);
                Eigen::Matrix3d R_n;
                computeR_n(theta1, phi, R_x, mean_normal2, R_n);

                // Rotate points to this orientation
                Eigen::MatrixXd X_n = X * R_n.transpose();  // (N*3)
                Eigen::RowVector3d minX_n;
                Eigen::RowVector3d maxX_n;
                Eigen::RowVector3i nodeMin;
                Eigen::RowVector3i nodeMax;

                // Compute min and max for each column:
                for (int col = 0; col < X_n.cols(); ++col) {
                    double min_val = std::numeric_limits<double>::infinity();
                    double max_val = -std::numeric_limits<double>::infinity();
                    int min_idx = -1;
                    int max_idx = -1;
                    for (int row = 0; row < X_n.rows(); ++row) {
                        double val = X_n(row, col);
                        if (val < min_val) {
                            min_val = val;
                            min_idx = row;
                        }
                        if (val > max_val) {
                            max_val = val;
                            max_idx = row;
                        }
                    }
                    minX_n(col) = min_val;
                    nodeMin(col) = min_idx;
                    maxX_n(col) = max_val;
                    nodeMax(col) = max_idx;
                }

                double V = (maxX_n - minX_n).prod();
                if (V < 0)
                    std::cout << "--------------> = " << maxX_n - minX_n
                              << std::endl;

                if (V < Vopt) {
                    // Update the optimal volume and rotation
                    min_R = R_n;
                    Vopt = V;
                    min_extent = maxX_n - minX_n;
                    min_center = 0.5 * (minX_n + maxX_n);
                }

                // prevNodeExt = [nodeMin nodeMax(3)]
                Eigen::RowVector4i prevNodeExt;
                prevNodeExt.head<3>() = nodeMin;
                prevNodeExt(3) = nodeMax(2);

                // Compute nodeExt
                for (int l = 0; l < 3; ++l) {
                    nodeExt[l].clear();
                    double threshold =
                            minX_n(l) +
                            tol * 1e2 * std::max(std::abs(minX_n(l)), 1.0);
                    for (int idx = 0; idx < X_n.rows(); ++idx) {
                        if (X_n(idx, l) <= threshold) {
                            nodeExt[l].push_back(idx);
                        }
                    }
                }

                nodeExt[3].clear();
                double threshold =
                        maxX_n(2) -
                        tol * 1e2 * std::max(std::abs(maxX_n(2)), 1.0);
                for (int idx = 0; idx < X_n.rows(); ++idx) {
                    if (X_n(idx, 2) >= threshold) {
                        nodeExt[3].push_back(idx);
                    }
                }

                // Loop on all theta1 crossing an edge on the Gaussian sphere
                while (theta1 < theta1_max(k) + tol) {
                    // std::endl; Computation of the p2's for -n1, -n2, -n3 and
                    // +n3
                    for (int n = 0; n < 4; ++n) {
                        if (p2(n) == -1) {
                            for (size_t m = 0; m < nodeExt[n].size(); ++m) {
                                int nodeIdx = nodeExt[n][m];
                                std::vector<int> faces = hullNodes[nodeIdx];
                                int nFaces = faces.size();
                                if (nFaces > 0) {
                                    if ((n == 2) != (phi > 0)) {
                                        std::reverse(faces.begin(),
                                                     faces.end());
                                    }

                                    int nIter = (n <= 1) ? nFaces : nFaces + 1;

                                    // Find l0
                                    int l0 = 0;
                                    if (p1(n) > -1) {
                                        auto it = std::find(faces.begin(),
                                                            faces.end(), p1(n));
                                        if (it != faces.end()) {
                                            l0 = std::distance(faces.begin(),
                                                               it);
                                            if (n <= 1) {
                                                l0 += 1;
                                            }
                                        } else {
                                            l0 = 0;
                                        }
                                    }

                                    // Inner loop over ll
                                    Eigen::RowVector3d intersection, newN1;
                                    Eigen::Matrix<double, 2, 3> normals;
                                    for (int ll = 0; ll < nIter; ++ll) {
                                        int l = (l0 + ll) % nFaces;
                                        Eigen::RowVector3i face =
                                                hullFaces.row(faces[l]);

                                        // Get normals
                                        normals.row(0) =
                                                hullNormals.row(faces[l]);
                                        if ((n == 2) != (phi > 0)) {
                                            normals.row(1) = hullNormals.row(
                                                    faces[((l - 1 + nFaces) %
                                                           nFaces)]);  // avant
                                        } else {
                                            normals.row(1) = hullNormals.row(
                                                    faces[((l + 1) %
                                                           nFaces)]);  // apres
                                        }

                                        // Find idxNode
                                        int idxNode = -1;
                                        for (int idx = 0; idx < 3; ++idx) {
                                            if (face(idx) == nodeIdx) {
                                                idxNode = idx;
                                                break;
                                            }
                                        }

                                        // Define edge
                                        int edge0 = face(idxNode);
                                        int edge1 = face((idxNode + 1) % 3);
                                        Eigen::Vector2i edge;
                                        edge << edge0, edge1;

                                        // Compute vector a
                                        Eigen::RowVector3d a =
                                                X.row(edge(1)) - X.row(edge(0));

                                        if (a.norm() < tol) {
                                            continue;
                                        }

                                        // Depending on n, compute c and newN1
                                        Eigen::RowVector3d c_vec;
                                        double sigma;

                                        if (n == 0) {
                                            c_vec = a.cross(e1);
                                            if (c_vec.norm() / a.norm() > tol) {
                                                sigma = -std::copysign(
                                                        1.0,
                                                        c_vec.dot(R_n.row(0)));
                                                if (std::abs(sigma) < tol)
                                                    sigma = 1.0;
                                                intersection =
                                                        sigma *
                                                        c_vec.normalized();
                                            } else {
                                                continue;
                                            }
                                            newN1 = -intersection;
                                        } else {
                                            if (n == 1) {
                                                c_vec = a.cross(e2);
                                                if (c_vec.norm() / a.norm() >
                                                    tol) {
                                                    sigma = -std::copysign(
                                                            1.0,
                                                            c_vec.dot(R_n.row(
                                                                    1)));
                                                    if (std::abs(sigma) < tol)
                                                        sigma = 1.0;
                                                    intersection =
                                                            sigma *
                                                            c_vec.normalized();
                                                } else {
                                                    continue;
                                                }

                                                // Compute c_vec again
                                                c_vec = intersection.cross(e1);
                                                if (c_vec.norm() /
                                                            intersection
                                                                    .norm() >
                                                    tol) {
                                                    sigma = std::copysign(
                                                            1.0,
                                                            c_vec.dot(R_n.row(
                                                                    0)));
                                                    if (std::abs(sigma) < tol)
                                                        sigma = 1.0;
                                                    newN1 = sigma *
                                                            c_vec.normalized();
                                                } else {
                                                    continue;
                                                }

                                            } else {
                                                Eigen::RowVector3d edge_x =
                                                        a * R_x.transpose();
                                                double a_val = edge_x(0) *
                                                               std::cos(phi) /
                                                               2.0;
                                                double b_val = -edge_x(1) *
                                                               std::cos(phi) /
                                                               2.0;
                                                double c_val = edge_x(2) *
                                                               std::sin(phi);

                                                double psi = std::atan2(b_val,
                                                                        a_val);
                                                double r = std::sqrt(
                                                        a_val * a_val +
                                                        b_val * b_val);

                                                if (std::abs(r) <
                                                    std::max(std::abs(c_val -
                                                                      b_val),
                                                             tol)) {
                                                    continue;
                                                }

                                                double temp =
                                                        (c_val - b_val) / r;
                                                temp = std::clamp(
                                                        temp, -1.0,
                                                        1.0);  // Ensure value
                                                               // is within [-1,
                                                               // 1]
                                                double asin_val =
                                                        std::asin(temp);

                                                Eigen::Vector4d theta1inter;
                                                theta1inter(0) =
                                                        0.5 * (asin_val - psi);
                                                theta1inter(1) =
                                                        0.5 *
                                                        (M_PI - asin_val - psi);

                                                theta1inter(0) = matlab_mod(
                                                        theta1inter(0),
                                                        2 * M_PI);

                                                theta1inter(1) = matlab_mod(
                                                        theta1inter(1),
                                                        2 * M_PI);

                                                theta1inter(2) = matlab_mod(
                                                        theta1inter(0) + M_PI,
                                                        2 * M_PI);

                                                theta1inter(3) = matlab_mod(
                                                        theta1inter(1) + M_PI,
                                                        2 * M_PI);

                                                // Adjust theta1inter if
                                                // mean_normal1.dot(x) >= 0
                                                if (mean_normal1.dot(x) >= 0) {
                                                    theta1inter.array() -= M_PI;
                                                }

                                                std::vector<int> idxTh1;
                                                for (int i = 0; i < 4; ++i) {
                                                    if (theta1inter(i) >
                                                        theta1) {
                                                        idxTh1.push_back(i);
                                                    }
                                                }

                                                // Set theta1inter accordingly
                                                double theta1inter_val;
                                                if (idxTh1.empty()) {
                                                    theta1inter_val =
                                                            theta1inter
                                                                    .minCoeff();
                                                } else {
                                                    theta1inter_val =
                                                            std::numeric_limits<
                                                                    double>::
                                                                    max();
                                                    for (int idx : idxTh1) {
                                                        if (theta1inter(idx) <
                                                            theta1inter_val) {
                                                            theta1inter_val =
                                                                    theta1inter(
                                                                            idx);
                                                        }
                                                    }
                                                }

                                                // Compute newR_n
                                                Eigen::Matrix3d newR_n;
                                                computeR_n(theta1inter_val, phi,
                                                           R_x, mean_normal2,
                                                           newR_n);

                                                newN1 = newR_n.row(0);
                                                intersection =
                                                        (2.0 * n - 5.0) *
                                                        newR_n.row(
                                                                2);  // BUG ???
                                            }
                                        }
                                        Eigen::MatrixXd a1(2, 3);
                                        Eigen::MatrixXd b1(2, 3);
                                        Eigen::MatrixXd c1(2, 3);
                                        if ((n <= 1) or (ll > 0) or
                                            (std::abs((X.row(edge(1)) -
                                                       X.row(edge(0))) *
                                                      R_n.row(2).transpose()) >
                                             tol)) {
                                            // a = [normals(1,:); intersection];
                                            a1.row(0) = normals.row(0);
                                            a1.row(1) = intersection;
                                        } else {
                                            // a = [(2*n-7)*R_n(3,:);
                                            // intersection];
                                            a1.row(0) = (2.0 * n - 5.0) *
                                                        R_n.row(2);
                                            a1.row(1) = intersection;
                                        }

                                        b1.row(0) = intersection;
                                        b1.row(1) = normals.row(1);

                                        Eigen::RowVector3d temp1, temp2;

                                        temp1 = a1.row(0);
                                        temp2 = b1.row(0);
                                        c1.row(0) = temp1.cross(temp2);
                                        temp1 = a1.row(1);
                                        temp2 = b1.row(1);
                                        c1.row(1) = temp1.cross(temp2);

                                        // Compute norm(diff(a)) < tol
                                        Eigen::RowVector3d diff_a =
                                                a1.row(1) - a1.row(0);
                                        double norm_diff_a = diff_a.norm();

                                        // Determine testBinary
                                        bool testBinary;
                                        if (norm_diff_a < tol) {
                                            testBinary = true;
                                        } else {
                                            // Compute c(1,:)*c(2,:)' >= -tol
                                            // In C++, c.row(0) is the first
                                            // row, c.row(1) is the second row
                                            double dot_product =
                                                    c1.row(0).dot(c1.row(1));
                                            testBinary = (dot_product >= -tol);
                                        }

                                        if (testBinary) {
                                            Eigen::RowVector3d aa = R_n.row(0);
                                            Eigen::RowVector3d bb = newN1;
                                            Eigen::RowVector3d cc =
                                                    aa.cross(bb);

                                            double sinDtheta1 = cc.dot(z);
                                            if (std::abs(sinDtheta1) < tol) {
                                                sinDtheta1 = 0.0;
                                            }
                                            double cosDtheta1 = aa.dot(bb);
                                            double dtheta1 = std::atan2(
                                                    sinDtheta1, cosDtheta1);

                                            if (dtheta1 > tol) {
                                                double nextTheta1 =
                                                        std::min(
                                                                theta1 +
                                                                        dtheta1,
                                                                theta1_p2(n)) -
                                                        tol;

                                                Eigen::Matrix3d midR_n;
                                                computeR_n(0.5 * (theta1 +
                                                                  nextTheta1),
                                                           phi, R_x,
                                                           mean_normal2,
                                                           midR_n);

                                                Eigen::Matrix3d nextR_n;
                                                computeR_n(nextTheta1, phi, R_x,
                                                           mean_normal2,
                                                           nextR_n);

                                                // Compute delta values
                                                double delta, delta2 = 1.0,
                                                              delta3;

                                                if (n <= 2) {
                                                    delta = -(
                                                            X.row(edge(0)).dot(
                                                                    midR_n.row(
                                                                            n)) -
                                                            X.row(prevNodeExt(
                                                                          n))
                                                                    .dot(midR_n.row(
                                                                            n)));
                                                    if (p2(n) > -1) {
                                                        delta2 = -(
                                                                X.row(edge(1)).dot(
                                                                        nextR_n.row(
                                                                                n)) -
                                                                X.row(nextNodeExt(
                                                                              n))
                                                                        .dot(nextR_n.row(
                                                                                n)));
                                                    }
                                                    delta3 = -(
                                                            X.row(edge(1)).dot(
                                                                    midR_n.row(
                                                                            n)) -
                                                            X.row(edge(0)).dot(
                                                                    midR_n.row(
                                                                            n)));
                                                } else {
                                                    delta = -(
                                                            -X.row(prevNodeExt(
                                                                           n))
                                                                     .dot(midR_n.row(
                                                                             2)) +
                                                            X.row(edge(0)).dot(
                                                                    midR_n.row(
                                                                            2)));
                                                    if (p2(n) > -1) {
                                                        delta2 = -(
                                                                -X.row(nextNodeExt(
                                                                               n))
                                                                         .dot(nextR_n.row(
                                                                                 2)) +
                                                                X.row(edge(1)).dot(
                                                                        nextR_n.row(
                                                                                2)));
                                                    }
                                                    delta3 =
                                                            (-X.row(edge(0)).dot(
                                                                     midR_n.row(
                                                                             2)) +
                                                             X.row(edge(1)).dot(
                                                                     midR_n.row(
                                                                             2)));
                                                }

                                                if (p2(n) == -1 || p1(n) > -1 ||
                                                    (p1(n) == -1 &&
                                                     p2(n) > -1 &&
                                                     (theta1 + dtheta1 <
                                                              theta1_p2(n) -
                                                                      tol ||
                                                      (delta >= 0 &&
                                                       delta2 > -tol &&
                                                       delta3 < tol)))) {
                                                    if (theta1 + dtheta1 <
                                                                theta1_max(k) &&
                                                        (delta >= 0 &&
                                                         delta2 > -tol &&
                                                         delta3 < tol)) {
                                                        prevNodeExt(n) =
                                                                edge(0);
                                                        int prevNodeExt_n =
                                                                prevNodeExt(n);
                                                        if ((n == 2) !=
                                                            (phi > 0)) {
                                                            p2(n) = faces
                                                                    [(l - 1 +
                                                                      nFaces) %
                                                                     nFaces];

                                                        } else {
                                                            p2(n) = faces
                                                                    [(l + 1) %
                                                                     nFaces];
                                                        }

                                                        theta1_p2(n) = theta1 +
                                                                       dtheta1;

                                                        // Adjust intersection
                                                        Eigen::RowVector3d tt =
                                                                normals.colwise()
                                                                        .mean();

                                                        intersection *= std::copysign(
                                                                1.0,
                                                                intersection.dot(
                                                                        tt));

                                                        nodes_p2[n] = findEquivalentExtrema(
                                                                intersection,
                                                                edge(1),
                                                                hullFaces,
                                                                hullNodes, X,
                                                                tol * 1e2,
                                                                prevNodeExt_n);

                                                        nextNodeExt(n) =
                                                                edge(1);

                                                        if (nextNodeExt(n) ==
                                                            -1) {
                                                            throw std::
                                                                    runtime_error(
                                                                            "ne"
                                                                            "xt"
                                                                            "No"
                                                                            "de"
                                                                            "Ex"
                                                                            "t("
                                                                            "n)"
                                                                            " i"
                                                                            "s "
                                                                            "ze"
                                                                            "r"
                                                                            "o");
                                                        }
                                                    }
                                                }
                                            }
                                        }  // End of testBinary
                                    }  // End of inner loop over ll
                                }
                            }
                        }

                        // Handle case when p2(n) == 0 after loop
                        if (p2(n) == -1) {
                            Eigen::MatrixXd X_n(X.rows(), 3);
                            X_n = X * R_n.transpose();

                            Eigen::MatrixXd x_n(X.rows(), 3);
                            Eigen::Matrix3d R_xn;
                            computeR_n(theta1_max(k), phi, R_x, mean_normal2,
                                       R_xn);
                            x_n = X * R_xn.transpose();

                            if (n <= 2) {
                                double minX_n = X_n.col(n).minCoeff();
                                int nodemin;
                                x_n.col(n).minCoeff(&nodemin);
                                double threshold =
                                        minX_n +
                                        tol * std::max(1e3 * std::abs(minX_n),
                                                       1.0);
                                if (X_n(nodemin, n) < threshold) {
                                    prevNodeExt(n) = nodemin;
                                }
                            } else {
                                double maxX_n = X_n.col(2).maxCoeff();
                                int nodemax;
                                x_n.col(2).maxCoeff(&nodemax);
                                double threshold =
                                        maxX_n -
                                        tol * std::max(1e3 * std::abs(maxX_n),
                                                       1.0);
                                if (X_n(nodemax, 2) > threshold) {
                                    prevNodeExt(3) = nodemax;
                                }
                            }
                        }

                    }  // End of loop over n

                    // Update p2 and compute theta1_new
                    for (int idx = 0; idx < 4; ++idx) {
                        if (p2(idx) == -1) {
                            p2(idx) = -2;
                        }
                    }
                    double theta1_new =
                            std::min(theta1_p2.minCoeff(), theta1_max(k));

                    // Find indices to update
                    std::vector<int> update_indices;
                    for (int idx = 0; idx < 4; ++idx) {
                        if (theta1_p2(idx) <= theta1_new + tol) {
                            update_indices.push_back(idx);
                        }
                    }

                    // Update p1, p2, theta1_p2
                    for (int idx : update_indices) {
                        p1(idx) = p2(idx);
                        p2(idx) = -1;
                        theta1_p2(idx) = theta1_max(k) + 2 * tol;
                    }

                    // Find the minimum volume within the range (theta1,
                    // theta1_new] Compute X_Min and X_Max
                    Eigen::Matrix3d X_Min;
                    Eigen::Matrix3d X_Max;
                    for (int idx = 0; idx < 3; ++idx) {
                        X_Min.row(idx) = X.row(prevNodeExt(idx));
                    }
                    X_Max.row(0) = X.row(static_cast<int>(hullEdges(i, 0)));
                    X_Max.row(1) = X.row(static_cast<int>(hullEdges(j, 0)));
                    X_Max.row(2) = X.row(prevNodeExt(3));

                    Eigen::Matrix3d c = X_Max - X_Min;
                    Eigen::RowVector3d c1 = c.row(0) - (c.row(0).dot(e1)) * e1;
                    double gamma1 = -std::acos(-c1.dot(x) / c1.norm());
                    Eigen::RowVector3d c2 = c.row(1) - (c.row(1).dot(e2)) * e2;
                    Eigen::RowVector3d cross_x_c2 = x.cross(c2);
                    double gamma2 =
                            M_PI - std::asin(cross_x_c2.dot(e2) / c2.norm());
                    Eigen::RowVector3d c3 = c.row(2) * R_x.transpose();

                    // Compute sign of sin(phi)
                    double sin_gamma1 = std::sin(gamma1);
                    double cos_gamma1 = std::cos(gamma1);
                    double sin_gamma2 = std::sin(gamma2);
                    double cos_gamma2 = std::cos(gamma2);
                    double sin_phi = std::sin(phi);
                    double cos_phi = std::cos(phi);
                    double sin_phi_sign =
                            (sin_phi > 0) -
                            (sin_phi <
                             0);  // Equivalent to MATLAB's sign(sin(phi))

                    // Compute num
                    Eigen::Vector3d num_0;
                    Eigen::Vector4d num_1, num_2, deriv;
                    Eigen::VectorXd num_3(5), num_4(5), num_5(8), num_6(8),
                            num_7(8), num_8(8), num_9(8), num(8);

                    num_0(0) = sin_phi_sign * (c3(2) * sin_phi);
                    num_0(1) = sin_phi_sign * (-c3(0) * cos_phi);
                    num_0(2) =
                            sin_phi_sign * (c3(1) * cos_phi + c3(2) * sin_phi);

                    // Extend num and compute new num
                    num_1.setZero();
                    num_1.head(3) = num_0 * sin_gamma1;

                    num_2.setZero();
                    num_2.tail(3) = num_0 * cos_gamma1;
                    num_2 = num_1 + num_2;

                    num_3.setZero();
                    num_3.head(4) = num_2 * cos_gamma2 * sin_phi;

                    num_4.setZero();
                    num_4.tail(4) = num_2;
                    num_4 = num_3 + num_4 * sin_gamma2;

                    deriv(0) = 4 * num_4(0);
                    deriv(1) = 3 * num_4(1);
                    deriv(2) = 2 * num_4(2);
                    deriv(3) = num_4(3);

                    // Prepare polynomials for further computations
                    // num = [deriv 0 0 0 0]*sin(phi)^2 + [0 0 deriv 0
                    // 0]*(sin(phi)^2 + 1) + [0 0 0 0 deriv] -
                    // (4*sin(phi)^2*[num 0 0 0] + 2*(sin(phi)^2 + 1)*[0 0 num
                    // 0]);

                    num_5.setZero();
                    num_5.head(4) = deriv * std::pow(sin_phi, 2);

                    num_6.setZero();
                    num_6.segment<4>(2) = deriv * (std::pow(sin_phi, 2) + 1.0);

                    num_7.setZero();
                    num_7.tail(4) = deriv;

                    num_8.setZero();
                    num_8.head(5) = 4 * num_4 * std::pow(sin_phi, 2);

                    num_9.setZero();
                    num_9.segment<5>(2) =
                            2 * num_4 * (std::pow(sin_phi, 2) + 1.0);

                    num = num_5 + num_6 + num_7 - num_8 - num_9;

                    // Find real roots of the polynomial
                    std::vector<double> rootsNum;

                    // Since Eigen doesn't have a built-in roots solver, you can
                    // use a polynomial root-finding library Alternatively, for
                    // polynomials up to degree 4, you can implement analytical
                    // solutions For this example, we'll assume you have a
                    // function to compute the roots of a polynomial
                    rootsNum = findRealRoots(num);

                    // Compute theta1Local
                    std::vector<double> theta1Local;
                    for (double root : rootsNum) {
                        // Check if root is real (already ensured)
                        double theta = std::atan(root);
                        theta = matlab_mod(theta, M_PI) - M_PI;
                        if (theta >= theta1) {
                            theta1Local.push_back(theta);
                        }
                    }

                    std::vector<double> theta1_;
                    for (double theta1local : theta1Local) {
                        // Check if root is real (already ensured)
                        if (theta1local < theta1_new) {
                            theta1_.push_back(theta1local);
                        }
                    }

                    // Add theta1_new to theta1_
                    theta1_.push_back(theta1_new);

                    // Iterate over theta1_
                    for (size_t m = 0; m < theta1_.size(); ++m) {
                        double theta1_m = theta1_[m];
                        computeR_n(theta1_m, phi, R_x, mean_normal2, R_n);

                        if (m < (theta1_.size() - 1)) {
                            // Compute V using X_Max and X_Min
                            Eigen::VectorXd diagDiff =
                                    (X_Max * R_n.transpose()).diagonal() -
                                    (X_Min * R_n.transpose()).diagonal();

                            double V = diagDiff.prod();
                            if (V < 0)
                                std::cout << "--------------> diagDiff = "
                                          << diagDiff << std::endl;

                            if (V < Vopt) {
                                min_R = R_n;
                                Vopt = V;
                                min_extent = diagDiff;
                                min_center =
                                        0.5 *
                                        ((X_Max * R_n.transpose()).diagonal() +
                                         (X_Min * R_n.transpose()).diagonal());
                            }
                        } else {
                            // Update X_Min and X_Max with nextNodeExt
                            std::vector<int> indicesMin = {prevNodeExt(0),
                                                           prevNodeExt(1),
                                                           prevNodeExt(2)};
                            for (int idx = 0; idx < 3; ++idx) {
                                if (nextNodeExt(idx) > -1) {
                                    indicesMin.push_back(nextNodeExt(idx));
                                }
                            }
                            Eigen::MatrixXd X_Min_updated(indicesMin.size(), 3);
                            for (size_t idx = 0; idx < indicesMin.size();
                                 ++idx) {
                                X_Min_updated.row(idx) = X.row(indicesMin[idx]);
                            }
                            X_Min_updated = X_Min_updated * R_n.transpose();

                            // Update X_Max with edges and nodes
                            std::vector<int> indicesMax = {
                                    static_cast<int>(hullEdges(i, 0)),
                                    static_cast<int>(hullEdges(i, 1)),
                                    static_cast<int>(hullEdges(j, 0)),
                                    static_cast<int>(hullEdges(j, 1)),
                                    prevNodeExt(3)};
                            if (nextNodeExt(3) > -1) {
                                indicesMax.push_back(nextNodeExt(3));
                            }

                            Eigen::MatrixXd X_Max_updated(indicesMax.size(), 3);
                            for (size_t idx = 0; idx < indicesMax.size();
                                 ++idx) {
                                X_Max_updated.row(idx) = X.row(indicesMax[idx]);
                            }
                            X_Max_updated = X_Max_updated * R_n.transpose();

                            // Compute V
                            Eigen::RowVectorXd X_Max_rowwise =
                                    X_Max_updated.colwise().maxCoeff();
                            Eigen::RowVectorXd X_Min_rowwise =
                                    X_Min_updated.colwise().minCoeff();

                            double V = (X_Max_rowwise - X_Min_rowwise).prod();

                            if (V < 0) {
                                std::cout << "--------------> X_Max_updated = "
                                          << X_Max_updated << std::endl;
                                std::cout << "--------------> X_Min_updated = "
                                          << X_Min_updated << std::endl;
                                std::cout << "--------------> rowwise = "
                                          << X_Max_rowwise - X_Min_rowwise
                                          << std::endl;
                            }

                            if (V < Vopt) {
                                min_R = R_n;
                                Vopt = V;
                                min_extent = X_Max_rowwise - X_Min_rowwise;
                                min_center =
                                        0.5 * (X_Max_rowwise + X_Min_rowwise);
                            }
                        }
                    }

                    // Update nodeExt and prevNodeExt
                    for (size_t m = 0; m < update_indices.size(); ++m) {
                        int n = update_indices[m];
                        nodeExt[n] = nodes_p2[n];
                        prevNodeExt(n) = nextNodeExt(n);
                    }

                    // Update theta1 for the next iteration
                    theta1 = theta1_new + tol;

                }  // End of while loop
            }  // End of loop over k
        }
    }

    Vopt = Vopt * std::pow(normFactor, 3);

    // If no better bounding box was found, return the default OBB
    if (Vopt == std::numeric_limits<double>::max()) {
        utility::LogError("Failed to find minimal oriented bounding box.");
        return OrientedBoundingBox();
    }

    // Scale back to original dimensions
    min_center *= normFactor;
    min_extent *= normFactor;
    min_center += mean;

    // Create the OrientedBoundingBox with the minimal volume
    OrientedBoundingBox obb;
    obb.R_ = min_R;
    obb.extent_ = min_extent;
    obb.center_ = min_center;
    return obb;
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Clear() {
    min_bound_.setZero();
    max_bound_.setZero();
    return *this;
}

bool AxisAlignedBoundingBox::IsEmpty() const { return Volume() <= 0; }

Eigen::Vector3d AxisAlignedBoundingBox::GetMinBound() const {
    return min_bound_;
}

Eigen::Vector3d AxisAlignedBoundingBox::GetMaxBound() const {
    return max_bound_;
}

Eigen::Vector3d AxisAlignedBoundingBox::GetCenter() const {
    return (min_bound_ + max_bound_) * 0.5;
}

AxisAlignedBoundingBox AxisAlignedBoundingBox::GetAxisAlignedBoundingBox()
        const {
    return *this;
}

OrientedBoundingBox AxisAlignedBoundingBox::GetOrientedBoundingBox(
        bool robust) const {
    return OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(*this);
}

OrientedBoundingBox AxisAlignedBoundingBox::GetMinimalOrientedBoundingBox(
        bool robust) const {
    return OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(*this);
}

AxisAlignedBoundingBox::AxisAlignedBoundingBox(const Eigen::Vector3d& min_bound,
                                               const Eigen::Vector3d& max_bound)
    : Geometry3D(Geometry::GeometryType::AxisAlignedBoundingBox),
      min_bound_(min_bound),
      max_bound_(max_bound),
      color_(1, 1, 1) {
    if ((max_bound_.array() < min_bound_.array()).any()) {
        open3d::utility::LogWarning(
                "max_bound {} of bounding box is smaller than min_bound {} "
                "in "
                "one or more axes. Fix input values to remove this "
                "warning.",
                max_bound_, min_bound_);
        max_bound_ = max_bound.cwiseMax(min_bound);
        min_bound_ = max_bound.cwiseMin(min_bound);
    }
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Transform(
        const Eigen::Matrix4d& transformation) {
    utility::LogError(
            "A general transform of a AxisAlignedBoundingBox would not be "
            "axis "
            "aligned anymore, convert it to a OrientedBoundingBox first");
    return *this;
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Translate(
        const Eigen::Vector3d& translation, bool relative) {
    if (relative) {
        min_bound_ += translation;
        max_bound_ += translation;
    } else {
        const Eigen::Vector3d half_extent = GetHalfExtent();
        min_bound_ = translation - half_extent;
        max_bound_ = translation + half_extent;
    }
    return *this;
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Scale(
        const double scale, const Eigen::Vector3d& center) {
    min_bound_ = center + scale * (min_bound_ - center);
    max_bound_ = center + scale * (max_bound_ - center);
    return *this;
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Rotate(
        const Eigen::Matrix3d& rotation, const Eigen::Vector3d& center) {
    utility::LogError(
            "A rotation of an AxisAlignedBoundingBox would not be "
            "axis-aligned "
            "anymore, convert it to an OrientedBoundingBox first");
    return *this;
}

std::string AxisAlignedBoundingBox::GetPrintInfo() const {
    return fmt::format("[({:.4f}, {:.4f}, {:.4f}) - ({:.4f}, {:.4f}, {:.4f})]",
                       min_bound_(0), min_bound_(1), min_bound_(2),
                       max_bound_(0), max_bound_(1), max_bound_(2));
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::operator+=(
        const AxisAlignedBoundingBox& other) {
    if (IsEmpty()) {
        min_bound_ = other.min_bound_;
        max_bound_ = other.max_bound_;
    } else if (!other.IsEmpty()) {
        min_bound_ = min_bound_.array().min(other.min_bound_.array()).matrix();
        max_bound_ = max_bound_.array().max(other.max_bound_.array()).matrix();
    }
    return *this;
}

AxisAlignedBoundingBox AxisAlignedBoundingBox::CreateFromPoints(
        const std::vector<Eigen::Vector3d>& points) {
    AxisAlignedBoundingBox box;
    if (points.empty()) {
        utility::LogWarning(
                "The number of points is 0 when creating axis-aligned "
                "bounding "
                "box.");
        box.min_bound_ = Eigen::Vector3d(0.0, 0.0, 0.0);
        box.max_bound_ = Eigen::Vector3d(0.0, 0.0, 0.0);
    } else {
        box.min_bound_ = std::accumulate(
                points.begin(), points.end(), points[0],
                [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                    return a.array().min(b.array()).matrix();
                });
        box.max_bound_ = std::accumulate(
                points.begin(), points.end(), points[0],
                [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                    return a.array().max(b.array()).matrix();
                });
    }
    return box;
}

double AxisAlignedBoundingBox::Volume() const { return GetExtent().prod(); }

std::vector<Eigen::Vector3d> AxisAlignedBoundingBox::GetBoxPoints() const {
    std::vector<Eigen::Vector3d> points(8);
    Eigen::Vector3d extent = GetExtent();
    points[0] = min_bound_;
    points[1] = min_bound_ + Eigen::Vector3d(extent(0), 0, 0);
    points[2] = min_bound_ + Eigen::Vector3d(0, extent(1), 0);
    points[3] = min_bound_ + Eigen::Vector3d(0, 0, extent(2));
    points[4] = max_bound_;
    points[5] = max_bound_ - Eigen::Vector3d(extent(0), 0, 0);
    points[6] = max_bound_ - Eigen::Vector3d(0, extent(1), 0);
    points[7] = max_bound_ - Eigen::Vector3d(0, 0, extent(2));
    return points;
}

std::vector<size_t> AxisAlignedBoundingBox::GetPointIndicesWithinBoundingBox(
        const std::vector<Eigen::Vector3d>& points) const {
    std::vector<size_t> indices;
    for (size_t idx = 0; idx < points.size(); idx++) {
        const auto& point = points[idx];
        if (point(0) >= min_bound_(0) && point(0) <= max_bound_(0) &&
            point(1) >= min_bound_(1) && point(1) <= max_bound_(1) &&
            point(2) >= min_bound_(2) && point(2) <= max_bound_(2)) {
            indices.push_back(idx);
        }
    }
    return indices;
}

}  // namespace geometry
}  // namespace open3d
