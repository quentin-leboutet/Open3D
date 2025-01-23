// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/BoundingVolume.h"

#include <Eigen/Eigenvalues>
#include <numeric>

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/Qhull.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace geometry {

OrientedBoundingEllipsoid& OrientedBoundingEllipsoid::Clear() {
    center_.setZero();
    radii_.setZero();
    R_ = Eigen::Matrix3d::Identity();
    color_.setOnes();
    return *this;
}

bool OrientedBoundingEllipsoid::IsEmpty() const { return Volume() <= 0; }

Eigen::Vector3d OrientedBoundingEllipsoid::GetMinBound() const {
    auto points = GetEllipsoidPoints();
    return ComputeMinBound(points);
}

Eigen::Vector3d OrientedBoundingEllipsoid::GetMaxBound() const {
    auto points = GetEllipsoidPoints();
    return ComputeMaxBound(points);
}

Eigen::Vector3d OrientedBoundingEllipsoid::GetCenter() const { return center_; }

AxisAlignedBoundingBox OrientedBoundingEllipsoid::GetAxisAlignedBoundingBox()
        const {
    return AxisAlignedBoundingBox::CreateFromPoints(GetEllipsoidPoints());
}

OrientedBoundingBox OrientedBoundingEllipsoid::GetOrientedBoundingBox(
        bool) const {
    return OrientedBoundingBox::CreateFromPoints(GetEllipsoidPoints());
}

OrientedBoundingBox OrientedBoundingEllipsoid::GetMinimalOrientedBoundingBox(
        bool robust) const {
    return OrientedBoundingBox::CreateFromPoints(GetEllipsoidPoints());
}

OrientedBoundingEllipsoid
OrientedBoundingEllipsoid::GetOrientedBoundingEllipsoid(bool) const {
    return *this;
}

OrientedBoundingEllipsoid& OrientedBoundingEllipsoid::Transform(
        const Eigen::Matrix4d& transformation) {
    utility::LogError(
            "A general transform of an OrientedBoundingEllipsoid is not "
            "implemented. "
            "Call Translate, Scale, and Rotate.");
    return *this;
}

OrientedBoundingEllipsoid& OrientedBoundingEllipsoid::Translate(
        const Eigen::Vector3d& translation, bool relative) {
    if (relative) {
        center_ += translation;
    } else {
        center_ = translation;
    }
    return *this;
}

OrientedBoundingEllipsoid& OrientedBoundingEllipsoid::Scale(
        const double scale, const Eigen::Vector3d& center) {
    radii_ *= scale;
    center_ = scale * (center_ - center) + center;
    return *this;
}

OrientedBoundingEllipsoid& OrientedBoundingEllipsoid::Rotate(
        const Eigen::Matrix3d& R, const Eigen::Vector3d& center) {
    R_ = R * R_;
    center_ = R * (center_ - center) + center;
    return *this;
}

double OrientedBoundingEllipsoid::Volume() const {
    return 4 * M_PI * radii_(0) * radii_(1) * radii_(2) / 3.0;
}

std::vector<Eigen::Vector3d> OrientedBoundingEllipsoid::GetEllipsoidPoints()
        const {
    Eigen::Vector3d x_axis = R_ * Eigen::Vector3d(radii_(0), 0, 0);
    Eigen::Vector3d y_axis = R_ * Eigen::Vector3d(0, radii_(1), 0);
    Eigen::Vector3d z_axis = R_ * Eigen::Vector3d(0, 0, radii_(2));
    std::vector<Eigen::Vector3d> points(6);
    points[0] = center_ + R_ * x_axis;
    points[1] = center_ - R_ * x_axis;
    points[2] = center_ + y_axis;
    points[3] = center_ - y_axis;
    points[4] = center_ + z_axis;
    points[5] = center_ - z_axis;
    return points;
}

OrientedBoundingEllipsoid OrientedBoundingEllipsoid::CreateFromPoints(
        const std::vector<Eigen::Vector3d>& points, bool robust) {
    // ------------------------------------------------------------
    // 0) Compute the convex hull of the input point cloud
    // ------------------------------------------------------------
    if (points.empty()) {
        utility::LogError("CreateFromPoints: Input point set is empty.");
        return OrientedBoundingEllipsoid();
    }
    std::shared_ptr<TriangleMesh> hullMesh;
    std::tie(hullMesh, std::ignore) = Qhull::ComputeConvexHull(points, robust);
    if (!hullMesh) {
        utility::LogError("Failed to compute convex hull.");
        return OrientedBoundingEllipsoid();
    }

    // Get convex hull vertices and triangles
    const std::vector<Eigen::Vector3d>& hullV = hullMesh->vertices_;
    // const std::vector<Eigen::Vector3i>& hullT = hullMesh->triangles_;
    const int numVertices = static_cast<int>(hullV.size());
    // int numTriangles = static_cast<int>(hullT.size());

    auto mapToClosestIdentity = [&](OrientedBoundingEllipsoid& obel) {
        Eigen::Matrix3d& R = obel.R_;
        Eigen::Vector3d& radii = obel.radii_;
        Eigen::Vector3d col[3] = {R.col(0), R.col(1), R.col(2)};
        double best_score = -1e9;
        Eigen::Matrix3d best_R;
        Eigen::Vector3d best_radii;
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
                    best_radii(0) = radii(p[0]);
                    best_radii(1) = radii(p[1]);
                    best_radii(2) = radii(p[2]);
                }
            }
        }

        // Update the OBB with the best orientation found
        obel.R_ = best_R;
        obel.radii_ = best_radii;
    };

    auto KhachiyanAlgo = [&](const Eigen::MatrixXd& A, double eps, int maxiter,
                             Eigen::MatrixXd& Q, Eigen::VectorXd& c) -> double {
        // Initialize uniform weights: p_i = 1/N for all i
        Eigen::VectorXd p = Eigen::VectorXd::Constant(
                numVertices, 1.0 / static_cast<double>(numVertices));

        // Lift matrix A to Ap by adding a bottom row of ones
        Eigen::MatrixXd Ap(4, numVertices);
        Ap.topRows(3) = A;
        Ap.row(3).setOnes();

        double currentEps = 2.0 * eps;  // Start with a large difference
        int iter = 0;

        // Main iterative loop
        while (iter < maxiter && currentEps > eps) {
            // Compute Λ_p = Ap * diag(p) * Ap^T in a more efficient way
            // ApP = Ap * diag(p)  => shape: (4) x N
            // Then Lambda_p = ApP * Ap^T => shape: (4) x (4)
            Eigen::MatrixXd ApP = Ap * p.asDiagonal();       // (4) x N
            Eigen::Matrix4d LambdaP = ApP * Ap.transpose();  // (4) x (4)

            // Compute inverse of Lambda_p via an LDLT factorization
            // (faster and more numerically stable than .inverse())
            Eigen::LDLT<Eigen::MatrixXd> ldltOfLambdaP(LambdaP);
            if (ldltOfLambdaP.info() != Eigen::Success) {
                throw std::runtime_error(
                        "LDLT decomposition failed. Matrix may be singular.");
            }

            // M = Ap^T * (Lambda_p^{-1} * Ap)
            // We'll do this in steps:
            //    1) X = Lambda_p^{-1} * Ap
            //    2) M = Ap^T * X
            // Dimensions:
            //    Lambda_p^{-1}: (4 x (4)
            //    Ap: (4) x N
            // => X: (4) x N
            // => M: Nx(4) * (4)xN -> NxN
            Eigen::MatrixXd X = ldltOfLambdaP.solve(Ap);  // (4) x N
            Eigen::MatrixXd M = Ap.transpose() * X;       // NxN

            // Find max diagonal element and index
            Eigen::Index maxIndex;
            double maxVal = M.diagonal().maxCoeff(&maxIndex);

            // Compute step size alpha (called step_size here)
            // Formula: alpha = (maxVal - 4) / ((4) * (maxVal - 1))
            const double step_size = (maxVal - 4) / (4 * (maxVal - 1.0));

            // Update weights p
            Eigen::VectorXd newP = (1.0 - step_size) * p;
            newP(maxIndex) += step_size;

            // Compute the change for the stopping criterion
            currentEps = (newP - p).norm();
            p.swap(newP);  // Efficient swap instead of copy

            ++iter;
        }

        // After convergence or reaching max iterations,
        // compute Q and center c for the ellipsoid.

        // 1) PN = A * diag(p) * A^T
        Eigen::MatrixXd AP = A * p.asDiagonal();  // 3 x N
        Eigen::Matrix3d PN = AP * A.transpose();  // 3 x 3

        // 2) M2 = A * p => a 3-dimensional vector
        Eigen::Vector3d M2 = A * p;  // 3 x 1

        // 3) M3 = M2 * M2^T => 3 x 3 outer product
        Eigen::Matrix3d M3 = M2 * M2.transpose();  // 3 x 3

        // 4) Invert (PN - M3) via LDLT
        Eigen::Matrix3d toInvert = PN - M3;  // 3 x 3
        Eigen::LDLT<Eigen::Matrix3d> ldltOfToInvert(toInvert);
        if (ldltOfToInvert.info() != Eigen::Success) {
            throw std::runtime_error(
                    "LDLT decomposition failed in final step.");
        }

        // Q = (toInvert)^{-1} / 3   => shape matrix of the ellipsoid
        // c = A * p                 => center of the ellipsoid
        Q = ldltOfToInvert.solve(Eigen::Matrix3d::Identity()) /
            static_cast<double>(3);
        c = M2;  // Already computed as A*p

        return currentEps;
    };

    // Assemble matrix A with dimensions d x n_points, where each column is a
    // point
    Eigen::MatrixXd A(3, numVertices);
    for (int i = 0; i < numVertices; ++i) {
        A.col(i) = hullV[i];
    }

    // Set parameters for Khachiyan's algorithm
    double eps = 1e-6;
    size_t maxiter = 1000;

    // Variables to store the resulting ellipsoid parameters
    Eigen::MatrixXd Q;
    Eigen::VectorXd c;

    // Call Khachiyan's algorithm to compute Q (shape matrix) and c (center)
    KhachiyanAlgo(A, eps, maxiter, Q, c);
    // Optionally check final_error for convergence quality

    // Use eigen-decomposition on Q to extract axes lengths and orientation
    // For ellipsoid defined by (x-c)^T Q (x-c) <= 1,
    // the eigenvectors of Q give orientation directions,
    // and the axes lengths (radii) are 1/sqrt(eigenvalues).
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(Q);
    if (eigenSolver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed.");
    }
    Eigen::VectorXd eigenvalues = eigenSolver.eigenvalues();
    Eigen::MatrixXd eigenvectors = eigenSolver.eigenvectors();

    // Compute radii = 1/sqrt(eigenvalues)
    Eigen::Vector3d radii = (1.0 / eigenvalues.array().sqrt()).matrix();

    // Construct the final oriented bounding ellipsoid
    OrientedBoundingEllipsoid obel;
    obel.center_ = c.head<3>();  // center vector of length 3
    obel.R_ = eigenvectors;      // orientation matrix
    obel.radii_ = radii;         // ellipsoid radii

    // Check oritntation and permute axes to closest identity
    mapToClosestIdentity(obel);

    return obel;
}

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

OrientedBoundingEllipsoid OrientedBoundingBox::GetOrientedBoundingEllipsoid(
        bool) const {
    return OrientedBoundingEllipsoid::CreateFromPoints(GetBoxPoints());
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

OrientedBoundingBox OrientedBoundingBox::CreateFromPointsMinimal(
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

OrientedBoundingEllipsoid AxisAlignedBoundingBox::GetOrientedBoundingEllipsoid(
        bool) const {
    return OrientedBoundingEllipsoid::CreateFromPoints(GetBoxPoints());
}

AxisAlignedBoundingBox::AxisAlignedBoundingBox(const Eigen::Vector3d& min_bound,
                                               const Eigen::Vector3d& max_bound)
    : Geometry3D(Geometry::GeometryType::AxisAlignedBoundingBox),
      min_bound_(min_bound),
      max_bound_(max_bound),
      color_(1, 1, 1) {
    if ((max_bound_.array() < min_bound_.array()).any()) {
        open3d::utility::LogWarning(
                "max_bound {} of bounding box is smaller than min_bound {} in "
                "one or more axes. Fix input values to remove this warning.",
                max_bound_, min_bound_);
        max_bound_ = max_bound.cwiseMax(min_bound);
        min_bound_ = max_bound.cwiseMin(min_bound);
    }
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Transform(
        const Eigen::Matrix4d& transformation) {
    utility::LogError(
            "A general transform of a AxisAlignedBoundingBox would not be axis "
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
            "A rotation of an AxisAlignedBoundingBox would not be axis-aligned "
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
                "The number of points is 0 when creating axis-aligned bounding "
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
