#ifndef MISC_H
#define MISC_H

#include "collision.h"
#include "consts.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

// Collection of all the stff that doesnt have a home (yet)

inline cv::Mat unwarp_card(const cv::Mat &frame, const std::vector<cv::Point2f> &corners) {

    // const int W = 146;
    // const int H = 204;
    constexpr int W = IMAGE.W;
    constexpr int H = IMAGE.H;
    const int padW = int(0.05 * W);
    const int padH = int(0.05 * H);

    // Calculate distances between consecutive corners
    std::vector<double> distances(4);
    for (int i = 0; i < 4; ++i) {
        distances[i] = cv::norm(corners[i] - corners[(i + 1) % 4]);
    }

    // cv::Mat src;
    std::vector<cv::Point2f> src;
    // Determine the correct ordering of corners
    if (distances[0] + distances[2] > distances[1] + distances[3]) {
        // Reorder corners if the sum of horizontal distances is greater than vertical
        std::vector<cv::Point2f> reordered(corners);
        std::rotate(reordered.begin(), reordered.begin() + 1, reordered.end());
        src = {cv::Point2f(reordered[0]), cv::Point2f(reordered[1]), cv::Point2f(reordered[2]),
               cv::Point2f(reordered[3])};
    } else {
        src = {cv::Point2f(corners[0]), cv::Point2f(corners[1]), cv::Point2f(corners[2]),
               cv::Point2f(corners[3])};
    }

    // Destination points (top-left, top-right, bottom-left, bottom-right)
    std::vector<cv::Point2f> dst = {cv::Point2f(W - padW, padH), cv::Point2f(padW, padH),
                                    cv::Point2f(padW, H - padH), cv::Point2f(W - padW, H - padH)};

    // Compute the perspective transform matrix
    cv::Mat matrix = cv::getPerspectiveTransform(src, dst);

    // Apply the perspective warp
    cv::Mat res;
    cv::warpPerspective(frame, res, matrix, cv::Size(W, H));

    return res;
}

inline std::vector<cv::Point2f> sort_corners_clockwise(const std::vector<cv::Point2f> &corners) {
    // Ensure there are exactly 4 points
    if (corners.size() != 4) {
        throw std::invalid_argument("Input must contain exactly 4 points.");
    }

    // Compute the centroid (center of mass)
    cv::Point2f center(0, 0);
    for (const auto &point : corners) {
        center += point;
    }
    center *= (1.0 / corners.size()); // Average to find the centroid

    // Compute angles of each point relative to the centroid
    std::vector<std::pair<double, cv::Point2f>> points_with_angles;
    for (const auto &point : corners) {
        auto angle = std::atan2(point.y - center.y, point.x - center.x);
        points_with_angles.push_back({angle, point});
    }

    // Sort by angle in descending order to achieve clockwise sorting
    std::sort(points_with_angles.begin(), points_with_angles.end(),
              [](const std::pair<double, cv::Point2f> &a, const std::pair<double, cv::Point2f> &b) {
                  return a.first > b.first; // Sort in descending order
              });

    // Extract sorted points
    std::vector<cv::Point2f> sorted_corners;
    for (const auto &pair : points_with_angles) {
        sorted_corners.push_back(pair.second);
    }

    return sorted_corners;
}

inline void printChannelMeans(const cv::Mat &image) {
    if (image.empty()) {
        std::cerr << "Error: Image is empty!" << std::endl;
        return;
    }

    if (image.channels() < 3) {
        std::cerr << "Error: Image does not have 3 channels!" << std::endl;
        return;
    }

    // Compute mean of each channel
    cv::Scalar meanValues = cv::mean(image);

    std::cout << "Mean values per channel:\n";
    std::cout << "Channel 0 (Blue or Red)  : " << meanValues[0] << std::endl;
    std::cout << "Channel 1 (Green)        : " << meanValues[1] << std::endl;
    std::cout << "Channel 2 (Red or Blue)  : " << meanValues[2] << std::endl;
}

typedef struct {
    int direction;
    double cost;
    cv::Point2f crossing_points;

} CostElement;

inline CostElement line_cost(const cv::Vec4i &line1, const cv::Vec4i &line2,
                             double paralell_cut = 20, double neg_multiplier = -2,
                             int base_cost = 10) {

    double p_limit = std::cos(paralell_cut * CV_PI / 180.0);

    cv::Point2f a1(static_cast<float>(line1[0]), static_cast<float>(line1[1]));
    cv::Point2f a2(static_cast<float>(line1[2]), static_cast<float>(line1[3]));
    cv::Point2f b1(static_cast<float>(line2[0]), static_cast<float>(line2[1]));
    cv::Point2f b2(static_cast<float>(line2[2]), static_cast<float>(line2[3]));

    cv::Point2f d1 = a2 - a1;
    cv::Point2f d2 = b2 - b1;
    double norm1 = norm(d1);
    double norm2 = norm(d2);
    double cos_alpha = std::abs(d1.dot(d2) / (norm1 * norm2));

    if (cos_alpha > p_limit) {
        cv::Point2f p1, p2;
        nearest_points_on_two_line_segments(a1, a2, b1, b2, p1, p2);
        cv::Point2f direction = p2 - p1;
        cv::Point2f crossing_point = (p1 + p2) * 0.5;
        double normd = norm(direction);

        double p_comp1 = std::abs(direction.dot(d1) / norm1);
        double p_comp2 = std::abs(direction.dot(d2) / norm2);
        double perp_comp1 = std::sqrt(normd * normd - p_comp1 * p_comp1);
        double perp_comp2 = std::sqrt(normd * normd - p_comp2 * p_comp2);

        double cost_parallel = (base_cost + p_comp1 + p_comp2) / (2 * (norm1 + norm2));
        double cost_perp = perp_comp1 + perp_comp2;

        return {0, cost_parallel + cost_perp, crossing_point};
    }

    double t1_inf = 0, t2_inf = 0;
    nearest_points_on_two_lines(a1, d1, b1, d2, t1_inf, t2_inf);

    cv::Point2f d1_c = (t1_inf < 0.5) ? -d1 : d1;
    cv::Point2f d2_c = (t2_inf > 0.5) ? -d2 : d2;
    double cprod = d1_c.x * d2_c.y - d1_c.y * d2_c.x;
    int turndir = (cprod > 0) ? -1 : 1;

    cv::Point2f c1 = a1 + t1_inf * d1;
    // cv::Point2f c2 = b1 + t2_inf * d2;

    double cost_a = std::max(-t1_inf, t1_inf - 1) * norm1;
    double cost_b = std::max(-t2_inf, t2_inf - 1) * norm2;

    if (cost_a < 0) cost_a *= neg_multiplier;
    if (cost_b < 0) cost_b *= neg_multiplier;

    double cost_length = base_cost + cost_a + cost_b;
    double total_length = norm1 + norm2 + cost_a + cost_b;

    return {turndir, cost_length / total_length, c1};
}

struct DFS {
    std::vector<int> best_path;

    std::vector<std::vector<double>> costs;
    std::vector<std::vector<int>> directions;
    std::vector<std::vector<cv::Point2f>> points;
    double best_cost = 10.0;
    int max_length = 6;
    int num_nodes = 10;

    DFS(int nodes) : num_nodes(nodes) {
        costs.resize(num_nodes,
                     std::vector<double>(num_nodes, std::numeric_limits<double>::infinity()));
        directions.resize(num_nodes, std::vector<int>(num_nodes, 0));
        points.resize(num_nodes, std::vector<cv::Point2f>(num_nodes, cv::Point2f(0.0f, 0.0f)));
    }
    void dfs_rec(std::vector<int> current_path, double current_cost, int current_counter) {
        // std::cout << "DFS\n";
        // for (auto &index : current_path) {
        //     std::cout << index << " ";
        // }
        // std::cout << " -- " << current_cost << " " << current_counter << "\n";

        // # Base case: Found a cycle
        if (current_path.size() > 0 && current_path.back() == current_path[0] &&
            current_counter == 4) {
            // std::cout << "Found path  -- " << current_cost << " " << current_counter << "\n";
            // If the found cycle has a lower cost, update the best known cycle
            if (current_cost < best_cost) {
                best_cost = current_cost;
                best_path = current_path;
            }
        }

        // # Pruning: Stop if cost is too high or cycle is too long
        if (current_cost > best_cost || current_path.size() > max_length) {
            // stopping short
            return;
        }
        int current = current_path.back();
        for (int neighbour = 0; neighbour < num_nodes; ++neighbour) {
            if (neighbour != current) {
                double cost = costs[current][neighbour];
                int dir = directions[current][neighbour];
                int node_parent_idx = std::max(0, static_cast<int>(current_path.size()) - 2);
                int parent = current_path[node_parent_idx];
                if (cost < best_cost && neighbour != parent) {
                    // recursive call
                    std::vector<int> branch = current_path; // Explicit copy
                    branch.push_back(neighbour);
                    dfs_rec(branch, current_cost + cost, current_counter + dir);
                }
            }
        }
    }
};

struct BFS {
    std::vector<int> best_path;

    std::vector<std::vector<double>> costs;
    std::vector<std::vector<int>> directions;
    std::vector<std::vector<cv::Point2f>> points;
    double best_cost = 10.0;
    int max_length = 6;
    int num_nodes = 10;

    BFS(int nodes) : num_nodes(nodes) {
        costs.resize(num_nodes,
                     std::vector<double>(num_nodes, std::numeric_limits<double>::infinity()));
        directions.resize(num_nodes, std::vector<int>(num_nodes, 0));
        points.resize(num_nodes, std::vector<cv::Point2f>(num_nodes, cv::Point2f(0.0f, 0.0f)));
    }

    void bfs(int start_node) {
        std::queue<std::tuple<std::vector<int>, double, int>> q;
        q.push({{start_node}, 0.0, 0});

        while (!q.empty()) {
            auto [current_path, current_cost, current_counter] = q.front();
            q.pop();

            int current = current_path.back();

            // Check if we found a cycle
            if (current_path.size() > 1 && current == start_node && current_counter == 4) {
                if (current_cost < best_cost) {
                    best_cost = current_cost;
                    best_path = current_path;
                }
                continue; // No need to expand this path further
            }

            // Prune paths that are already too costly or too long
            if (current_cost > best_cost || current_path.size() > max_length) { continue; }

            // Expand to neighbors
            for (int neighbour = 0; neighbour < num_nodes; ++neighbour) {
                if (neighbour != current) {
                    double cost = costs[current][neighbour];
                    int dir = directions[current][neighbour];

                    int node_parent_idx = std::max(0, static_cast<int>(current_path.size()) - 2);
                    int parent = current_path[node_parent_idx];

                    if (cost < best_cost && neighbour != parent) {
                        std::vector<int> new_path = current_path;
                        new_path.push_back(neighbour);
                        q.push({new_path, current_cost + cost, current_counter + dir});
                    }
                }
            }
        }
    }
};

inline int argmind(std::vector<double> array) {
    return static_cast<int>(
        std::distance(array.begin(), std::min_element(array.begin(), array.end())));
}

inline std::vector<cv::Point2f> find_best_cycle(std::vector<cv::Vec4i> edges,
                                                double max_cost = 2.0) {
    std::vector<cv::Point2f> retval;

    int num_nodes = static_cast<int>(edges.size());
    DFS dfs_solver(num_nodes);
    dfs_solver.best_cost = max_cost;

    // Initialize cost matrix using line_cost function
    dfs_solver.costs.resize(
        num_nodes, std::vector<double>(num_nodes, std::numeric_limits<double>::infinity()));

    // std::cout << "Precalculate costs ... \n";
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (i != j) {
                CostElement cost = line_cost(edges[i], edges[j]);
                dfs_solver.costs[i][j] = cost.cost;
                dfs_solver.directions[i][j] = cost.direction;
                dfs_solver.points[i][j] = cost.crossing_points;
            }
        }
    }

    std::vector<double> min_rows;
    min_rows.reserve(num_nodes);

    for (const auto &row : dfs_solver.costs) {
        auto min_el = std::min_element(row.begin(), row.end());
        min_rows.push_back(*min_el);
    }

    // Start from the point with best cost
    // std::cout << "DFS pass  n=" << num_nodes << "\n";
    int start = argmind(min_rows);
    dfs_solver.dfs_rec({start}, 0.0, 0);
    if (dfs_solver.best_path.size() < 4) { return retval; }

    // check if thepath neighbout has a better path
    start = dfs_solver.best_path[2];
    dfs_solver.dfs_rec({start}, 0.0, 0);
    // std::cout << " Solving DFS: n=" << num_nodes;
    // std::cout << " cost: " << dfs_solver.best_cost;
    // std::cout << " size: " << dfs_solver.best_path.size();
    // std::cout << '\n';

    if (dfs_solver.best_cost < max_cost) {
        // Convert best path indices to cv::Point format
        for (int i = 0; i < dfs_solver.best_path.size() - 1; i++) {
            // for (int idx : dfs_solver.best_path) {
            // std::cout << i << ' ';
            int ind1 = dfs_solver.best_path[i];
            int ind2 = dfs_solver.best_path[i + 1];
            retval.emplace_back(dfs_solver.points[ind1][ind2]);
        }
    }

    return retval;
}

// Expand a contour outward by a fixed absolute distance
inline std::vector<cv::Point2f> expand_contour_absolute(const std::vector<cv::Point2f> &points,
                                                        float expansion_distance = 5.0) {
    cv::Point2f centroid(0, 0);
    for (const auto &p : points) {
        centroid += p;
    }
    centroid *= (1.0f / points.size());

    std::vector<cv::Point2f> expanded_points;
    for (const auto &p : points) {
        cv::Point2f direction = p - centroid;
        float norm = static_cast<float>(cv::norm(direction));
        cv::Point2f expanded_p = (norm > 0) ? (p + (direction / norm) * expansion_distance) : p;
        expanded_points.push_back(expanded_p);
    }
    return expanded_points;
}

inline void plot_contour_on_image(const std::vector<cv::Point2f> &points, cv::Mat &image,
                                  cv::Scalar color = cv::Scalar(255, 255, 255), int size = 5) {
    for (const auto &point : points) {
        cv::circle(image, point, size, color, -1);
    }
    return;
}

inline void plot_lines_on_image(const std::vector<cv::Vec4i> &lines, cv::Mat &image,
                                cv::Scalar color = cv::Scalar(0, 0, 255)) {
    for (auto line : lines) {
        cv::Point2i p1(line[0], line[1]), p2(line[2], line[3]);
        cv::line(image, p1, p2, color);
    }
    return;
}

inline bool is_point_inside_hull(const cv::Point2f &point, const std::vector<cv::Point2f> &hull) {
    return cv::pointPolygonTest(hull, point, false) >= 0; // >= 0 means inside or on edge
}

inline std::vector<cv::Vec4i>
filter_lines_outside_cycle(const std::vector<cv::Vec4i> &lines,
                           const std::vector<cv::Point2f> &cycle_nodes, float expand = 10.0) {

    if (cycle_nodes.size() < 3) return lines; // No valid cycle

    // Expand the convex hull
    std::vector<cv::Point2f> expanded_contour = expand_contour_absolute(cycle_nodes, expand);

    // Filter lines based on midpoint
    std::vector<cv::Vec4i> filtered_lines;
    for (const auto &line : lines) {
        cv::Point2f p1(static_cast<float>(line[0]), static_cast<float>(line[1]));
        cv::Point2f p2(static_cast<float>(line[2]), static_cast<float>(line[3]));
        cv::Point2f midpoint = (p1 + p2) * 0.5;

        if (!is_point_inside_hull(midpoint, expanded_contour)) { filtered_lines.push_back(line); }
    }

    return filtered_lines;
}

inline std::vector<cv::Point2f> toPoint2f(const std::vector<cv::Point> &points) {
    std::vector<cv::Point2f> points2f;
    points2f.reserve(points.size()); // Preallocate memory for efficiency

    for (const auto &pt : points) {
        points2f.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
    }

    return points2f;
}

inline std::vector<cv::Point2f> approximate_quadrilateral(const std::vector<cv::Point2f> &contour) {
    std::vector<cv::Point2f> approx;

    double epsilon = 0.02 * cv::arcLength(contour, true); // Adjust approximation factor
    cv::approxPolyDP(contour, approx, epsilon, true);     // Approximate the contour

    if (approx.size() == 4) {
        return std::vector<cv::Point2f>(approx.begin(), approx.end()); // Convert to float
    } else {
        return {}; // Return empty vector if approximation fails
    }
}

#endif