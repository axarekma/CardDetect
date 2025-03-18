#ifndef COLLISION_H
#define COLLISION_H

#include <cmath>
#include <limits>
#include <opencv2/opencv.hpp>

inline double nearest_point_on_line_t(const cv::Point2f &l1, const cv::Point2f &d,
                                      const cv::Point2f &p) {
    cv::Point2f temp = p - l1;
    double t = temp.dot(d) / d.dot(d);
    return t;
}
inline void nearest_points_on_two_lines(const cv::Point2f &a1, const cv::Point2f &d1,
                                        const cv::Point2f &b1, const cv::Point2f &d2, double &t1,
                                        double &t2) {
    const double EPS = 1e-15;

    cv::Point2f b1ma1 = b1 - a1;

    double d1d1 = d1.dot(d1);
    double d1d2 = d1.dot(d2);
    double d2d2 = d2.dot(d2);

    double b1ma1dotd1 = b1ma1.dot(d1);
    double b1ma1dotd2 = b1ma1.dot(d2);

    double D = d1d1 * d2d2 - d1d2 * d1d2;
    double Dt1 = b1ma1dotd1 * d2d2 - d1d2 * b1ma1dotd2;
    double Dt2 = b1ma1dotd1 * d1d2 - d1d1 * b1ma1dotd2;

    if (std::abs(D) > EPS) {
        t1 = Dt1 / D;
        t2 = Dt2 / D;
    } else {
        t1 = 0.5;
        cv::Point2f p1 = a1 + d1 * t1;
        t2 = nearest_point_on_line_t(b1, d2, p1);
    }
}

inline cv::Point2f nearest_point_on_line_segment(const cv::Point2f &l1, const cv::Point2f &l2,
                                                 const cv::Point2f &p) {
    cv::Point2f d = l2 - l1;
    double t = (p - l1).dot(d) / d.dot(d);
    t = std::clamp(t, 0.0, 1.0);
    return l1 + d * t;
}

inline void nearest_points_on_two_line_segments(const cv::Point2f &a1, const cv::Point2f &a2,
                                                const cv::Point2f &b1, const cv::Point2f &b2,
                                                cv::Point2f &p1, cv::Point2f &p2) {
    cv::Point2f d1 = a2 - a1;
    cv::Point2f d2 = b2 - b1;

    double t1_inf = 0, t2_inf = 0;
    nearest_points_on_two_lines(a1, d1, b1, d2, t1_inf, t2_inf);

    double dist_inf = std::numeric_limits<double>::infinity();
    if (t1_inf >= 0 && t1_inf <= 1 && t2_inf >= 0 && t2_inf <= 1) {
        cv::Point2f c1 = a1 + d1 * t1_inf;
        cv::Point2f c2 = b1 + d2 * t2_inf;
        dist_inf = cv::norm(c1 - c2);
    }

    cv::Point2f aBeginNearest = nearest_point_on_line_segment(b1, b2, a1);
    cv::Point2f aEndNearest = nearest_point_on_line_segment(b1, b2, a2);
    cv::Point2f bBeginNearest = nearest_point_on_line_segment(a1, a2, b1);
    cv::Point2f bEndNearest = nearest_point_on_line_segment(a1, a2, b2);

    double dist_aBegin = cv::norm(a1 - aBeginNearest);
    double dist_aEnd = cv::norm(a2 - aEndNearest);
    double dist_bBegin = cv::norm(b1 - bBeginNearest);
    double dist_bEnd = cv::norm(b2 - bEndNearest);

    if (dist_inf <= dist_aBegin && dist_inf <= dist_aEnd && dist_inf <= dist_bBegin &&
        dist_inf <= dist_bEnd) {
        p1 = a1 + d1 * t1_inf;
        p2 = b1 + d2 * t2_inf;
    } else if (dist_aBegin <= dist_aEnd && dist_aBegin <= dist_bBegin && dist_aBegin <= dist_bEnd) {
        p1 = a1;
        p2 = aBeginNearest;
    } else if (dist_aEnd <= dist_bBegin && dist_aEnd <= dist_bEnd) {
        p1 = a2;
        p2 = aEndNearest;
    } else if (dist_bBegin <= dist_bEnd) {
        p1 = bBeginNearest;
        p2 = b1;
    } else {
        p1 = bEndNearest;
        p2 = b2;
    }
}

#endif