#ifndef CONSTS_H
#define CONSTS_H

#include <array>
#include <opencv2/opencv.hpp>

inline std::array<cv::Scalar, 10> DEFAULTCOLORS = {
    // The default colors from matplotlib, they look nice.
    cv::Scalar(31, 119, 180),  // Blue
    cv::Scalar(255, 127, 14),  // Orange
    cv::Scalar(44, 160, 44),   // Green
    cv::Scalar(214, 39, 40),   // Red
    cv::Scalar(148, 103, 189), // Purple
    cv::Scalar(140, 86, 75),   // Brown
    cv::Scalar(227, 119, 194), // Pink
    cv::Scalar(127, 127, 127), // Gray
    cv::Scalar(188, 189, 34),  // Yellow
    cv::Scalar(23, 190, 207)   // Cyan
};

constexpr struct {
    // This is the size of the small images in scryfall
    int W = 146;
    int H = 204;
} IMAGE;

#endif // CONSTS_H