#ifndef FRAME_PROCESSOR_H
#define FRAME_PROCESSOR_H

#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <vector>

#include "SimilarityModel.h"
#include "misc.h"
#include "types.h"

class FrameProcessor {
  public:
    void process_frames(SharedFrame &input, SharedFrame &output, SharedCards &cards,
                        bool &keepRunning);

    std::vector<cv::Vec4i> get_edge_lines(const cv::Mat &image, cv::Mat *edgesOut);

    // binary amsk version
    cv::Mat process_image(const cv::Mat &frame);
    cv::Mat fill_holes(const cv::Mat &image_in);
    std::vector<std::vector<cv::Point2f>> find_cards(const cv::Mat &frame);

    // parameters for the detection
    int threshold = 10;
    int gaussWindow = 7;
    int minLineLength = 50;
    int maxLineGap = 10;
};

#endif
