#include "FrameProcessor.h"

void FrameProcessor::process_frames(SharedFrame &input, SharedFrame &output, SharedCards &cards,
                                    bool &keepRunning) {
    while (keepRunning) {
        cv::Mat raw;
        cv::Mat edges;
        std::vector<Card> frame_cards;
        // load frame
        if (input.frame.empty()) {
            continue; // Skip this iteration if frame is not ready
        }
        {
            std::lock_guard<std::mutex> lock(input.mutex);
            raw = input.frame.clone();
        }

        std::vector<cv::Vec4i> frame_lines = get_edge_lines(raw, &edges);
        // plot edges on processed frame
        if (edges.channels() == 1) { cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR); }
        plot_lines_on_image(frame_lines, edges, cv::Scalar(0, 0, 255));

        // find cards with DFS on lines
        std::vector<cv::Point2f> cycle = find_best_cycle(frame_lines);
        std::vector<cv::Vec4i> filtered_lines = filter_lines_outside_cycle(frame_lines, cycle);
        int cardId = 0;
        int HARD_CUTOFF = 10;
        while (cycle.size() > 0 && cardId < HARD_CUTOFF) {
            std::vector<cv::Point2f> cycle_approx = approximate_quadrilateral(cycle);
            if (cycle_approx.size() == 4) {
                // TODO: unnecessary copy of image
                plot_contour_on_image(cycle_approx, edges, DEFAULTCOLORS[cardId]);
                frame_cards.push_back(Card(raw, cycle_approx, cardId));
            }
            cardId++;
            cycle = find_best_cycle(filtered_lines);
            filtered_lines = filter_lines_outside_cycle(filtered_lines, cycle);
        }

        // update processedframe
        {
            std::lock_guard<std::mutex> lock(output.mutex);
            output.frame = edges;
            output.fps.update();
        }
        // update cards
        {
            std::lock_guard<std::mutex> lock(cards.mutex);
            cards.cards = frame_cards;
        }
    }
}

std::vector<cv::Vec4i> FrameProcessor::get_edge_lines(const cv::Mat &image,
                                                      cv::Mat *edgesOut = nullptr) {
    cv::Mat gbImage, edges;
    // Apply Gaussian blur
    cv::GaussianBlur(image, gbImage, cv::Size(gaussWindow, gaussWindow), 0);
    // Apply Canny edge detection
    cv::Canny(gbImage, edges, 50, 150, 3);
    // Apply the Probabilistic Hough Transform
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180, threshold, minLineLength, maxLineGap);
    // Output edges if requested
    if (edgesOut) { *edgesOut = edges.clone(); }
    return lines;
}

cv::Mat FrameProcessor::process_image(const cv::Mat &frame) {
    cv::Mat gray, edges, edges_filled;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 20, 150);
    // Create a kernel for dilation
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
    cv::dilate(edges, edges, kernel);
    edges_filled = fill_holes(edges);
    kernel = cv::Mat::ones(7, 7, CV_8U);
    cv::morphologyEx(edges_filled, edges_filled, cv::MORPH_OPEN, kernel);
    return edges_filled;
}

cv::Mat FrameProcessor::fill_holes(const cv::Mat &image_in) {
    cv::Mat image = image_in.clone();

    // Flood-fill from the borders (in-place)
    cv::floodFill(image, cv::Point(0, 0), cv::Scalar(255));              // Top-left corner
    cv::floodFill(image, cv::Point(image.cols - 1, 0), cv::Scalar(255)); // Top-right corner
    cv::floodFill(image, cv::Point(0, image.rows - 1), cv::Scalar(255)); // Bottom-left corner
    cv::floodFill(image, cv::Point(image.cols - 1, image.rows - 1),
                  cv::Scalar(255)); // Bottom-right corner

    // Perform a bitwise OR with the original image and the inverted filled image
    return image_in | ~image;
}

std::vector<std::vector<cv::Point2f>> FrameProcessor::find_cards(const cv::Mat &frame) {
    std::vector<std::vector<cv::Point2f>> retval;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(frame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Process each contour
    for (auto &cnt : contours) {
        // Get convex hull
        std::vector<cv::Point> hull;
        cv::convexHull(cnt, hull);

        // Approximate polygon
        std::vector<cv::Point> approx;
        cv::approxPolyDP(hull, approx, 0.05 * cv::arcLength(cnt, true), true);

        // Calculate areas
        double contour_area = cv::contourArea(cnt);
        double hull_area = cv::contourArea(hull);
        double approx_area = cv::contourArea(approx);

        // Compute ratios
        double convex_ratio = (hull_area > 0) ? contour_area / hull_area : 0;
        double approx_ratio = (approx_area > 0) ? contour_area / approx_area : 0;

        // Check criteria
        bool criteria = (approx.size() == 4) && (hull_area > 50 * 50) && (convex_ratio > 0.7) &&
                        (approx_ratio > 0.7);

        if (criteria) { retval.push_back(toPoint2f(approx)); }
    }
    return retval;
}
