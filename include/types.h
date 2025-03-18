#ifndef TYPES_H
#define TYPES_H

#include <opencv2/opencv.hpp>

#include "consts.h"
#include "misc.h"

class Card {
  public:
    // Constructor to initialize Card with a frame and corners
    Card(cv::Mat frame, std::vector<cv::Point2f> corners, int id)
        : frame(frame), corners(corners), id(id) {}

    bool is_inside(cv::Point2f point) { return is_point_inside_hull(point, corners); }
    // You can add other member functions as needed, e.g., displaying the card

    cv::Mat display_frame() {
        //
        cv::Mat show_frame;

        if (!matched_card.empty()) {
            cv::Mat scaled_match;
            double scale = static_cast<double>(unwarped_image.rows) / matched_card.rows;
            cv::resize(matched_card, scaled_match, cv::Size(), scale, scale);
            cv::hconcat(unwarped_image, scaled_match, show_frame);
        } else if (!unwarped_image.empty()) {
            // std::cout << "showing unwarped \n";
            show_frame = unwarped_image.clone();
        } else {
            std::cout << "showing orifg \n";
            show_frame = frame.clone();
            for (const auto &point : corners) {
                cv::circle(show_frame, point, 5, cv::Scalar(0, 0, 255),
                           -1); // Draw red points (radius 5)
            }
        }
        return show_frame;
    }

    void display_card() {
        if (b_show) {
            //
            cv::Mat show_frame = display_frame();
            std::string windowName =
                "Card " + std::to_string(id);   // Create a window name based on the ID
            cv::imshow(windowName, show_frame); // Display the card frame
            if (cv::waitKey(1) == 27) {}
        }
    }

    void unwarp() {
        auto sorted_points = sort_corners_clockwise(corners);
        unwarped_image = unwarp_card(frame, sorted_points);
    }

    cv::Mat frame;
    cv::Mat unwarped_image;
    cv::Mat matched_card;

  private:
    std::vector<cv::Point2f> corners;
    bool b_show = true;
    int id;
};

class FPSOverlay {
  public:
    FPSOverlay(double alpha = 0.9)
        : alpha(alpha), fps(0), prev_time(std::chrono::high_resolution_clock::now()) {}

    void update() {
        // Compute raw FPS
        auto curr_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = curr_time - prev_time;
        double raw_fps = 1.0 / duration.count();
        prev_time = curr_time;

        // Apply smoothing
        fps = alpha * fps + (1.0 - alpha) * raw_fps;

        // Prevent initial bias
        if (fps <= 0) { fps = raw_fps; }
    }

    void overlay_fps(cv::Mat &frame) {
        // Overlay FPS on the frame
        std::string text = "FPS: " + std::to_string(static_cast<int>(fps));
        cv::putText(frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 0), 2);
    }

  private:
    double alpha;                                             // Smoothing factor
    double fps;                                               // Smoothed FPS value
    std::chrono::high_resolution_clock::time_point prev_time; // Previous timestamp
};

struct SharedFrame {
    cv::Mat frame;
    FPSOverlay fps;
    std::mutex mutex;
};
struct SharedCards {
    std::vector<Card> cards;
    std::mutex mutex;
};

struct CardEntry {
    std::string name;
    std::string id;
    std::string set;
    std::string image_path;
};

#endif