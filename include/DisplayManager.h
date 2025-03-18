#ifndef DISPLAY_MANAGER_H
#define DISPLAY_MANAGER_H

#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>

#include "SimilarityModel.h"
#include "misc.h"
#include "types.h"

class DisplayManager {
  public:
    FPSOverlay FPSclock;
    cv::Mat leftImage;
    cv::Mat rightImage;
    cv::Mat combinedView;
    int leftWidth;
    bool windowsInitialized = false;
    SimilarityModel *model;
    std::vector<Card> local_cards;

    void set_model(SimilarityModel &a_model) { this->model = &a_model; };

    void display_frames(SharedFrame &frame_raw, SharedFrame &frame_proc, SharedCards &cards,
                        bool &keepRunning, int maxWidth);

    void onMouse(int event, int x, int y, int flags);
    static void mouseCallbackWrapper(int event, int x, int y, int flags, void *userdata) {
        DisplayManager *self = static_cast<DisplayManager *>(userdata);
        self->onMouse(event, x, y, flags);
    }
};

#endif
