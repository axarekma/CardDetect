#ifndef CAMERA_CAPTURE_H
#define CAMERA_CAPTURE_H

#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>

#include "misc.h"
#include "types.h"

class CameraCapture {
  public:
    CameraCapture(int deviceID = 0);
    void capture_frames(SharedFrame &frame, bool &keepRunning);

  private:
    cv::VideoCapture cap;
};

#endif
