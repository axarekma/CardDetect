#include "CameraCapture.h"

CameraCapture::CameraCapture(int deviceID) { cap.open(deviceID); }

void CameraCapture::capture_frames(SharedFrame &shared_frame, bool &keepRunning) {
    cv::Mat frame;
    while (keepRunning) {
        cap >> frame;
        if (frame.empty()) continue;

        {
            std::lock_guard<std::mutex> lock(shared_frame.mutex);
            shared_frame.frame = frame;
            shared_frame.fps.update();
        }
    }
}
