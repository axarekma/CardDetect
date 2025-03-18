#include "DisplayManager.h"

void DisplayManager::display_frames(SharedFrame &frame_raw, SharedFrame &frame_proc,
                                    SharedCards &cards, bool &keepRunning, int maxWidth) {

    if (!windowsInitialized) {
        cv::namedWindow("Debug View", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Hover Window", cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback("Debug View", mouseCallbackWrapper, this);
        windowsInitialized = true;

        // Init the hover window on top of the main view]
        cv::Rect debugWindowRect = cv::getWindowImageRect("Debug View");
        const int additional_pad = 80;
        int hoverWindowX = debugWindowRect.x;
        int hoverWindowY = debugWindowRect.y - IMAGE.H - additional_pad;
        cv::moveWindow("Hover Window", hoverWindowX, hoverWindowY);
    }

    while (keepRunning) {
        // Skip if frames aren't ready
        if (frame_raw.frame.empty() || frame_proc.frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        // set raw image
        {
            std::lock_guard<std::mutex> lock(frame_raw.mutex);
            double scale =
                std::min(1.0, static_cast<double>(maxWidth) / (2 * frame_raw.frame.cols));
            cv::resize(frame_raw.frame, leftImage, cv::Size(), scale, scale);
            frame_raw.fps.overlay_fps(leftImage);
        }

        // set processed image
        {
            std::lock_guard<std::mutex> lock(frame_proc.mutex);
            double scale =
                std::min(1.0, static_cast<double>(maxWidth) / (2 * frame_proc.frame.cols));
            cv::resize(frame_proc.frame, rightImage, cv::Size(), scale, scale);
            if (rightImage.channels() == 1) {
                cv::cvtColor(rightImage, rightImage, cv::COLOR_GRAY2BGR);
            }
            frame_proc.fps.overlay_fps(rightImage);
        }

        // copy cards
        {
            std::lock_guard<std::mutex> lock(cards.mutex);
            local_cards = cards.cards;
        }

        leftWidth = leftImage.cols;
        cv::hconcat(std::vector<cv::Mat>{leftImage, rightImage}, combinedView);
        cv::imshow("Debug View", combinedView);
        if (cv::waitKey(1) == 27) // Exit on 'ESC' key
        {
            keepRunning = false;
        }
    }
}

// Actual mouse handler
void DisplayManager::onMouse(int event, int x, int y, int flags) {
    (void)flags; // not using eventflags here

    if (event == cv::EVENT_MOUSEMOVE) {

        constexpr int W = IMAGE.W;
        constexpr int H = IMAGE.H;
        int concat_W = 4 * W;
        cv::Mat hoverImg = cv::Mat::ones(H, concat_W, CV_8UC3) * 255;

        bool isLeftImage = (x < leftWidth);
        std::string side = isLeftImage ? "Raw Image (Left)" : "Processed Image (Right)";

        int localX = isLeftImage ? x : x - leftWidth;

        for (auto card : local_cards) {
            int original_width = local_cards[0].frame.cols;
            float hover_scale = static_cast<float>(1.0 * original_width / leftWidth);
            cv::Point2f hover_point(hover_scale * localX, hover_scale * y);
            if (card.is_inside(hover_point)) {
                card.unwarp();
                model->find_similar(card.unwarped_image, 5);
                cv::Mat top1 = model->image_at(model->topKIndices[0]);
                cv::Mat top2 = model->image_at(model->topKIndices[1]);
                cv::Mat top3 = model->image_at(model->topKIndices[2]);

                model->overlay_top(top1, 0);
                model->overlay_top(top2, 1);
                model->overlay_top(top3, 2);

                std::vector<cv::Mat> images_to_concatenate = {card.unwarped_image, top1, top2,
                                                              top3};
                cv::hconcat(images_to_concatenate, hoverImg);
            }
        }
        cv::imshow("Hover Window", hoverImg);
    }
}
