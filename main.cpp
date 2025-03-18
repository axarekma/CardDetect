#include <mutex>
#include <queue>
#include <thread>

#include "CameraCapture.h"
#include "DisplayManager.h"
#include "FrameProcessor.h"
#include "SimilarityModel.h"
#include "misc.h"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
const std::string data_dir = TOSTRING(DATA_DIR);

SharedFrame frame_raw;
SharedFrame frame_edit;
SharedCards cards;
bool keepRunning = true;

const std::string model_path = data_dir + "/model_res.onnx";
const std::string database_path = data_dir + "/card_database.json";
const std::string precalc_features_path = data_dir + "/precalc_features.bin";
const std::string test_frame = data_dir + "/frame_0.png";

SimilarityModel model(model_path);

int const WIDTH = 1000;

void printInfo(const std::string &description, const std::string &path) {
    const int width = 30;
    std::cout << std::left << std::setw(width) << description << ": " << path << "\n";
}

void run_camera() {
    CameraCapture camera;
    FrameProcessor processor;
    DisplayManager display;
    display.set_model(model);

    std::thread cameraThread([&]() { camera.capture_frames(frame_raw, keepRunning); });
    std::thread processingThread(
        [&]() { processor.process_frames(frame_raw, frame_edit, cards, keepRunning); });
    std::thread displayThread(
        [&]() { display.display_frames(frame_raw, frame_edit, cards, keepRunning, WIDTH); });

    cameraThread.join();
    processingThread.join();
    displayThread.join();
}
void run_static() {
    cv::Mat colorImage = cv::imread(test_frame);
    {
        std::lock_guard<std::mutex> lock(frame_raw.mutex);
        cv::cvtColor(colorImage, frame_raw.frame, cv::COLOR_RGB2BGR);
    }

    FrameProcessor processor;
    DisplayManager display;
    display.set_model(model);

    std::thread processingThread(
        [&]() { processor.process_frames(frame_raw, frame_edit, cards, keepRunning); });
    std::thread displayThread(
        [&]() { display.display_frames(frame_raw, frame_edit, cards, keepRunning, WIDTH); });

    processingThread.join();
    displayThread.join();
}

int main() {
    std::cout << "Loading files...\n";
    printInfo("Base dir", data_dir);
    printInfo("Model Path", model_path);
    printInfo("Database Path", database_path);
    printInfo("Precomputed Features Path", precalc_features_path);
    std::cout << "\n";

    if (!model.load_database(database_path)) {
        std::cerr << "Error loading database \n";
        return 0;
    }

    model.load_feature_vectors_binary(precalc_features_path);
    if (!model.features_exist()) {
        std::cout << "No features loaded\n";
        std::cout << "PreCalculating features ... \n";
        model.calculate_features(64, -1);
        std::cout << "done \n";
        model.save_feature_vectors_binary(precalc_features_path);
    }

    run_camera();
    // run_static();

    return 0;
}
