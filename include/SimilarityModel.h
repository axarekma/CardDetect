#ifndef SIMILARITY_MODEL_H
#define SIMILARITY_MODEL_H

#include "types.h"
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

class SimilarityModel {
  public:
    SimilarityModel(const std::string &model_path);

    bool load_database(const std::string &json_path);
    bool features_exist() { return feature_vectors.size() > 0; };
    void calculate_features(int batch_size = 32, int DB_MAX = -1);

    void save_feature_vectors_binary(const std::string &filename);
    void load_feature_vectors_binary(const std::string &filename);

    void find_top_k_indices(const std::vector<double> &similarities, int k);

    // Get feature vector for a single image
    cv::Mat extract_feature(const cv::Mat &image);

    // Extract similarities of query against the databank
    std::vector<int> find_similar(const cv::Mat &query_image, int k);

    // Retrunt he image in the database
    cv::Mat image_at(int index);

    void overlay_top(cv::Mat card, int ind);

    std::vector<cv::Mat> feature_vectors; // Stored features
    std::vector<int> topKIndices;
    std::vector<double> topKScores;

    cv::dnn::Net net;                         // OpenCV DNN module network
    cv::Size input_size = cv::Size(146, 204); // Input size expected by the model W,H
    std::vector<CardEntry> database;          // List of cards with image paths

    // Preprocess image to match model input format
    cv::Mat preprocess_image(const cv::Mat &image, bool swapRB);
};

#endif // SIMILARITY_MODEL_H
