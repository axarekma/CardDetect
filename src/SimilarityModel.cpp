#include "SimilarityModel.h"
#include "misc.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp> // Use nlohmann JSON for parsing
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>

using json = nlohmann::json;

// Overload the << operator for std::ostream
std::ostream &operator<<(std::ostream &os, const CardEntry &card) {
    os << "CardEntry { "
       << "Name: " << card.name
       << ", "
       //    << "ID: " << card.id << ", "
       << "Set: " << card.set
       << ", "
       //    << "Image Path: " << card.image_path
       << " }";
    return os;
}

double cosine_similarity(const cv::Mat &A, const cv::Mat &B) {
    if (A.rows != B.rows || A.cols != B.cols || A.total() == 0) {
        throw std::invalid_argument("Vectors must have the same size and be non-empty.");
    }

    double dotProduct = A.dot(B);
    double normA = cv::norm(A, cv::NORM_L2);
    double normB = cv::norm(B, cv::NORM_L2);

    if (normA == 0.0 || normB == 0.0) {
        throw std::runtime_error("One of the vectors has zero magnitude.");
    }

    return dotProduct / (normA * normB);
}

void print_blob_shape(const cv::Mat &blob, const std::string &name = "Blob") {
    std::cout << name << " shape: [";
    for (int i = 0; i < blob.dims; i++) {
        std::cout << blob.size[i];
        if (i < blob.dims - 1) { std::cout << ", "; }
    }
    std::cout << "]" << std::endl;
}

void save_image(const cv::Mat &img, const std::string &name = "image.jpg") {
    cv::imwrite(name, img);
}
void save_float_image(const cv::Mat &img, const std::string &name = "image.jpg") {
    cv::Mat normalized_vis;
    img.convertTo(normalized_vis, CV_8U, 255);
    cv::imwrite(name, normalized_vis);
}

// Constructor: Load the ONNX model
SimilarityModel::SimilarityModel(const std::string &model_path) {
    net = cv::dnn::readNetFromONNX(model_path);
    if (net.empty()) {
        std::cerr << "Error loading the ONNX model!" << std::endl;
        exit(-1);
    }
}

// Load card database from JSON file
bool SimilarityModel::load_database(const std::string &json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        std::cerr << "Error opening JSON file: " << json_path << std::endl;
        return false;
    }

    json j;
    file >> j;

    for (const auto &item : j) {
        CardEntry card;
        card.name = item["name"];
        card.id = item["id"];
        card.set = item["set"];
        card.image_path = item["image"];
        database.push_back(card);
    }

    std::cout << "Loaded " << database.size() << " cards from database.\n";
    return true;
}

// Preprocess image for model input
cv::Mat SimilarityModel::preprocess_image(const cv::Mat &image, bool swapRB) {
    cv::Mat resized_image, float_image;
    // this is not strictly needed, as the databank is of equal size
    cv::resize(image, resized_image, input_size);

    if (swapRB) {
        cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB); // Swap RGB -> BGR
    }

    resized_image.convertTo(float_image, CV_32F, 1.0 / 255); // Normalize to [0,1]

    // saveImage(image, "input.jpg");
    // saveImage(resized_image, "resized_image.jpg");
    // saveFloatImage(float_image, "float_image.jpg");
    return float_image;
}

// Compute features for all images in the database
void SimilarityModel::calculate_features(int batch_size, int DB_MAX) {
    feature_vectors.clear();

    size_t total = database.size();
    if (DB_MAX > 0) { total = std::min(database.size(), static_cast<size_t>(DB_MAX)); }

    // Progress bar settings
    const int bar_width = 50;

    std::cout << "Computing features for " << total << " images...\n";

    for (size_t i = 0; i < total; i += batch_size) {
        std::vector<cv::Mat> batch_images;

        for (size_t j = i; j < i + batch_size && j < database.size(); ++j) {
            cv::Mat img = cv::imread(database[j].image_path);

            if (img.empty()) {
                std::cerr << "Error loading image: " << database[j].image_path << "\n";
                continue;
            }
            batch_images.push_back(preprocess_image(img, true));
        }

        if (batch_images.empty()) continue;

        cv::Mat batch_blob = cv::dnn::blobFromImages(batch_images);
        // printBlobShape(batch_blob, "batch_blob");

        // Perform batch inference
        net.setInput(batch_blob);
        cv::Mat batch_features = net.forward();

        // Store extracted features
        for (int j = 0; j < batch_features.rows; ++j) {
            feature_vectors.push_back(batch_features.row(j).clone());
        }

        // Update progress bar
        float progress = static_cast<float>(std::min(i + batch_size, total)) / total;
        int pos = static_cast<int>(bar_width * progress);

        std::cout << "[";
        for (int j = 0; j < bar_width; ++j) {
            if (j < pos)
                std::cout << "=";
            else if (j == pos)
                std::cout << ">";
            else
                std::cout << " ";
        }

        int percent = static_cast<int>(progress * 100.0);
        std::cout << "] " << percent << "% " << std::min(i + batch_size, total) << "/" << total
                  << "\r";
        std::cout.flush();
    }

    std::cout << "Feature extraction complete.\n";
}

void SimilarityModel::load_feature_vectors_binary(const std::string &filename) {
    feature_vectors.clear();

    std::ifstream infile(filename, std::ios::binary);
    while (infile) {
        int rows, cols;
        infile.read(reinterpret_cast<char *>(&rows), sizeof(int));
        infile.read(reinterpret_cast<char *>(&cols), sizeof(int));
        if (infile.eof()) break;
        cv::Mat feature(rows, cols, CV_32F);
        infile.read(reinterpret_cast<char *>(feature.data), rows * cols * sizeof(float));
        feature_vectors.push_back(feature);
    }
    infile.close();
}

void SimilarityModel::save_feature_vectors_binary(const std::string &filename) {

    std::ofstream outfile(filename, std::ios::binary);
    for (const auto &feature : feature_vectors) {
        int rows = feature.rows;
        int cols = feature.cols;
        outfile.write(reinterpret_cast<char *>(&rows), sizeof(int));
        outfile.write(reinterpret_cast<char *>(&cols), sizeof(int));
        outfile.write(reinterpret_cast<char *>(feature.data), rows * cols * sizeof(float));
    }
    outfile.close();
}

void SimilarityModel::find_top_k_indices(const std::vector<double> &similarities, int k) {
    if (k <= 0 || k > similarities.size()) { throw std::invalid_argument("Invalid k value."); }

    // clear containers
    topKIndices.clear();
    topKScores.clear();

    // Min-heap storing (similarity, index) pairs
    using Pair = std::pair<double, int>;
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> minHeap;

    for (int i = 0; i < similarities.size(); i++) {
        minHeap.emplace(similarities[i], i);
        if (minHeap.size() > k) {
            minHeap.pop(); // Remove smallest element
        }
    }

    // Extract top-k indices
    while (!minHeap.empty()) {
        topKScores.push_back(minHeap.top().first);
        topKIndices.push_back(minHeap.top().second);
        minHeap.pop();
    }

    // Reverse the indices to have the best match first
    std::reverse(topKIndices.begin(), topKIndices.end());
    std::reverse(topKScores.begin(), topKScores.end());
}

std::vector<int> SimilarityModel::find_similar(const cv::Mat &query_image, int k) {
    // Preprocess the image before passing it to the network
    cv::Mat flipped_query;
    cv::rotate(query_image, flipped_query, cv::ROTATE_180);
    cv::Mat input_blob = preprocess_image(query_image, true);
    cv::Mat input_blob_flip = preprocess_image(flipped_query, true);

    // printChannelMeans(query_image);
    // printChannelMeans(input_blob);

    // Forward pass through the network to get the feature vector
    net.setInput(cv::dnn::blobFromImage(input_blob));
    cv::Mat output1 = net.forward().clone();
    net.setInput(cv::dnn::blobFromImage(input_blob_flip));
    cv::Mat output2 = net.forward().clone();

    std::vector<double> similarities;
    std::vector<double> similarities_flip;
    for (int i = 0; i < feature_vectors.size(); ++i) {
        double cos_sim1 = cosine_similarity(output1, feature_vectors[i]);
        double cos_sim2 = cosine_similarity(output2, feature_vectors[i]);
        similarities_flip.push_back(cos_sim2);
        similarities.push_back(std::max(cos_sim1, cos_sim2));
    }

    find_top_k_indices(similarities, k);
    return topKIndices;
}

cv::Mat SimilarityModel::image_at(int index) {
    cv::Mat image;

    // Check if index is within bounds
    if (index < 0 || index >= database.size()) {
        std::cerr << "Error: Index " << index << " is out of range! \n";
        return cv::Mat(); // Return an empty image
    }

    // Check if file exists
    if (!std::filesystem::exists(database[index].image_path)) {
        image = cv::Mat::ones(IMAGE.H, IMAGE.W, CV_8UC3) * 128;
        cv::putText(image, "NOT FOUND", cv::Point(20, 100), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 0, 0), 3);
        cv::putText(image, "NOT FOUND", cv::Point(20, 100), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 1);
        return image;
    }

    // Load the image from file
    image = cv::imread(database[index].image_path);
    if (image.empty()) {
        image = cv::Mat::ones(IMAGE.H, IMAGE.W, CV_8UC3) * 128;
        cv::putText(image, "Load Failed", cv::Point(20, 100), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 0, 0), 3);
        cv::putText(image, "Load Failed", cv::Point(20, 100), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 1);
        return image;
    }

    return image; // Return the loaded image
}

void SimilarityModel::overlay_top(cv::Mat card, int ind) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3) << topKScores[ind];
    std::string line_score = "Score: " + ss.str();
    int index = topKIndices[ind];
    std::string line_info = database[index].name + " (" + database[index].set + ")";

    cv::putText(card, line_info, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 3);
    cv::putText(card, line_info, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1);

    cv::putText(card, line_score, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 3);
    cv::putText(card, line_score, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1);
}
