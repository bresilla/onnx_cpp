#pragma once
#include "utils.hpp"
#include <utility>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

using torch::indexing::Slice;
using torch::indexing::None;



class TORCHInf {
    public:
        TORCHInf(const std::string &modelPath, const bool &isGPU, float confThreshold, float maskThreshold, float iouThreshold=0.1f);
        std::vector<Detection> predict(cv::Mat &image);
    private:
        float generate_scale(cv::Mat& image, const std::vector<int>& target_size);
        float letterbox(cv::Mat &input_image, cv::Mat &output_image, const std::vector<int> &target_size);
        cv::Rect toBox(const cv::Mat& input, const cv::Rect& range);

        torch::Device device = torch::Device(torch::kCPU);
        std::string model_path;
        float conf_threshold;
        float mask_threshold;
        float nms_threshold;
};