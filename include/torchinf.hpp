#pragma once
#include "utils.hpp"
#include <utility>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>
#include <torch/script.h>

using torch::indexing::Slice;
using torch::indexing::None;



class TORCHInf {
    public:
        TORCHInf(const std::string &modelPath, const bool &isGPU, float confThreshold, float maskThreshold, float iouThreshold=0.1f);
        std::vector<Detection> predict(cv::Mat &image);
    private:
        torch::Device device = torch::Device(torch::kCPU);
        std::string model_path;

        float generate_scale(cv::Mat& image, const std::vector<int>& target_size);
        torch::Tensor xyxy2xywh(const torch::Tensor& x);
        torch::Tensor xywh2xyxy(const torch::Tensor& x);
        float letterbox(cv::Mat &input_image, cv::Mat &output_image, const std::vector<int> &target_size);
        torch::Tensor nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold);
        torch::Tensor non_max_suppression(torch::Tensor& prediction, float conf_thres = 0.25, float iou_thres = 0.45, int max_det = 300);
        torch::Tensor clip_boxes(torch::Tensor& boxes, const std::vector<int>& shape);
        torch::Tensor scale_boxes(const std::vector<int>& img1_shape, torch::Tensor& boxes, const std::vector<int>& img0_shape);
};