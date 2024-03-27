#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <utility>
#include <spdlog/spdlog.h>
#include "detections.hpp"


class OPENCVInf {
    public:
        OPENCVInf( const std::string &onnxModelPath, const bool &runWithCuda, const float &accuThresh, const float &maskThresh, const cv::Size &modelInputShape = cv::Size(640, 640));
        std::vector<Detection> predict(cv::Mat& srcImg, bool only_bbox = true);

    private:
        std::vector<Detection> decode_output(cv::Mat& output0, cv::Mat& output1);
        cv::Mat get_mask_rel(const cv::Mat& mask_info, const cv::Mat& mask_data, cv::Rect box);
        cv::Mat get_mask_abs(const cv::Mat& mask_info, const cv::Mat& mask_data, cv::Rect box);
        std::vector<cv::Point> get_contour(const cv::Mat& mask, bool join = true);

        std::string model_path;
        cv::Size model_shape;
        cv::Size seg_size;
        bool cuda_enabled;

        cv::dnn::Net net;

        int seg_ch = 32;
        float accu_thresh = 0.2;
        float mask_thresh = 0.2;
        cv::Size origin_size;
        cv::Size2f scaled_size;

        bool only_bbox;
    
    public:
        std::vector<std::string> class_names = { "apple" };
};