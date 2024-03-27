#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <utility>
#include <spdlog/spdlog.h>
#include "detections.hpp"


class ONNXInf {
    public:
        explicit ONNXInf(std::nullptr_t){};
        ONNXInf(const std::string &modelPath, const bool &isGPU, float confThreshold, float maskThreshold, float iouThreshold=0.1f);
        std::vector<Detection> predict(cv::Mat &image);
        int classNums = 80;

    private:
        Ort::Env env{nullptr};
        Ort::SessionOptions sessionOptions{nullptr};
        Ort::Session session{nullptr};

        void preprocessing(cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape);
        std::vector<Detection> postprocessing(const cv::Size &resizedImageShape, const cv::Size &originalImageShape, std::vector<Ort::Value> &outputTensors);
        static void getBestClassInfo(std::vector<float>::iterator it, float &bestConf, int &bestClassId, const int _classNums);
        cv::Mat getMask(const cv::Mat &maskProposals, const cv::Mat &maskProtos);
        bool isDynamicInputShape{};

        std::vector<const char *> inputNames;
        std::vector<Ort::AllocatedStringPtr> input_names_ptr;

        std::vector<const char *> outputNames;
        std::vector<Ort::AllocatedStringPtr> output_names_ptr;

        std::vector<std::vector<int64_t>> inputShapes;
        std::vector<std::vector<int64_t>> outputShapes;
        float confThreshold = 0.3f;
        float iouThreshold = 0.4f;

        bool hasMask = false;
        float maskThreshold = 0.5f;
};