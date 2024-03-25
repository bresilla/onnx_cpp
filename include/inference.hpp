#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <utility>
#include <spdlog/spdlog.h>

struct Detection {
    int id{};
    float accu{};
    cv::Rect bbox;
    cv::Mat mask;
    std::vector<cv::Point> contour = {};
};

namespace utils{
    static std::vector<cv::Scalar> colors;
    void visualizeDetection(cv::Mat &image, std::vector<Detection> &results, const std::vector<std::string> &classNames);
    void letterbox(const cv::Mat &image, cv::Mat &outImage, const cv::Size &newShape, const cv::Scalar &color, bool auto_, bool scaleFill, bool scaleUp, int stride);
    void scaleCoords(cv::Rect &coords, cv::Mat &mask, const float maskThreshold, const cv::Size &imageShape, const cv::Size &imageOriginalShape);
    std::vector<cv::Point> get_contour(const cv::Mat& mask, bool join=true);
}

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

class OPENCVInf {
    public:
        OPENCVInf( const std::string &onnxModelPath, const bool &runWithCuda, const float &accuThresh, const float &maskThresh, const cv::Size &modelInputShape = cv::Size(640, 640));
        std::vector<Detection> predict(cv::Mat& srcImg, bool only_bbox = false);

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