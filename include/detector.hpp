#pragma once
#include <utility>
#include <opencv2/opencv.hpp>

struct Detection {
    int id{};
    float accu{};
    cv::Rect bbox;
    cv::Mat mask;
    std::vector<cv::Point> contour = {};
};

class Detector {
    public:
        virtual std::vector<Detection> predict(cv::Mat &image, bool only_boxes=false) = 0;
};

namespace utils{
    static std::vector<cv::Scalar> colors;
    void visualizeDetection(cv::Mat &image, std::vector<Detection> &results, const std::vector<std::string> &classNames);
    void letterbox(const cv::Mat &image, cv::Mat &outImage, const cv::Size &newShape, const cv::Scalar &color, bool auto_, bool scaleFill, bool scaleUp, int stride);
    void scaleCoords(cv::Rect &coords, cv::Mat &mask, const float maskThreshold, const cv::Size &imageShape, const cv::Size &imageOriginalShape);
    std::vector<cv::Point> get_contour(const cv::Mat& mask, bool join=true);
}
