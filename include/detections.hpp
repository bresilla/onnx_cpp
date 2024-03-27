#pragma once

#include <utility>
#include <opencv2/opencv.hpp>
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
