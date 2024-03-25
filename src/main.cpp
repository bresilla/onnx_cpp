#include <opencv2/opencv.hpp>
#include "inference.hpp"
#include "spdlog/spdlog.h"


int main(int argc, char *argv[]){
    float confThreshold = 0.1f;
    float iouThreshold = 0.1f;
    float maskThreshold = 0.1f;
    bool isGPU = false;

    bool opencv = true;

    spdlog::info("Start inference");

    std::string modelPath = "/doc/work/data/RIWO/data/calyx/runs/segment/train/weights/best.onnx";
    std::string imagePath = "/doc/work/data/RIWO/data/calyx/images/20210914_105512764176_rgb_trigger003_apple1.png";
    cv::Mat image = cv::imread(imagePath);


 
    std::vector<Detection> result;
    if (opencv){
        spdlog::info("Using OpenCV inference");
        OPENCVInf inf = OPENCVInf(modelPath, cv::Size(640, 640), isGPU, confThreshold, maskThreshold, 32, cv::Size(160, 160));
        result = inf.segment(image);
    } else {
        spdlog::info("Using ONNX inference");
        ONNXInf inf = ONNXInf(modelPath, isGPU, confThreshold, iouThreshold, maskThreshold);
        result = inf.predict(image);
    }

    for (auto &det : result){
        spdlog::info("Id: {}, Accu: {}, Bbox: ({}, {}, {}, {})", det.id, det.accu, det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height);
    }

    spdlog::info("Inference done");

    // utils::visualizeDetection(image, result, classNames);


    return 0;
}