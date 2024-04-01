#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include "spdlog/spdlog.h"

#include "detector.hpp"
#include "onnxinf.hpp"
#include "opencvinf.hpp"
#include "torchinf.hpp"


int main(int argc, char *argv[]){
    float confThreshold = 0.01f;
    float maskThreshold = 0.01f;
    bool isGPU = true;
    std::string inftype = "onnx";

    if (argc > 1){
        if (std::string(argv[1]) == "opencv"){
            inftype = "opencv";
        } else if (std::string(argv[1]) == "onnx"){
            inftype = "onnx";
        } else if (std::string(argv[1]) == "torch"){
            inftype = "torch";
        } else {
            spdlog::error("Invalid inference type");
            return 1;
        }
    } else {
        spdlog::error("Please provide inference type");
        return 1;
    }

    spdlog::info("Start inference");    

    std::string modelPath = "/doc/work/data/RIWO/models/calix/best.onnx";
    cv::Mat image = cv::imread("/doc/work/data/RIWO/data/calyx/images/20210914_105512764176_rgb_trigger003_apple1.png");

    
    Detector *detector;
    if (inftype == "opencv"){
        detector = new OPENCVInf(modelPath, isGPU, confThreshold, maskThreshold);
    } else if (inftype == "onnx"){
        detector = new ONNXInf(modelPath, isGPU, confThreshold, maskThreshold);
    } else if (inftype == "torch"){
        modelPath = "/doc/work/data/RIWO/models/calix/best.torchscript";
        detector = new TORCHInf(modelPath, isGPU, confThreshold, maskThreshold);
    }

    std::vector<Detection> results = detector->predict(image);
    spdlog::info("Inference done");

    std::vector<std::string> classNames = {"calyx"};
    utils::visualizeDetection(image, results, classNames);

    cv::imshow("Result", image);
    cv::waitKey(0);


    return 0;
}