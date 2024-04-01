#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include "spdlog/spdlog.h"

#include "utils.hpp"
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
    std::string imagePath = "/doc/work/data/RIWO/data/calyx/images/20210914_105512764176_rgb_trigger003_apple1.png";
    cv::Mat image = cv::imread(imagePath);

    std::vector<std::string> classNames = {"calyx"};
 
    std::vector<Detection> results;
    
    if (inftype == "opencv"){
        spdlog::info("Using OpenCV");
        OPENCVInf inf = OPENCVInf(modelPath, isGPU, confThreshold, maskThreshold);
        while (true) {
            results = inf.predict(image);
            for (auto &result : results){
                spdlog::info("Detection: id: {}, accu: {}, bbox: ({}, {}, {}, {})", result.id, result.accu, result.bbox.x, result.bbox.y, result.bbox.width, result.bbox.height);
            }
        }
        
    } else if (inftype == "onnx"){
        spdlog::info("Using ONNX");
        ONNXInf inf = ONNXInf(modelPath, isGPU, confThreshold, maskThreshold);
        while (true) {
            results = inf.predict(image);
            for (auto &result : results){
                spdlog::info("Detection: id: {},  accu: {}, bbox: ({}, {}, {}, {})", result.id, result.accu, result.bbox.x, result.bbox.y, result.bbox.width, result.bbox.height);
            }
        }
    } else if (inftype == "torch"){
        spdlog::info("Using Torch");
        std::string modelPath = "/doc/work/data/RIWO/models/calix/best.torchscript";
        TORCHInf inf = TORCHInf(modelPath, isGPU, confThreshold, maskThreshold);
        inf.predict(image);
        for (auto &result : results){
            spdlog::info("Detection: id: {},  accu: {}, bbox: ({}, {}, {}, {})", result.id, result.accu, result.bbox.x, result.bbox.y, result.bbox.width, result.bbox.height);
        }
    }


    spdlog::info("Inference done");

    // utils::visualizeDetection(image, results, classNames);

    // cv::imshow("Result", image);
    // cv::waitKey(0);


    return 0;
}