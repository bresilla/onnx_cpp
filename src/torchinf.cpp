#include "torchinf.hpp"

TORCHInf::TORCHInf(const std::string &modelPath, const bool &isGPU, float confThreshold, float maskThreshold, float iouThreshold) {
    if (isGPU && torch::cuda::is_available()){
        this->device = torch::Device(torch::kCUDA);
    }
    this->model_path = modelPath;
    this->conf_threshold = confThreshold;
    this->mask_threshold = maskThreshold;
    this->nms_threshold = confThreshold;
}

std::vector<Detection> TORCHInf::predict(cv::Mat &image, bool only_bbox) {
    std::vector<Detection> results;
    try {
        torch::jit::script::Module yolo_model;
        yolo_model = torch::jit::load(model_path);
        yolo_model.eval();
        yolo_model.to(this->device, torch::kFloat32);

        cv::Mat input_image;
        letterbox(image, input_image, {640, 640});

        torch::Tensor image_tensor = torch::from_blob(input_image.data, {input_image.rows, input_image.cols, 3}, torch::kByte).to(this->device);
        image_tensor = image_tensor.toType(torch::kFloat32).div(255);
        image_tensor = image_tensor.permute({2, 0, 1});
        image_tensor = image_tensor.unsqueeze(0);

        std::vector<torch::jit::IValue> inputs{ image_tensor };
        auto net_outputs = yolo_model.forward(inputs).toTuple();

        at::Tensor main_output = net_outputs->elements()[0].toTensor().to(this->device);
        at::Tensor mask_output = net_outputs->elements()[1].toTensor().to(this->device);

        cv::Mat detect_buffer = cv::Mat(main_output.sizes()[1], main_output.sizes()[2], CV_32F, (float*)main_output.data_ptr()).t();

        cv::Mat segment_buffer(32, 25600, CV_32F);
        std::memcpy((void*)segment_buffer.data, mask_output.data_ptr(), sizeof(float) * 32 * 160 * 160);


        std::vector<cv::Rect> mask_boxes;
        std::vector<cv::Rect> boxes;
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Mat> masks;
        for (int i = 0; i < detect_buffer.rows; ++i) {
            const cv::Mat result = detect_buffer.row(i);
            const cv::Mat classes_scores = result.colRange(4, main_output.sizes()[1] - 32);
            cv::Point class_id_point;
            double score;
            cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);
            if (score > this->conf_threshold) {
                class_ids.push_back(class_id_point.x);
                confidences.push_back(score);
                const float mask_scale = 0.25f; // 160/640 = 0.25
                const cv::Mat detection_box = result.colRange(0, 4);
                const cv::Rect mask_box = toBox(detection_box * mask_scale, cv::Rect(0, 0, 160, 160));
                const cv::Rect image_box = toBox(detection_box, cv::Rect(0, 0, image.cols, image.rows));
                mask_boxes.push_back(mask_box);
                boxes.push_back(image_box);
                masks.push_back(result.colRange(main_output.sizes()[1] - 32, main_output.sizes()[1]));
            }
        }

        std::vector<int> nms_indexes;
        cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_indexes);

        Detection result;
        for (const int index : nms_indexes) {
            result.bbox = boxes[index];
            result.accu = confidences[index];
            result.id = class_ids[index];
            result.mask = masks[index];
            result.contour = utils::get_contour(result.mask);
            results.push_back(result);
        }

    } catch (const c10::Error& e) {
        std::cout << e.msg() << std::endl;
    }
    return results;
}

float TORCHInf::generate_scale(cv::Mat& image, const std::vector<int>& target_size) {
    int origin_w = image.cols;
    int origin_h = image.rows;

    int target_h = target_size[0];
    int target_w = target_size[1];

    float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
    float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
    float resize_scale = std::min(ratio_h, ratio_w);
    return resize_scale;
}


float TORCHInf::letterbox(cv::Mat &input_image, cv::Mat &output_image, const std::vector<int> &target_size) {
    if (input_image.cols == target_size[1] && input_image.rows == target_size[0]) {
        if (input_image.data == output_image.data) {
            return 1.;
        } else {
            output_image = input_image.clone();
            return 1.;
        }
    }

    float resize_scale = generate_scale(input_image, target_size);
    int new_shape_w = std::round(input_image.cols * resize_scale);
    int new_shape_h = std::round(input_image.rows * resize_scale);
    float padw = (target_size[1] - new_shape_w) / 2.;
    float padh = (target_size[0] - new_shape_h) / 2.;

    int top = std::round(padh - 0.1);
    int bottom = std::round(padh + 0.1);
    int left = std::round(padw - 0.1);
    int right = std::round(padw + 0.1);

    cv::resize(input_image, output_image,
               cv::Size(new_shape_w, new_shape_h),
               0, 0, cv::INTER_AREA);

    cv::copyMakeBorder(output_image, output_image, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114.));
    return resize_scale;
}

cv::Rect TORCHInf::toBox(const cv::Mat& input, const cv::Rect& range) {
    const float cx = input.at<float>(0);
    const float cy = input.at<float>(1);
    const float ow = input.at<float>(2);
    const float oh = input.at<float>(3);
    cv::Rect box;
    box.x = cvRound(cx - 0.5f * ow);
    box.y = cvRound(cy - 0.5f * oh);
    box.width = cvRound(ow);
    box.height = cvRound(oh);
    return box & range;
}