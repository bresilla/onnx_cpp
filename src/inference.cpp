#include "inference.hpp"

namespace utils {
	std::vector<cv::Point> get_contour(const cv::Mat& mask, bool join){
		cv::Mat mask2 = mask.clone();
		cv::Mat mask8;
		mask2.convertTo(mask8, CV_8UC1, 255);
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(mask8, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		if (join){
			std::vector<cv::Point> contour;
			for (int i = 0; i < contours.size(); i++){
				contour.insert(contour.end(), contours[i].begin(), contours[i].end());
			}
			return contour;
		}
		else{
			return contours[0];
		}
	}

	void visualizeDetection(cv::Mat &im, std::vector<Detection> &results, const std::vector<std::string> &classNames) {
		cv::Mat image = im.clone();
		for (const Detection &result : results) {
			int x = result.bbox.x;
			int y = result.bbox.y;
			int conf = (int)std::round(result.accu * 100);
			int classId = result.id;
			std::string label = classNames[classId] + " 0." + std::to_string(conf);
			int baseline = 0;
			cv::Size size = cv::getTextSize(label, cv::FONT_ITALIC, 0.4, 1, &baseline);
			image(result.bbox).setTo(colors[classId + classNames.size()], result.mask);
			cv::rectangle(image, result.bbox, colors[classId], 2);
			cv::rectangle(image, cv::Point(x, y), cv::Point(x + size.width, y + 12), colors[classId], -1);
			cv::putText(image, label, cv::Point(x, y - 3 + 12), cv::FONT_ITALIC, 0.4, cv::Scalar(0, 0, 0), 1);
		}
		cv::addWeighted(im, 0.4, image, 0.6, 0, im);
	}

    void draw_result(cv::Mat &img, std::vector<Detection>& detection, std::vector<cv::Scalar> color){
        cv::Mat mask = img.clone();
        for (int i = 0; i < detection.size(); i++){
            int left = detection[i].bbox.x;
            int top = detection[i].bbox.y;
            
            cv::rectangle(img, detection[i].bbox, color[detection[i].id], 2 );
            std::string classString = std::to_string(detection[i].id) + ' ' + std::to_string(detection[i].accu).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(left, top - 40, textSize.width + 10, textSize.height + 20);
            cv::rectangle(img, textBox, color[detection[i].id], cv::FILLED);
            cv::putText(img, classString, cv::Point(left + 5, top - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
            if (detection[i].mask.size() == cv::Size(0, 0)){ continue; }
            if (detection[i].mask.rows && detection[i].mask.cols > 0){ mask.setTo(color[detection[i].id], detection[i].mask); }
            cv::drawContours(img, std::vector<std::vector<cv::Point>>{detection[i].contour}, -1, color[detection[i].id], 2);
        }
        cv::addWeighted(img, 0.6, mask, 0.4, 0, img);
    }

	void letterbox(const cv::Mat &image, cv::Mat &outImage, const cv::Size &newShape = cv::Size(640, 640), 
						const cv::Scalar &color = cv::Scalar(114, 114, 114), bool auto_ = true, bool scaleFill = false, bool scaleUp = true, int stride = 32) {
		cv::Size shape = image.size();
		float r = std::min((float)newShape.height / (float)shape.height, (float)newShape.width / (float)shape.width);
		if (!scaleUp) { r = std::min(r, 1.0f); }

		float ratio[2]{r, r};
		int newUnpad[2]{(int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r)};

		auto dw = (float)(newShape.width - newUnpad[0]);
		auto dh = (float)(newShape.height - newUnpad[1]);

		if (auto_) {
			dw = (float)((int)dw % stride);
			dh = (float)((int)dh % stride);
		} else if (scaleFill) {
			dw = 0.0f;
			dh = 0.0f;
			newUnpad[0] = newShape.width;
			newUnpad[1] = newShape.height;
			ratio[0] = (float)newShape.width / (float)shape.width;
			ratio[1] = (float)newShape.height / (float)shape.height;
		}

		dw /= 2.0f;
		dh /= 2.0f;

		if (shape.width != newUnpad[0] && shape.height != newUnpad[1]) {
			cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
		}

		int top = int(std::round(dh - 0.1f));
		int bottom = int(std::round(dh + 0.1f));
		int left = int(std::round(dw - 0.1f));
		int right = int(std::round(dw + 0.1f));
		cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
	}

	void scaleCoords(cv::Rect &coords, cv::Mat &mask, const float maskThreshold, const cv::Size &imageShape, const cv::Size &imageOriginalShape) {
		float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height, (float)imageShape.width / (float)imageOriginalShape.width);

		int pad[2] = {(int)(((float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f), (int)(((float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f)};

		coords.x = (int)std::round(((float)(coords.x - pad[0]) / gain));
		coords.x = std::max(0, coords.x);
		coords.y = (int)std::round(((float)(coords.y - pad[1]) / gain));
		coords.y = std::max(0, coords.y);

		coords.width = (int)std::round(((float)coords.width / gain));
		coords.width = std::min(coords.width, imageOriginalShape.width - coords.x);
		coords.height = (int)std::round(((float)coords.height / gain));
		coords.height = std::min(coords.height, imageOriginalShape.height - coords.y);
		mask = mask(cv::Rect(pad[0], pad[1], imageShape.width - 2 * pad[0], imageShape.height - 2 * pad[1]));

		cv::resize(mask, mask, imageOriginalShape, cv::INTER_LINEAR);
		mask = mask(coords) > maskThreshold;
	}
}



ONNXInf::ONNXInf(const std::string &modelPath, const bool &isGPU, float confThreshold, float iouThreshold, float maskThreshold) {
    this->confThreshold = confThreshold;
    this->iouThreshold = iouThreshold;
    this->maskThreshold = maskThreshold;
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "YOLOV8");
    sessionOptions = Ort::SessionOptions();

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;
    if (isGPU && (cudaAvailable != availableProviders.end())){ 
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption); 
    }
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);

    const size_t num_input_nodes = session.GetInputCount();   //==1
    const size_t num_output_nodes = session.GetOutputCount(); //==1,2
    if (num_output_nodes > 1) { this->hasMask = true; }

    Ort::AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < num_input_nodes; i++) {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        this->inputNames.push_back(input_name.get());
        input_names_ptr.push_back(std::move(input_name));

        Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(i);
        std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        this->inputShapes.push_back(inputTensorShape);
        this->isDynamicInputShape = false;
        // checking if width and height are dynamic
        if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1) {
            std::cout << "Dynamic input shape" << std::endl;
            this->isDynamicInputShape = true;
        }
    }
    for (int i = 0; i < num_output_nodes; i++) {
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        this->outputNames.push_back(output_name.get());
        output_names_ptr.push_back(std::move(output_name));

        Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(i);
        std::vector<int64_t> outputTensorShape = outputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        this->outputShapes.push_back(outputTensorShape);
        if (i == 0) {
            if (!this->hasMask)
                classNums = outputTensorShape[1] - 4;
            else
                classNums = outputTensorShape[1] - 4 - 32;
        }
    }
}

void ONNXInf::getBestClassInfo(std::vector<float>::iterator it, float &bestConf, int &bestClassId, const int _classNums){
    // first 4 element are box
    bestClassId = 4;
    bestConf = 0;
    for (int i = 4; i < _classNums + 4; i++) {
        if (it[i] > bestConf) {
            bestConf = it[i];
            bestClassId = i - 4;
        }
    }
}

cv::Mat ONNXInf::getMask(const cv::Mat &maskProposals, const cv::Mat &maskProtos){
    cv::Mat protos = maskProtos.reshape(0, {(int)this->outputShapes[1][1], (int)this->outputShapes[1][2] * (int)this->outputShapes[1][3]});
    cv::Mat matmul_res = (maskProposals * protos).t();
    cv::Mat masks = matmul_res.reshape(1, {(int)this->outputShapes[1][2], (int)this->outputShapes[1][3]});
    cv::Mat dest;
    // sigmoid
    cv::exp(-masks, dest);
    dest = 1.0 / (1.0 + dest);
    cv::resize(dest, dest, cv::Size((int)this->inputShapes[0][2], (int)this->inputShapes[0][3]), cv::INTER_LINEAR);
    return dest;
}

void ONNXInf::preprocessing(cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape){
    cv::Mat resizedImage, floatImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    utils::letterbox(resizedImage, resizedImage, cv::Size((int)this->inputShapes[0][2], (int)this->inputShapes[0][3]), cv::Scalar(114, 114, 114), this->isDynamicInputShape, false, true, 32);

    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize{floatImage.cols, floatImage.rows};
    // hwc -> chw
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i) {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

std::vector<Detection> ONNXInf::postprocessing(const cv::Size &resizedImageShape, const cv::Size &originalImageShape, std::vector<Ort::Value> &outputTensors){
    // for box
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    float *boxOutput = outputTensors[0].GetTensorMutableData<float>();
    //[1,4+n,8400]=>[1,8400,4+n] or [1,4+n+32,8400]=>[1,8400,4+n+32]
    cv::Mat output0 = cv::Mat(cv::Size((int)this->outputShapes[0][2], (int)this->outputShapes[0][1]), CV_32F, boxOutput).t();
    float *output0ptr = (float *)output0.data;
    int rows = (int)this->outputShapes[0][2];
    int cols = (int)this->outputShapes[0][1];
    // std::cout << rows << cols << std::endl;
    // if hasMask
    std::vector<std::vector<float>> picked_proposals;
    cv::Mat mask_protos;

    for (int i = 0; i < rows; i++){
        std::vector<float> it(output0ptr + i * cols, output0ptr + (i + 1) * cols);
        float confidence;
        int classId;
        this->getBestClassInfo(it.begin(), confidence, classId, classNums);

        if (confidence > this->confThreshold){
            if (this->hasMask){
                std::vector<float> temp(it.begin() + 4 + classNums, it.end());
                picked_proposals.push_back(temp);
            }
            int centerX = (int)(it[0]);
            int centerY = (int)(it[1]);
            int width = (int)(it[2]);
            int height = (int)(it[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;
            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, this->confThreshold, this->iouThreshold, indices);

    if (this->hasMask) {
        float *maskOutput = outputTensors[1].GetTensorMutableData<float>();
        std::vector<int> mask_protos_shape = {1, (int)this->outputShapes[1][1], (int)this->outputShapes[1][2], (int)this->outputShapes[1][3]};
        mask_protos = cv::Mat(mask_protos_shape, CV_32F, maskOutput);
    }

    std::vector<Detection> results;
    for (int idx : indices) {
        Detection res;
        res.bbox = cv::Rect(boxes[idx]);
        if (this->hasMask)
            res.mask = this->getMask(cv::Mat(picked_proposals[idx]).t(), mask_protos);
        else
            res.mask = cv::Mat::zeros((int)this->inputShapes[0][2], (int)this->inputShapes[0][3], CV_8U);

		res.contour = utils::get_contour(res.mask, false);

        utils::scaleCoords(res.bbox, res.mask, this->maskThreshold, resizedImageShape, originalImageShape);
        res.accu = confs[idx];
        res.id = classIds[idx];
        results.emplace_back(res);
    }
    return results;
}

std::vector<Detection> ONNXInf::predict(cv::Mat &image) {
    float *blob = nullptr;
    std::vector<int64_t> inputTensorShape{1, 3, -1, -1};
    this->preprocessing(image, blob, inputTensorShape);
    
	if (inputTensorShape.empty()) { return {}; }
	size_t inputTensorSize = 1;
	for (const auto &element : inputTensorShape) {
		inputTensorSize *= element;
	}

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);
    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputTensorShape.data(), inputTensorShape.size()));
    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr}, this->inputNames.data(), inputTensors.data(), 1, this->outputNames.data(), this->outputNames.size());
    cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);
    std::vector<Detection> result = this->postprocessing(resizedShape, image.size(), outputTensors);
    delete[] blob;
    return result;
}

OPENCVInf::OPENCVInf( const std::string &onnxModelPath,  const cv::Size &modelInputShape,  const bool &runWithCuda, const float &accuThresh, const float &maskThresh, const int &segCh, const cv::Size &segSize){
    model_path = onnxModelPath;
    model_shape = modelInputShape;
    cuda_enabled = runWithCuda;
    seg_size = segSize;
    
    load_network();
}


void OPENCVInf::load_network(){
    net = cv::dnn::readNetFromONNX(model_path);
    if (cuda_enabled){
        std::cout << "\nRunning on CUDA" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else {
        std::cout << "\nRunning on CPU" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}


std::vector<Detection> OPENCVInf::detect(cv::Mat& srcImg){
	only_bbox = true;
	std::vector<Detection> result = run_detection(srcImg);
	return result;

}


std::vector<Detection> OPENCVInf::segment(cv::Mat& srcImg){
	only_bbox = false;
	std::vector<Detection> result = run_detection(srcImg);
	return result;
}


std::vector<Detection> OPENCVInf::run_detection(cv::Mat& srcImg){
	std::vector<Detection> result;
	cv::dnn::Net net = cv::dnn::readNet(model_path.c_str()); //default DNN_TARGET_CPU
	origin_size = srcImg.size();
    scaled_size = { 640.0 / srcImg.cols, 640.0 / srcImg.rows };
	cv::Mat image;
	resize(srcImg, image, cv::Size(640, 640));
	cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
	net.setInput(blob);
	std::vector<cv::Mat> outputs;
	net.forward(outputs, net.getUnconnectedOutLayersNames());
	result = decode_output(outputs[0], outputs[1]);
	return result;
}


std::vector<Detection> OPENCVInf::decode_output(cv::Mat& output0, cv::Mat& output1) {
	std::vector<Detection> results;
	cv::Mat out1 = cv::Mat(cv::Size(output0.size[2], output0.size[1]), CV_32F, (float*)output0.data).t();
	std::vector<int> class_ids;
	std::vector<float> accus;
	std::vector<cv::Rect> boxes;
	std::vector<std::vector<float>> masks;
	int data_width = class_names.size() + 4 + 32;
	int rows = out1.rows;
	float* pdata = (float*)out1.data;

	for (int r = 0; r < rows; ++r){
		cv::Mat scores(1, class_names.size(), CV_32FC1, pdata + 4);
		cv::Point class_id;
		double max_socre;
		minMaxLoc(scores, 0, &max_socre, 0, &class_id);
		if (max_socre >= accu_thresh){
			masks.push_back(std::vector<float>(pdata + 4 + class_names.size(), pdata + data_width));
			float w = pdata[2] / scaled_size.width;
			float h = pdata[3] / scaled_size.height;
			int left = MAX(int((pdata[0] - 0) / scaled_size.width - 0.5 * w + 0.5), 0);
			int top = MAX(int((pdata[1] - 0) / scaled_size.height - 0.5 * h + 0.5), 0);
			class_ids.push_back(class_id.x);
			accus.push_back(max_socre);
			boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
		}
		pdata += data_width;
	}

	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, accus, accu_thresh, mask_thresh, nms_result);

	for (int i = 0; i < nms_result.size(); ++i){
		int idx = nms_result[i];
		boxes[idx] = boxes[idx] & cv::Rect(0, 0, origin_size.width, origin_size.height);
		Detection result = { class_ids[idx], accus[idx], boxes[idx] };
		if (only_bbox){
			results.push_back(result);
			continue;
		}
		result.mask  = get_mask_abs(cv::Mat(masks[idx]).t(), output1, boxes[idx]);
		result.contour = get_contour(result.mask);
		results.push_back(result);
	}
	return results;
}


cv::Mat OPENCVInf::get_mask_rel(const cv::Mat& mask_info, const cv::Mat& mask_data, cv::Rect box) {
	int r_x = floor((box.x * scaled_size.width + 0) / model_shape.width * seg_size.width);
	int r_y = floor((box.y * scaled_size.height + 0) / model_shape.height * seg_size.height);
	int r_w = ceil(((box.x + box.width) * scaled_size.width + 0) / model_shape.width * seg_size.width) - r_x;
	int r_h = ceil(((box.y + box.height) * scaled_size.height + 0) / model_shape.height * seg_size.height) - r_y;
	r_w = MAX(r_w, 1);
	r_h = MAX(r_h, 1);
	if (r_x + r_w > seg_size.width) { seg_size.width - r_x > 0 ? r_w = seg_size.width - r_x : r_x -= 1; }
	if (r_y + r_h > seg_size.height) { seg_size.height - r_y > 0 ? r_h = seg_size.height - r_y : r_y -= 1; }
	std::vector<cv::Range> roi_rangs = {cv::Range(0, 1), cv::Range::all(), cv::Range(r_y, r_h + r_y), cv::Range(r_x, r_w + r_x)};
	cv::Mat temp_mask = mask_data(roi_rangs).clone();
	cv::Mat protos = temp_mask.reshape(0, { seg_ch,r_w * r_h });
	cv::Mat matmul_res = (mask_info * protos).t();
	cv::Mat masks_feature = matmul_res.reshape(1, { r_h,r_w });
	cv::Mat dest;
	cv::exp(-masks_feature, dest); //sigmoid
	dest = 1.0 / (1.0 + dest);
	int left = floor((model_shape.width / seg_size.width * r_x - 0) / scaled_size.width);
	int top = floor((model_shape.height / seg_size.height * r_y - 0) / scaled_size.height);
	int width = ceil(model_shape.width / seg_size.width * r_w / scaled_size.width);
	int height = ceil(model_shape.height / seg_size.height * r_h / scaled_size.height);
	cv::Mat mask;
	resize(dest, mask, cv::Size(width, height));
	return mask(box - cv::Point(left, top)) > mask_thresh;
}


cv::Mat OPENCVInf::get_mask_abs(const cv::Mat& mask_info, const cv::Mat& mask_data, cv::Rect box) {
	cv::Mat rel_mask = get_mask_rel(mask_info, mask_data, box);
	cv::Mat abs_mask = cv::Mat::zeros(origin_size, CV_8UC1);
	cv::resize(rel_mask, abs_mask(box), box.size());
	return abs_mask;
}


std::vector<cv::Point> OPENCVInf::get_contour(const cv::Mat& mask, bool join){
	cv::Mat mask2 = mask.clone();
	cv::Mat mask8;
	mask2.convertTo(mask8, CV_8UC1, 255);
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(mask8, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	if (join){
		std::vector<cv::Point> contour;
		for (int i = 0; i < contours.size(); i++){
			contour.insert(contour.end(), contours[i].begin(), contours[i].end());
		}
		return contour;
	}
	else{
		return contours[0];
	}
}
