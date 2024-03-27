#include "opencvinf.hpp"


OPENCVInf::OPENCVInf( const std::string &onnxModelPath, const bool &runWithCuda, const float &accuThresh, const float &maskThresh, const cv::Size &modelInputShape){
    model_path = onnxModelPath;
    model_shape = modelInputShape;
    cuda_enabled = runWithCuda;
    seg_size = cv::Size(160, 160);
    
    net = cv::dnn::readNetFromONNX(model_path);
    if (cuda_enabled){
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}


std::vector<Detection> OPENCVInf::predict(cv::Mat& srcImg, bool only_bbox){
	std::vector<Detection> result;
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
		result.contour = utils::get_contour(result.mask);
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