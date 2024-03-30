#include "utils.hpp"


namespace utils {
    void visualizeDetection(cv::Mat &img, std::vector<Detection> &detection, const std::vector<std::string> &classNames){
        cv::Mat mask = img.clone();
        for (int i = 0; i < detection.size(); i++){
            auto color = cv::Scalar(0, 255, 0);
            int left = detection[i].bbox.x;
            int top = detection[i].bbox.y;
            std::string label = classNames[detection[i].id];
            cv::rectangle(img, detection[i].bbox, color, 2 );
            std::string classString = label + ' ' + std::to_string(detection[i].accu).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(left, top - 40, textSize.width + 10, textSize.height + 20);
            cv::rectangle(img, textBox, color, cv::FILLED);
            cv::putText(img, classString, cv::Point(left + 5, top - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
            // if (detection[i].mask.size() == cv::Size(0, 0)){ continue; }
            // spdlog::info("Visualizing detection");
            
            // if (detection[i].mask.rows && detection[i].mask.cols > 0){ mask.setTo(color, detection[i].mask); }
            cv::drawContours(img, std::vector<std::vector<cv::Point>>{detection[i].contour}, -1, color, 2);
        }
        cv::addWeighted(img, 0.6, mask, 0.4, 0, img);
    }

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