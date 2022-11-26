#include "polygon_mineral/picodet_openvino.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#define image_size 320

int resize_uniform(cv::Mat &src, cv::Mat &dst, cv::Size dst_size){
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));
    cv::resize(src,dst,cv::Size(dst_w,dst_h),0,0,0);
    return 0;
}

void draw_bboxes(const cv::Mat &bgr, const std::vector<BoxInfo> &bboxes) {
    cv::Mat image = bgr.clone();
    int src_w = image.cols;
    int src_h = image.rows;
    float width_ratio = (float)src_w / (float)image_size;
    float height_ratio = (float)src_h / (float)image_size;

    for (size_t i = 0; i < bboxes.size(); i++) {
        const BoxInfo &bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(255,0,0);
        cv::line(image,cv::Point((bbox.x1 ) * width_ratio,(bbox.y1 ) * height_ratio),cv::Point((bbox.x2 ) * width_ratio,(bbox.y2 ) * height_ratio),color,2);
        cv::line(image,cv::Point((bbox.x2 ) * width_ratio,(bbox.y2 ) * height_ratio),cv::Point((bbox.x3 ) * width_ratio,(bbox.y3 ) * height_ratio),color,2);
        cv::line(image,cv::Point((bbox.x3 ) * width_ratio,(bbox.y3 ) * height_ratio),cv::Point((bbox.x4 ) * width_ratio,(bbox.y4 ) * height_ratio),color,2);
        cv::line(image,cv::Point((bbox.x4 ) * width_ratio,(bbox.y4 ) * height_ratio),cv::Point((bbox.x1 ) * width_ratio,(bbox.y1 ) * height_ratio),color,2);
    }

//    cv::imshow("output",image);
//    cv::waitKey(0);
//    cv::destroyAllWindows();
}

int image_demo(PicoDet &detector, const char *imagepath) {
    std::vector<std::string> filenames;
    cv::Mat image = cv::imread(imagepath);
    cv::Mat resized_img;
    float score_threshold=0.1;
    float nms_threshold=0.01;

    resize_uniform(image, resized_img, cv::Size(image_size, image_size));

    auto results = detector.detect(resized_img, score_threshold, nms_threshold);
    draw_bboxes(image, results);
    return 0;
}
int main() {
    std::cout << "start init model" << std::endl;
    auto detector = PicoDet("/home/yamabuki/Downloads/picodet_s_processed_benchmark.xml");
    std::cout << "success" << std::endl;
    const char * images="/home/yamabuki/Downloads/5.jpg";
    auto start = std::chrono::high_resolution_clock::now();
    image_demo(detector, images);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "inference time: " << 1/diff.count() << "fps" << std::endl;
    }
