#include "polygon_mineral/picodet_openvino.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#define image_size 320


struct object_rect {
    int x;
    int y;
    int width;
    int height;
};

int resize_uniform(cv::Mat &src, cv::Mat &dst, cv::Size dst_size,
                   object_rect &effect_area) {
    int w = src.cols;
    int h = src.rows;
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

    float ratio_src = w * 1.0 / h;
    float ratio_dst = dst_w * 1.0 / dst_h;

    int tmp_w = 0;
    int tmp_h = 0;
    if (ratio_src > ratio_dst) {
        tmp_w = dst_w;
        tmp_h = floor((dst_w * 1.0 / w) * h);
    } else if (ratio_src < ratio_dst) {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0 / h) * w);
    } else {
        cv::resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }
    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));

    if (tmp_w != dst_w) {
        int index_w = floor((dst_w - tmp_w) / 2.0);
        for (int i = 0; i < dst_h; i++) {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3,
                   tmp_w * 3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    } else if (tmp_h != dst_h) {
        int index_h = floor((dst_h - tmp_h) / 2.0);
        memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    } else {
        printf("error\n");
    }
    return 0;
}

void draw_bboxes(const cv::Mat &bgr, const std::vector<BoxInfo> &bboxes,
                 object_rect effect_roi) {
    cv::Mat image = bgr.clone();
    int src_w = image.cols;
    int src_h = image.rows;
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;

    for (size_t i = 0; i < bboxes.size(); i++) {
        const BoxInfo &bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(255,0,0);
        cv::line(image,cv::Point((bbox.x1 - effect_roi.x) * width_ratio,(bbox.y1 - effect_roi.y) * height_ratio),cv::Point((bbox.x2 - effect_roi.x) * width_ratio,(bbox.y2 - effect_roi.y) * height_ratio),color,2);
        cv::line(image,cv::Point((bbox.x2 - effect_roi.x) * width_ratio,(bbox.y2 - effect_roi.y) * height_ratio),cv::Point((bbox.x3 - effect_roi.x) * width_ratio,(bbox.y3 - effect_roi.y) * height_ratio),color,2);
        cv::line(image,cv::Point((bbox.x3 - effect_roi.x) * width_ratio,(bbox.y3 - effect_roi.y) * height_ratio),cv::Point((bbox.x4 - effect_roi.x) * width_ratio,(bbox.y4 - effect_roi.y) * height_ratio),color,2);
        cv::line(image,cv::Point((bbox.x4 - effect_roi.x) * width_ratio,(bbox.y4 - effect_roi.y) * height_ratio),cv::Point((bbox.x1 - effect_roi.x) * width_ratio,(bbox.y1 - effect_roi.y) * height_ratio),color,2);
    }

    cv::imshow("output",image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int image_demo(PicoDet &detector, const char *imagepath) {
    std::vector<std::string> filenames;
    cv::Mat image = cv::imread(imagepath);
    object_rect effect_roi;
    cv::Mat resized_img;
    float score_threshold=0.3;
    float nms_threshold=0.01;
    resize_uniform(image, resized_img, cv::Size(image_size, image_size),
                   effect_roi);
    auto results = detector.detect(resized_img, score_threshold, nms_threshold);
    draw_bboxes(image, results, effect_roi);

    return 0;
}
int main() {
    std::cout << "start init model" << std::endl;
    auto detector = PicoDet("/home/yamabuki/Downloads/picodet_s_processed_benchmark.xml");
    std::cout << "success" << std::endl;
    const char * images="/home/yamabuki/Downloads/5.jpg";
    image_demo(detector, images);
    }
