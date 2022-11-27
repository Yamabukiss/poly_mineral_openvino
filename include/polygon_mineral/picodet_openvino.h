#ifndef _PICODET_OPENVINO_H_
#define _PICODET_OPENVINO_H_

#include <inference_engine.hpp>
#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sensor_msgs/Image.h>
#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#define image_size 320

typedef struct HeadInfo {
    std::string cls_layer;
    std::string dis_layer;
    int stride;
} HeadInfo;

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float x3;
    float y3;
    float x4;
    float y4;
    float score;
    int label;
} BoxInfo;

class PicoDet {
public:
    PicoDet(const char *param);

    ~PicoDet();

    InferenceEngine::ExecutableNetwork network_;
    InferenceEngine::InferRequest infer_request_;
    // static bool hasGPU;

    std::vector<HeadInfo> heads_info_{
            // cls_pred|dis_pred|stride
            {"transpose_0.tmp_0", "transpose_1.tmp_0", 8},
            {"transpose_2.tmp_0", "transpose_3.tmp_0", 16},
            {"transpose_4.tmp_0", "transpose_5.tmp_0", 32},
            {"transpose_6.tmp_0", "transpose_7.tmp_0", 64},
    };

    std::vector<BoxInfo> detect(cv::Mat image, float score_threshold,
                                float nms_threshold);
    void receiveFromCam(const sensor_msgs::ImageConstPtr& image);
    void resize_uniform(cv::Mat &src, cv::Mat &dst,  const cv::Size &dst_size);
    void draw_bboxes(const cv::Mat &bgr, const std::vector<BoxInfo> &bboxes);

    ros::Subscriber img_subscriber;
    ros::Publisher result_publisher;

private:
    void preprocess(cv::Mat &image, InferenceEngine::Blob::Ptr &blob);
    void decode_infer(const float *&cls_pred, const float *&dis_pred, int stride,
                      float threshold,
                      std::vector<std::vector<BoxInfo>> &results);
    BoxInfo disPred2Bbox(const float *&dfl_det, int label, float score, int x,
                         int y, int stride);
    static void nms(std::vector<BoxInfo> &result, float nms_threshold);
    std::string input_name_;
    cv_bridge::CvImagePtr cv_image_;
    int input_size_ = image_size;
    int num_class_ = 2;
    int reg_max_ = 7;
};

#endif