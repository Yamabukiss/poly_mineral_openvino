#ifndef _PICODET_OPENVINO_H_
#define _PICODET_OPENVINO_H_

#include <inference_engine.hpp>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <polygon_mineral/dynamicConfig.h>

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
    double score;
    int label;
} BoxInfo;

class PicoDet {
public:
    PicoDet();

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

    std::vector<BoxInfo> detect(cv::Mat image, double score_threshold,
                                double nms_threshold);

    void onInit();

    void receiveFromCam(const sensor_msgs::ImageConstPtr &image);

    void resizeUniform(cv::Mat &src, cv::Mat &dst, const cv::Size &dst_size);

    void drawBboxes(const cv::Mat &bgr, const std::vector<BoxInfo> &bboxes);

    void updateKalmen(const cv::Point &p1, const cv::Point &p2, const cv::Point &p3, const cv::Point &p4);

    void dynamicCallback(polygon_mineral::dynamicConfig &config);

    void preProcess(cv::Mat &image, InferenceEngine::Blob::Ptr &blob);

    void decodeInfer(const float *&cls_pred, const float *&dis_pred, int stride,
                     double threshold,
                     std::vector<std::vector<BoxInfo>> &results);

    BoxInfo disPred2Bbox(const float *&dfl_det, int label, double score, int x,
                         int y, int stride);

    static void nms(std::vector<BoxInfo> &result, float nms_threshold);

//    cv::Mat_<float> process_noise_matrix_;
//    cv::Mat_<float> measure_noise_matrix_;
    cv::Mat measurement_;
    dynamic_reconfigure::Server<polygon_mineral::dynamicConfig> server_;
    dynamic_reconfigure::Server<polygon_mineral::dynamicConfig>::CallbackType callback_;
    double nms_thresh_;
    double score_thresh_;
    int measure_noise_=1;
    int process_noise_=5;
    cv::Mat_<float> transition_matrix_;
//    cv::Mat_<float> measurement_matrix_;
//    cv::Mat_<float> error_cov_;
    std::string input_name_;
    cv_bridge::CvImagePtr cv_image_;
    ros::NodeHandle nh_;
    cv::KalmanFilter kalmanFilter_p1_;
    cv::KalmanFilter kalmanFilter_p2_;
    cv::KalmanFilter kalmanFilter_p3_;
    cv::KalmanFilter kalmanFilter_p4_;
    ros::Subscriber img_subscriber_;
    ros::Publisher result_publisher_;
    int input_size_ = image_size;
    int num_class_ = 4;
    int reg_max_ = 7;
};
#endif