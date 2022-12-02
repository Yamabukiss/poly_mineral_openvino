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
#include "std_msgs/Int8.h"

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

    void dynamicCallback(polygon_mineral::dynamicConfig &config);

    void preProcess(cv::Mat &image, InferenceEngine::Blob::Ptr &blob);

    void decodeInfer(const float *&cls_pred, const float *&dis_pred, int stride,
                     double threshold,
                     std::vector<std::vector<BoxInfo>> &results);
    std::vector<cv::Point> pointAssignment(const std::vector<cv::Point> &frame_points,const std::vector<std::vector<cv::Point>> &last_frame_points_saver);

    BoxInfo disPred2Bbox(const float *&dfl_det, int label, double score, int x,
                         int y, int stride);

    static void nms(std::vector<BoxInfo> &result, float nms_threshold);

    void flipSolver(int label);

    dynamic_reconfigure::Server<polygon_mineral::dynamicConfig> server_;
    dynamic_reconfigure::Server<polygon_mineral::dynamicConfig>::CallbackType callback_;
    std::vector<std::vector<cv::Point>> last_frame_points_vec_;
    double nms_thresh_;
    double score_thresh_;
    double delay_;
    std::string input_name_;
    cv_bridge::CvImagePtr cv_image_;
    ros::NodeHandle nh_;
    ros::Subscriber img_subscriber_;
    ros::Publisher result_publisher_;
    ros::Publisher direction_publisher_;
    int num_class_ = 9;
    int reg_max_ = 7;
    int image_size_ = 320;
};
#endif