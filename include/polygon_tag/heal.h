#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <string>
#include <vector>
#include <inference_engine.hpp>
#include "std_msgs/Float32MultiArray.h"
#include "dynamic_reconfigure/server.h"
#include "sensor_msgs/CameraInfo.h"
#include "nodelet/nodelet.h"
#include <pluginlib/class_list_macros.h>
struct Object
{
    cv::Point p1;
    cv::Point p2;
    cv::Point p3;
    cv::Point p4;
    int label;
    float prob;
};
