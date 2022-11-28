#include "polygon_mineral/picodet_openvino.h"

#define image_size 320

void PicoDet::resizeUniform(cv::Mat &src, cv::Mat &dst, const cv::Size &dst_size){
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));
    cv::resize(src,dst,cv::Size(dst_w,dst_h),0,0,1);
}

void PicoDet::drawBboxes(const cv::Mat &bgr, const std::vector<BoxInfo> &bboxes) {
    cv::Mat image = bgr.clone();
    int src_w = image.cols;
    int src_h = image.rows;
    float width_ratio = (float)src_w / (float)image_size;
    float height_ratio = (float)src_h / (float)image_size;

    for (size_t i = 0; i < bboxes.size(); i++) {
        const BoxInfo &bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(210,100,240);
        cv::line(image,cv::Point((bbox.x1 ) * width_ratio,(bbox.y1 ) * height_ratio),cv::Point((bbox.x2 ) * width_ratio,(bbox.y2 ) * height_ratio),color,2);
        cv::line(image,cv::Point((bbox.x2 ) * width_ratio,(bbox.y2 ) * height_ratio),cv::Point((bbox.x3 ) * width_ratio,(bbox.y3 ) * height_ratio),color,2);
        cv::line(image,cv::Point((bbox.x3 ) * width_ratio,(bbox.y3 ) * height_ratio),cv::Point((bbox.x4 ) * width_ratio,(bbox.y4 ) * height_ratio),color,2);
        cv::line(image,cv::Point((bbox.x4 ) * width_ratio,(bbox.y4 ) * height_ratio),cv::Point((bbox.x1 ) * width_ratio,(bbox.y1 ) * height_ratio),color,2);
        cv::Point center ((bbox.x1+bbox.x2+bbox.x3+bbox.x4)/4*width_ratio,(bbox.y1+bbox.y2+bbox.y3+bbox.y4)/4*height_ratio);
        cv::circle(image,center,3,color,3);

        updateKalmen(cv::Point(bbox.x1,bbox.y1),cv::Point(bbox.x2,bbox.y2),cv::Point(bbox.x3,bbox.y3),cv::Point(bbox.x4,bbox.y4));
        cv::Scalar kalman_color = cv::Scalar(240,210,100);
        cv::line(image,cv::Point((kalmanFilter_p1_.statePost.at<float>(0) ) * width_ratio,(kalmanFilter_p1_.statePost.at<float>(1) ) * height_ratio),cv::Point((kalmanFilter_p2_.statePost.at<float>(0) ) * width_ratio,(kalmanFilter_p2_.statePost.at<float>(1) ) * height_ratio),kalman_color,2);
        cv::line(image,cv::Point((kalmanFilter_p2_.statePost.at<float>(0) ) * width_ratio,(kalmanFilter_p2_.statePost.at<float>(1) ) * height_ratio),cv::Point((kalmanFilter_p3_.statePost.at<float>(0) ) * width_ratio,(kalmanFilter_p3_.statePost.at<float>(1) ) * height_ratio),kalman_color,2);
        cv::line(image,cv::Point((kalmanFilter_p3_.statePost.at<float>(0) ) * width_ratio,(kalmanFilter_p3_.statePost.at<float>(1) ) * height_ratio),cv::Point((kalmanFilter_p4_.statePost.at<float>(0) ) * width_ratio,(kalmanFilter_p4_.statePost.at<float>(1) ) * height_ratio),kalman_color,2);
        cv::line(image,cv::Point((kalmanFilter_p4_.statePost.at<float>(0) ) * width_ratio,(kalmanFilter_p4_.statePost.at<float>(1) ) * height_ratio),cv::Point((kalmanFilter_p1_.statePost.at<float>(0) ) * width_ratio,(kalmanFilter_p1_.statePost.at<float>(1) ) * height_ratio),kalman_color,2);
        cv::Point kalmen_center ((kalmanFilter_p1_.statePost.at<float>(0)+kalmanFilter_p2_.statePost.at<float>(0)+kalmanFilter_p3_.statePost.at<float>(0)+kalmanFilter_p4_.statePost.at<float>(0))/4*width_ratio,(kalmanFilter_p1_.statePost.at<float>(1)+kalmanFilter_p2_.statePost.at<float>(1)+kalmanFilter_p3_.statePost.at<float>(1)+kalmanFilter_p4_.statePost.at<float>(1))/4*height_ratio);
        cv::circle(image,kalmen_center,3,kalman_color,3);
    }

    result_publisher_.publish(cv_bridge::CvImage(std_msgs::Header(),cv_image_->encoding , image).toImageMsg());

}

void PicoDet::receiveFromCam(const sensor_msgs::ImageConstPtr& image)
{
    cv_image_ = boost::make_shared<cv_bridge::CvImage>(*cv_bridge::toCvShare(image, image->encoding));
    cv::Mat resized_img;

    resizeUniform(cv_image_->image, resized_img, cv::Size(image_size, image_size));

    auto results = detect(resized_img, score_thresh_, nms_thresh_);
    drawBboxes(cv_image_->image, results);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "poly_mineral");

    std::cout << "start init model" << std::endl;
    PicoDet detect;
    std::cout << "success" << std::endl;
    detect.onInit();
    while (ros::ok())
    {
        ros::spin();
    }
    }
