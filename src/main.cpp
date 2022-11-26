#include "polygon_mineral/picodet_openvino.h"

#define image_size 320

void PicoDet::resize_uniform(cv::Mat &src, cv::Mat &dst, const cv::Size &dst_size){
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));
    cv::resize(src,dst,cv::Size(dst_w,dst_h),0,0,1);
}

void PicoDet::draw_bboxes(const cv::Mat &bgr, const std::vector<BoxInfo> &bboxes) {
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

    result_publisher.publish(cv_bridge::CvImage(std_msgs::Header(),cv_image_->encoding , image).toImageMsg());

}

void PicoDet::receiveFromCam(const sensor_msgs::ImageConstPtr& image)
{
    cv_image_ = boost::make_shared<cv_bridge::CvImage>(*cv_bridge::toCvShare(image, image->encoding));
    cv::Mat resized_img;
    static float score_threshold=0.1;
    static float nms_threshold=0.01;

    resize_uniform(cv_image_->image, resized_img, cv::Size(image_size, image_size));

    auto results = detect(resized_img, score_threshold, nms_threshold);
    draw_bboxes(cv_image_->image, results);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "poly_mineral");
    ros::NodeHandle nh;

    std::cout << "start init model" << std::endl;
    auto detector = PicoDet("/home/yamabuki/Downloads/picodet_s_processed_benchmark.xml");
    std::cout << "success" << std::endl;
    detector.img_subscriber= nh.subscribe("/galaxy_camera/image_raw", 1, &PicoDet::receiveFromCam,&detector);
    detector.result_publisher = nh.advertise<sensor_msgs::Image>("result_publisher", 1);
    while (ros::ok())
    {
        ros::spin();
    }
    }
