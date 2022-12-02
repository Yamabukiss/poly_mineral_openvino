#include "polygon_mineral/picodet_openvino.h"

int g_num=6000;


void PicoDet::drawBboxes(const cv::Mat &bgr, const std::vector<BoxInfo> &bboxes) {
    cv::Mat image = bgr.clone();
    std::string label_array[9]={"mineral","solid","exchanger","exchanger","barcode","barcode","90","180","270"};
    static int src_w = image.cols;
    static int src_h = image.rows;
    static float width_ratio = (float)src_w / (float)image_size_;
    static float height_ratio = (float)src_h / (float)image_size_;
    if (!bboxes.empty())
    {
        static std::vector<std::vector<cv::Point>> last_frame_points_saver;
        last_frame_points_saver=last_frame_points_vec_;
        last_frame_points_vec_.clear();
        for (size_t i = 0; i < bboxes.size(); i++) {
            const BoxInfo &bbox = bboxes[i];
            cv::Point center ((bbox.x1+bbox.x2+bbox.x3+bbox.x4)/4*width_ratio,(bbox.y1+bbox.y2+bbox.y3+bbox.y4)/4*height_ratio);

            std::vector<cv::Point> points_vec;
            points_vec.emplace_back(center);
            points_vec.emplace_back(cv::Point(bbox.x1* width_ratio,bbox.y1* height_ratio));
            points_vec.emplace_back(cv::Point(bbox.x2* width_ratio,bbox.y2* height_ratio));
            points_vec.emplace_back(cv::Point(bbox.x3* width_ratio,bbox.y3* height_ratio));
            points_vec.emplace_back(cv::Point(bbox.x4* width_ratio,bbox.y4* height_ratio));

            std::vector<cv::Point> added_weights_points=pointAssignment(points_vec,last_frame_points_saver);
            last_frame_points_vec_.emplace_back(added_weights_points);
            flipSolver(bbox.label);
            static cv::Scalar color = cv::Scalar(205,235,255);
            cv::line(image,added_weights_points[1],added_weights_points[2],color,3);
            cv::line(image,added_weights_points[2],added_weights_points[3],color,3);
            cv::line(image,added_weights_points[3],added_weights_points[4],color,3);
            cv::line(image,added_weights_points[4],added_weights_points[1],color,3);
            cv::circle(image,added_weights_points[0],3,color,3);
            cv::putText(image,label_array[bbox.label],cv::Point(std::max(0,added_weights_points[1].x-10),std::max(added_weights_points[1].y-10,0)),cv::FONT_HERSHEY_SCRIPT_SIMPLEX,2,color,3);
    }
    last_frame_points_saver.clear();
    }
    else
    {
        if (!last_frame_points_vec_.empty()) last_frame_points_vec_.clear();
    }


    result_publisher_.publish(cv_bridge::CvImage(std_msgs::Header(),cv_image_->encoding , image).toImageMsg());

}

void PicoDet::receiveFromCam(const sensor_msgs::ImageConstPtr& image)
{
    cv_image_ = boost::make_shared<cv_bridge::CvImage>(*cv_bridge::toCvShare(image, image->encoding));
//    std::string save_dir = "/home/yamabuki/test/";
//    cv::imwrite(save_dir+std::to_string(g_num)+".jpg",cv_image_->image);
//    cv::waitKey(500);
//    g_num++;
    cv::Mat resized_img;

    resizeUniform(cv_image_->image, resized_img, cv::Size(image_size_, image_size_));

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
