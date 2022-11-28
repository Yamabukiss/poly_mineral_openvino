#include "polygon_mineral/picodet_openvino.h"


inline float fast_exp(float x) {
  union {
    uint32_t i;
    float f;
  } v{};
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}


template <typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length) {
  const _Tp alpha = *std::max_element(src, src + length);
  _Tp denominator{0};

  for (int i = 0; i < length; ++i) {
    dst[i] = fast_exp(src[i] - alpha);
    denominator += dst[i];
  }

  for (int i = 0; i < length; ++i) {
    dst[i] /= denominator;
  }

  return 0;
}

void PicoDet::onInit()
{
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork model = ie.ReadNetwork("/home/yamabuki/Downloads/picodet_s_processed4.xml");
    // prepare input settings
    InferenceEngine::InputsDataMap inputs_map(model.getInputsInfo());
    input_name_ = inputs_map.begin()->first;
    InferenceEngine::InputInfo::Ptr input_info = inputs_map.begin()->second;
    // prepare output settings
    InferenceEngine::OutputsDataMap outputs_map(model.getOutputsInfo());

    for (auto &output_info : outputs_map) {
        output_info.second->setPrecision(InferenceEngine::Precision::FP32);
    }

    std::map<std::string, std::string> config = {
            { InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO },
            { InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::NUMA },
            { InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS,
                    InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_NUMA },
//            { InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, "16" },
    };


    // get network
    network_ = ie.LoadNetwork(model, "CPU",config);
    infer_request_ = network_.CreateInferRequest();

    img_subscriber_= nh_.subscribe("/hk_camera/image_raw", 1, &PicoDet::receiveFromCam,this);
    result_publisher_ = nh_.advertise<sensor_msgs::Image>("result_publisher", 1);

    callback_ = boost::bind(&PicoDet::dynamicCallback, this, _1);
    server_.setCallback(callback_);

}

PicoDet::PicoDet() {}

void PicoDet::dynamicCallback(polygon_mineral::dynamicConfig &config)
{
    nms_thresh_=config.nms_thresh;
    score_thresh_=config.score_thresh;
    delay_=config.delay;
    ROS_INFO("Seted Complete");
}


PicoDet::~PicoDet() {}

void PicoDet::preProcess(cv::Mat &image, InferenceEngine::Blob::Ptr &blob) {
  int img_w = image.cols;
  int img_h = image.rows;
  int channels = 3;

  InferenceEngine::MemoryBlob::Ptr mblob =
      InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
  if (!mblob) {
    THROW_IE_EXCEPTION
        << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
        << "but by fact we were not able to cast inputBlob to MemoryBlob";
  }
  auto mblobHolder = mblob->wmap();
  float *blob_data = mblobHolder.as<float *>();

  for (size_t c = 0; c < channels; c++) {
    for (size_t h = 0; h < img_h; h++) {
      for (size_t w = 0; w < img_w; w++) {
        blob_data[c * img_w * img_h + h * img_w + w] =
            (float)image.at<cv::Vec3b>(h, w)[c];
      }
    }
  }
}

std::vector<cv::Point> PicoDet::pointAssignment(const std::vector<cv::Point> &frame_points,const std::vector<std::vector<cv::Point>> &last_frame_points_saver)
{
//    return frame_points;
    if (last_frame_points_saver.empty()) return frame_points;
    else
    {
        static std::vector<int> l2_distance_vec;
        for (auto last_points : last_frame_points_saver)
        {
            l2_distance_vec.emplace_back(sqrt(pow(frame_points[0].x-last_points[0].x,2) + pow(frame_points[0].y-last_points[0].y,2)));
        }
        auto min_iter = std::min_element(l2_distance_vec.begin(),l2_distance_vec.end());
        int min_index=std::distance(std::begin(l2_distance_vec), min_iter) ;

        static std::vector<cv::Point> matched_points;
        matched_points=last_frame_points_saver[min_index];

        std::vector<cv::Point> result_points;
        for (int i = 0;i < 5;i++)
        {
            cv::Point added_weights_point  (matched_points[i].x * (1-delay_) + frame_points[i].x * delay_ , matched_points[i].y * (1-delay_) + frame_points[i].y * delay_);
            result_points.emplace_back(added_weights_point);
        }
        l2_distance_vec.clear();
        matched_points.clear();
        return result_points;
    }
}

std::vector<BoxInfo> PicoDet::detect(cv::Mat image, double score_threshold,
                                     double nms_threshold) {
  InferenceEngine::Blob::Ptr input_blob = infer_request_.GetBlob(input_name_);

  preProcess(image, input_blob);
  // do inference
  infer_request_.StartAsync();
  infer_request_.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
  auto end = std::chrono::high_resolution_clock::now();

  // get output
  std::vector<std::vector<BoxInfo>> results;
  results.resize(this->num_class_);

  for (const auto &head_info : this->heads_info_) {
    const InferenceEngine::Blob::Ptr dis_pred_blob =
        infer_request_.GetBlob(head_info.dis_layer);
    const InferenceEngine::Blob::Ptr cls_pred_blob =
        infer_request_.GetBlob(head_info.cls_layer);

    auto mdis_pred =
        InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob);
    auto mdis_pred_holder = mdis_pred->rmap();
    const float *dis_pred = mdis_pred_holder.as<const float *>();

    auto mcls_pred =
        InferenceEngine::as<InferenceEngine::MemoryBlob>(cls_pred_blob);
    auto mcls_pred_holder = mcls_pred->rmap();
    const float *cls_pred = mcls_pred_holder.as<const float *>();
    this->decodeInfer(cls_pred, dis_pred, head_info.stride, score_threshold,
                       results);
  }

  std::vector<BoxInfo> dets;
  for (int i = 0; i < (int)results.size(); i++) {
    this->nms(results[i], nms_threshold);

    for (auto &box : results[i]) {
      dets.push_back(box);
    }
  }
  return dets;
}

void PicoDet::decodeInfer(const float *&cls_pred, const float *&dis_pred,
                           int stride, double threshold,
                           std::vector<std::vector<BoxInfo>> &results) {
  int feature_h = ceil((float)image_size_ / stride);
  int feature_w = ceil((float)image_size_ / stride);
  for (int idx = 0; idx < feature_h * feature_w; idx++) {
    int row = idx / feature_w;
    int col = idx % feature_w;
    float score = 0;
    int cur_label = 0;

    for (int label = 0; label < num_class_; label++) {
      if (cls_pred[idx * num_class_ + label] > score) {
        score = cls_pred[idx * num_class_ + label];
        cur_label = label;
      }
    }
    if (score > threshold) {
      const float *bbox_pred = dis_pred + idx * (reg_max_ + 1) * 8;
      results[cur_label].push_back(
          this->disPred2Bbox(bbox_pred, cur_label, score, col, row, stride));
    }

  }
}

void PicoDet::resizeUniform(cv::Mat &src, cv::Mat &dst, const cv::Size &dst_size){
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));
    cv::resize(src,dst,cv::Size(dst_w,dst_h),0,0,1);
}


BoxInfo PicoDet::disPred2Bbox(const float *&dfl_det, int label, double score,
                              int x, int y, int stride) {
  float ct_x = (x + 0.5) * stride;
  float ct_y = (y + 0.5) * stride;
  std::vector<float> dis_pred;
  dis_pred.resize(8);
  for (int i = 0; i < 8; i++) {
    float dis = 0;
    float *dis_after_sm = new float[reg_max_ + 1];
    activation_function_softmax(dfl_det + i * (reg_max_ + 1), dis_after_sm,
                                reg_max_ + 1);
    for (int j = 0; j < reg_max_ + 1; j++) {
      dis += j * dis_after_sm[j];
    }
    dis *= stride;
    dis_pred[i] = dis;
    delete[] dis_after_sm;
  }
  float x1 = (std::max)(ct_x - dis_pred[0], .0f);
  float y1 = (std::max)(ct_y - dis_pred[1], .0f);
  float x2 = (std::min)(ct_x + dis_pred[2], (float)this->image_size_);
  float y2 = (std::max)(ct_y - dis_pred[3], .0f);

  float x3 = (std::min)(ct_x + dis_pred[4], (float)this->image_size_);
  float y3 = (std::min)(ct_y + dis_pred[5], (float)this->image_size_);
  float x4 = (std::max)(ct_x - dis_pred[6], .0f);
  float y4 = (std::min)(ct_y + dis_pred[7], (float)this->image_size_);
  return BoxInfo{x1 , y1 , x2 , y2 , x3 , y3 , x4 , y4 , score , label };
}

void PicoDet::nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH) {

  std::sort(input_boxes.begin(), input_boxes.end(),
            [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
  std::vector<float> vArea(input_boxes.size());
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    vArea[i] = (input_boxes.at(i).x3 - input_boxes.at(i).x1 + 1) *
               (input_boxes.at(i).y3 - input_boxes.at(i).y1 + 1);
  }
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    for (int j = i + 1; j < int(input_boxes.size());) {
      float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
      float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
      float xx2 = (std::min)(input_boxes[i].x3, input_boxes[j].x3);
      float yy2 = (std::min)(input_boxes[i].y3, input_boxes[j].y3);
      float w = (std::max)(float(0), xx2 - xx1 + 1);
      float h = (std::max)(float(0), yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (vArea[i] + vArea[j] - inter);
      if (ovr >= NMS_THRESH) {
        input_boxes.erase(input_boxes.begin() + j);
        vArea.erase(vArea.begin() + j);
      } else {
        j++;
      }
    }
  }
}