#include "../include/polygon_mineral/heal.h"


inline void dataPreprocess(cv::Mat &img)
{
    cv::resize(img, img, cv::Size(320, 320),0,0,CV_INTER_LINEAR);
}

void blobFromImage(cv::Mat& img,InferenceEngine::MemoryBlob::Ptr &memory_blob)
{
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    if (!memory_blob)
    {
        THROW_IE_EXCEPTION << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
                           << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto mblob_holder = memory_blob->wmap();

    float* blob_data = mblob_holder.as<float*>();

    for (size_t c = 0; c < channels; c++)
    {
        for (size_t h = 0; h < img_h; h++)
        {
            for (size_t w = 0; w < img_w; w++)
            {
                blob_data[c * img_w * img_h + h * img_w + w] = (float)img.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}

void decode(const float* &net_pred)
{
    std::cout<<net_pred[2149]<<std::endl;
}

int main()
{
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network = ie.ReadNetwork("/home/yamabuki/Downloads/picodet_s_processed.xml");
    std::string input_name = network.getInputsInfo().begin()->first;
    std::string output_name = network.getOutputsInfo().begin()->first;
    InferenceEngine::DataPtr output_info = network.getOutputsInfo().begin()->second;
    output_info->setPrecision(InferenceEngine::Precision::FP32);
    std::map<std::string, std::string> config = {
            { InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO },
            { InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::NUMA },
            { InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS,
                    InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_NUMA },
            { InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, "16" }
    };
    InferenceEngine::ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU", config);
    InferenceEngine::InferRequest infer_request = executable_network.CreateInferRequest();
    const InferenceEngine::Blob::Ptr output_blob = infer_request.GetBlob(output_name);
    InferenceEngine::MemoryBlob::CPtr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(output_blob);
    auto moutput_holder = moutput->rmap();
    const float* net_pred =
            reinterpret_cast<const float *>(moutput_holder.as<const InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>());
    InferenceEngine::Blob::Ptr img_blob = infer_request.GetBlob(input_name);
    InferenceEngine::MemoryBlob::Ptr memory_blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(img_blob);
    cv::Mat img=cv::imread("/home/yamabuki/Downloads/5.jpg");
    auto start = std::chrono::high_resolution_clock::now();
    dataPreprocess(img);
    blobFromImage(img,memory_blob);
    infer_request.StartAsync();
    infer_request.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "inference time: " << diff.count() << "s" << std::endl;
    decode(net_pred);
    return 0;
}