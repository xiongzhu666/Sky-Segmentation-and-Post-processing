#include "ncnn/benchmark.h"
#include "ncnn/datareader.h"
#include "ncnn/net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

int main(int argc, char** argv)
{
    ncnn::Net skynet;
    skynet.opt.use_vulkan_compute = true;
    skynet.load_param("../skysegsmall_sim-opt-fp16.param");
    skynet.load_model("../skysegsmall_sim-opt-fp16.bin");
    cv::Mat bgr = cv::imread("../eval/test_800.jpg", 1);
    cv::Mat dst = bgr.clone();
    while (dst.rows >640 && dst.cols >640 ) {
        pyrDown(dst, dst, cv::Size(dst.cols / 2, dst.rows / 2));
    }
    int w = bgr.cols;
    int h = bgr.rows;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(dst.data, ncnn::Mat::PIXEL_BGR2RGB, dst.cols, dst.rows, 320, 320);
    const float mean_vals[3] =  {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};
    const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Extractor ex = skynet.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input("input.1", in);
    ncnn::Mat out;
    ex.extract("1959", out);
    cv::Mat opencv_mask(out.h, out.w, CV_32FC1);
    memcpy((uchar*)opencv_mask.data, out.data, out.w * out.h * sizeof(float));
    cv::resize(opencv_mask,opencv_mask,cv::Size(w,h),cv::INTER_LINEAR);
    cv::imwrite("../test.png", 255*opencv_mask);

    return 0;
}

