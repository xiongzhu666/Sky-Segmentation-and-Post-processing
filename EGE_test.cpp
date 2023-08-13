#include "ncnn/benchmark.h"
#include "ncnn/datareader.h"
#include "ncnn/net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <chrono>
#include<ctime>
using namespace std;


void get_ncnnmat_max_min(const ncnn::Mat& m, float &Max, float &Min)
{
    float min = 100000;
    float max = -10000;
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int z=0; z<m.d; z++)
        {
            for (int y=0; y<m.h; y++)
            {
                for (int x=0; x<m.w; x++)
                {
                    //printf("%f ", ptr[x]);
                    if(ptr[x] > max)
                    {
                        max = ptr[x];
                    }
                    if(ptr[x] < min)
                    {
                        min = ptr[x];
                    }
                }
                ptr += m.w;
                //printf("\n");
            }
            //printf("\n");
        }
        //printf("------------------------\n");
    }
    Max = max;
    Min = min;
}
int main(int argc, char** argv)
{
    ncnn::Net skynet;
    //skynet.opt.use_vulkan_compute = true;
    skynet.load_param("../EGE_165.ncnn.param");
    skynet.load_model("../EGE_165.ncnn.bin");
    cv::Mat bgr = cv::imread("../eval/233129.jpg", 1);
    cv::Mat dst = bgr.clone();
//    while (dst.rows > 768 && dst.cols >768 ) {
//        pyrDown(dst, dst, cv::Size(dst.cols / 2, dst.rows / 2));
//    }
    int w = bgr.cols;
    int h = bgr.rows;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(dst.data, ncnn::Mat::PIXEL_BGR2RGB, dst.cols, dst.rows, 384, 384);
    const float mean_vals[3] =  {117.790845, 117.790845, 117.790845};
    const float norm_vals[3] = {1.f/64.18484, 1.f/64.18484, 1.f/64.18484};
    in.substract_mean_normalize(mean_vals, norm_vals);

    float MAX, MIN;
    get_ncnnmat_max_min(in, MAX, MIN);
    printf("max: %f, min: %f \n",MAX,MIN);

    const float mean_vals_1[3] =  {MIN, MIN, MIN};
    const float norm_vals_1[3] = {255.f/(MAX-MIN), 255.f/(MAX-MIN), 255.f/(MAX-MIN)};
    //const float norm_vals_1[3] = {(MAX-MIN)/255.f, (MAX-MIN)/255.f, (MAX-MIN)/255.f};
    ncnn::Mat out;
    in.substract_mean_normalize(mean_vals_1, norm_vals_1);
    for (int i = 0; i < 20; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        ncnn::Extractor ex = skynet.create_extractor();
        ex.set_light_mode(true);
        ex.set_num_threads(6);
        ex.input("in0", in);
        
        ex.extract("out5", out);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "代码块运行时间：" << duration.count() << " ms" << std::endl;
    }


    cv::Mat opencv_mask(out.h, out.w, CV_32FC1);
    memcpy((uchar*)opencv_mask.data, out.data, out.w * out.h * sizeof(float));
    cv::resize(opencv_mask,opencv_mask,cv::Size(w,h),cv::INTER_LINEAR);
    cv::imwrite("../test.png", 255*opencv_mask);

    return 0;
}