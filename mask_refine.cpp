#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include<cmath>
#include <string>
#include<list>

using namespace std;
using namespace cv;
//int n = 0;

struct pix
{
    cv::Point p;
    cv::Vec3f rgb;
};

std::vector<float> bias(const std::vector<float> &x, float b = 0.8)
{
    /*** bias计算：参考论文公式2 ***/
    std::vector<float> denom;
    for (int i = 0; i < x.size(); ++i) {
        float tmp = x[i]/(((1.f / b) - 2.f) * (1.f - x[i]) + 1.f);
        denom.push_back(tmp);
    }
    return denom;
}
cv::Mat probability_to_confidence(cv::Mat &mask, const cv::Mat &rgb, float low_thresh=0.3, float high_thresh=0.5) {
    /* *
     * 计算两个高低掩模：设置高低两个阈值，
     * 低掩模初始化为0，小于低阈值则置1
     * 高掩模初始化为0，大于高阈值则置1
     * */
    cv::Mat low = Mat::zeros(mask.size(), CV_8UC1);
    cv::Mat high = Mat::zeros(mask.size(), CV_8UC1);

    int h = mask.rows;
    int w = mask.cols;
    std::vector<pix> sky;
    std::vector<pix> unknow;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            if((float)mask.at<float>(i, j) < low_thresh)
            {
                low.at<uchar>(i, j) = 1;
            }
            if((float)mask.at<float>(i, j) > high_thresh)
            {
                high.at<uchar>(i, j) = 1;
            }

        }
    }
    //std::cout << "density: " << sky.size() << " " << unknow.size() << endl;
    /* Density Estimation Algorithm
     * 根据论文结果，加入密集度评估后树枝之间会处理得更好，
     * 目前结果并无明显改善，可能是算法理解有偏差，有待研究
     * */
//    int const num = 1024;         //根据论文，为了减小计算量，在天空区域随机取1024个点
//    list<int>::iterator it_sky;   //迭代器
//    list<int> sky_index;          //定义链表，保存生成的随机数
//    int begin, end;               //数字范围
//    int sum;                      //随机数个数
//    begin = 0;
//    end = sky.size()-1;
//    while (sky_index.size() < num)
//    {
//        sky_index.push_back(rand() % (end - begin + 1) + begin);
//        sky_index.sort();         //排序
//        sky_index.unique();       //去除相邻的重复随机数中的第一个
//    }
//    cout << "sky pix num: " << sky_index.size() << endl;
//    const float sigma = 0.01f;
//    const double K = 1.f/sqrt(pow((2.f*CV_PI*sigma*sigma),3));
//    for (int k = 0; k < unknow.size(); ++k) {
//        if(k == 0) {
//            cv::Vec3f unknowrgb = unknow[k].rgb;
//            double new_pi = 0.f;
//            for (it_sky = sky_index.begin(); it_sky != sky_index.end(); it_sky++) {
//                cv::Vec3f skyrgb = sky[*it_sky].rgb;
//                double sum_tmp = 0.f;
//                for (int i = 0; i < 3; ++i) {
//                    sum_tmp = sum_tmp + (unknowrgb[i] - skyrgb[i]) * (unknowrgb[i] - skyrgb[i]);
//                }
//                double sim_rgb = exp((-1.f / 2.f * sigma * sigma) * sum_tmp);
//                sim_rgb = sim_rgb * K;
//                new_pi = new_pi + sim_rgb;
//            }
//            new_pi = new_pi / float(sky_index.size());
//            // cout << "unknow: " << unknow[k].p << " " << new_pi << endl;
//            mask.at<float>(unknow[k].p.y, unknow[k].p.x) = new_pi;
//        }
//        else
//        {
//            continue;
//        }
//    }

    std::vector<float> confidence_low_tmp ;
    std::vector<float> confidence_high_tmp;
    /*根据论文公式1：
     * 1. 低掩模：（l - p）/ l  高掩模：（p - h）/ (1 - h)
     * 2. 分别计算bias
     * */
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            if((int)low.at<uchar>(i, j) == 1)
            {
                confidence_low_tmp.push_back((low_thresh-mask.at<float>(i, j)) / low_thresh);
            }
            if((int)high.at<uchar>(i, j) == 1)
            {
                confidence_high_tmp.push_back((mask.at<float>(i, j) - high_thresh) / (1.f - high_thresh));
            }
        }
    }
    std::vector<float>confidence_low = bias(confidence_low_tmp);
    std::vector<float>confidence_high = bias(confidence_high_tmp);
    cv::Mat confidence =  cv::Mat::zeros(mask.size(), CV_32F);
    float eps = 0.01;
    vector<float>::iterator iter1 = confidence_low.begin();
    vector<float>::iterator iter2 = confidence_high.begin();
    /**参考公式1.计算最后的置信度map**/
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            if((int)low.at<uchar>(i, j) == 1 )
            {
                confidence.at<float>(i, j) = *iter1;
                ++iter1;
            }
            else if((int)high.at<uchar>(i, j) == 1)
            {
                confidence.at<float>(i, j) = *iter2;
                ++iter2;
            }
            else
            {
                continue;
            }
            if(confidence.at<float>(i, j) < eps)
            {
                confidence.at<float>(i, j) = eps;
            }
        }
    }
    return confidence;
}
cv::Mat downsample2_antialiased(const cv::Mat &X)
{
    /*向下采样方法：卷积 + 金字塔2倍尺度下采样*/
    /** filter2D和sepFilter2D两种卷积方法都可以 **/
    Mat dst;
    Mat kx = (Mat_<float>(4, 1) << 1.f/8.f, 3.f/8.f, 3.f/8.f, 1.f/8.f);
    Mat ky = (Mat_<float>(1, 4) << 1.f/8.f, 3.f/8.f, 3.f/8.f, 1.f/8.f);
    Mat kern = (Mat_<float>(3, 3) << 2.f/9.f, 5.f/9.f, 2.f/9.f,
            2.f/9.f, 5.f/9.f, 2.f/9.f,
            2.f/9.f, 5.f/9.f, 2.f/9.f);

    sepFilter2D(X, dst, -1, kx, ky,Point(1,1),0,BORDER_REPLICATE);

    Mat dowmsample;
    // opencv降采样
    float w = (float)dst.cols/ 2.f;
    float h = (float)dst.rows/ 2.f;
    pyrDown(dst, dowmsample,Size(round(w), round(h)));
    return dowmsample;
}

cv::Mat self_resize(cv::Mat &X, cv::Size size)
{
    int w = X.cols;
    int h = X.rows;
    /*若输入图像长宽都大于2倍的目标尺寸，则不断进行向下采样*/
    while(X.cols >= 2 * size.width && X.rows >= 2 * size.height)
    {
        X = downsample2_antialiased(X);
    }
    Mat out;
    /* 线性插值到目标尺寸 */
    cv::resize(X,out,cv::Size(size.width, size.height),0,0, cv::INTER_LINEAR);
    return out;
}

cv::Mat weighted_downsample(cv::Mat &X, const cv::Mat &confidence, int scale, const cv::Size &target_size_input)
{

    Mat XX = X.clone();
    Mat confi = confidence.clone();
    cv::Size target_size;
    int w = XX.cols;
    int h = XX.rows;
    if(scale != -1)
    {
        target_size = cv::Size((round)((float)w/(float)scale),
                               (round)((float)h/(float)scale));
    }
    else
    {
        target_size = target_size_input;
    }

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            if(XX.channels() == 3) {
                XX.at<cv::Vec3f>(i, j)[0] = confi.at<float>(i, j) * XX.at<cv::Vec3f>(i, j)[0];
                XX.at<cv::Vec3f>(i, j)[1] = confi.at<float>(i, j) * XX.at<cv::Vec3f>(i, j)[1];
                XX.at<cv::Vec3f>(i, j)[2] = confi.at<float>(i, j) * XX.at<cv::Vec3f>(i, j)[2];
            }
            if(XX.channels() == 1)
            {
                XX.at<float>(i, j) = confi.at<float>(i, j) * XX.at<float>(i, j);
            }
        }
    }

    Mat numerator = self_resize(XX, target_size);

    Mat denom = self_resize(confi, target_size);

    for (int i = 0; i < numerator.rows; ++i) {
        for (int j = 0; j < numerator.cols; ++j) {
            if(numerator.channels() == 3) {
                numerator.at<cv::Vec3f>(i, j)[0] = numerator.at<cv::Vec3f>(i, j)[0] / denom.at<float>(i, j);
                numerator.at<cv::Vec3f>(i, j)[1] = numerator.at<cv::Vec3f>(i, j)[1] / denom.at<float>(i, j);
                numerator.at<cv::Vec3f>(i, j)[2] = numerator.at<cv::Vec3f>(i, j)[2] / denom.at<float>(i, j);
            }
            if(numerator.channels() == 1)
            {
                numerator.at<float>(i, j) = numerator.at<float>(i, j) / denom.at<float>(i, j);
            }
        }
    }
    return numerator;
}
std::vector<cv::Mat> weighted_downsample(std::vector<cv::Mat> &M_vec, cv::Mat &confidence, int scale, const cv::Size &target_size_input)
{
    /**传入矩阵通道数为6，分解成2个3通道mat**/
    Mat confi = confidence.clone();
    cv::Size target_size;
    int w = M_vec[0].cols;
    int h = M_vec[0].rows;
    if(scale != -1)
    {
        target_size = cv::Size((round)((float)w/(float)scale),
                               (round)((float)h/(float)scale));
    }
    else
    {
        target_size = target_size_input;
    }
    Mat ch1[3], ch2[3];
    ch1[0] = M_vec[0].clone();
    ch1[1] = M_vec[1].clone();
    ch1[2] = M_vec[2].clone();
    ch2[0] = M_vec[3].clone();
    ch2[1] = M_vec[4].clone();
    ch2[2] = M_vec[5].clone();
    Mat m1,m2;
    merge(ch1, 3, m1);
    merge(ch2, 3, m2);
    /*2个3通道map分别乘以置信度map*/
    for (int i = 0; i < m1.rows; ++i) {
        for (int j = 0; j < m1.cols; ++j) {
            m1.at<cv::Vec3f>(i, j)[0] = confidence.at<float>(i, j) * m1.at<cv::Vec3f>(i, j)[0];
            m1.at<cv::Vec3f>(i, j)[1] = confidence.at<float>(i, j) * m1.at<cv::Vec3f>(i, j)[1];
            m1.at<cv::Vec3f>(i, j)[2] = confidence.at<float>(i, j) * m1.at<cv::Vec3f>(i, j)[2];
            m2.at<cv::Vec3f>(i, j)[0] = confidence.at<float>(i, j) * m2.at<cv::Vec3f>(i, j)[0];
            m2.at<cv::Vec3f>(i, j)[1] = confidence.at<float>(i, j) * m2.at<cv::Vec3f>(i, j)[1];
            m2.at<cv::Vec3f>(i, j)[2] = confidence.at<float>(i, j) * m2.at<cv::Vec3f>(i, j)[2];
        }
    }

    Mat m1_re = self_resize(m1,target_size);
    Mat m2_re = self_resize(m2,target_size);

    std::vector<Mat> m1_ch,m2_ch;
    split(m1_re, m1_ch);
    split(m2_re, m2_ch);
    std::vector<Mat> chs;
    chs.push_back(m1_ch[0]);
    chs.push_back(m1_ch[1]);
    chs.push_back(m1_ch[2]);
    chs.push_back(m2_ch[0]);
    chs.push_back(m2_ch[1]);
    chs.push_back(m2_ch[2]);

    Mat conf = confidence.clone();
    Mat denom = self_resize(conf,target_size);
    for (int l = 0; l < chs.size(); ++l) {
        for (int k = 0; k < chs[0].rows; ++k) {
            for (int i = 0; i < chs[0].cols; ++i) {
                chs[l].at<float>(k, i) = chs[l].at<float>(k, i) / denom.at<float>(k, i);
            }
        }
    }
    return chs;
}
std::vector<cv::Mat> outer_product_images(const cv::Mat &X, const cv::Mat &Y)
{
    Mat x_input = X.clone();
    Mat y_input = Y.clone();
    vector<Mat> channels_x, channels_y;
    split(X, channels_x);
    split(Y, channels_y);
    std::vector<cv::Mat> triu_mat;
    // 矩阵乘法，3*3 = 9, 取上对角矩阵，参考公式4
    for (int i = 0; i < channels_x.size(); ++i) {
        for (int j = 0; j < channels_y.size(); ++j) {
            Mat mul_mat = channels_x[i].mul(channels_y[j]);
            if(i <= j)
            {
                triu_mat.push_back(mul_mat);
            }
        }
    }
    return triu_mat;
}
cv::Mat solve_ldl3(const std::vector<Mat> &Covar, const cv::Mat &Residual)
{
    // LDL-decomposition 解压缩算法，引用了文献24
    /*
     * 参考公式7：
        d1 = A11  1
        L_12 = A12 / d1   1
        d2 = A22 - L_12 * A12
        L_13 = A13 / d1
        L_23 = (A23 - L_13 * A12) / d2
        d3 = A33 - L_13 * A13 - L_23 * L_23 * d2
        y1 = b1
        y2 = b2 - L_12 * y1
        y3 = b3 - L_13 * y1 - L_23 * y2
        x3 = y3 / d3
        x2 = y2 / d2 - L_23 * x3
        x1 = y1 / d1 - L_12 * x2 - L_13 * x3
     */
    Mat A11 = Covar[0].clone();
    Mat A12 = Covar[1].clone();
    Mat A13 = Covar[2].clone();
    Mat A22 = Covar[3].clone();
    Mat A23 = Covar[4].clone();
    Mat A33 = Covar[5].clone();
    cv::Mat residual = Residual.clone();
    std::vector<Mat> b;
    split(residual, b);
    int w = A11.cols;
    int h = A11.rows;
    cv::Mat L_12 = Mat::zeros(cv::Size (w,h), CV_32F);
    cv::Mat L_13 = Mat::zeros(cv::Size (w,h), CV_32F);
    cv::Mat L_23 = Mat::zeros(cv::Size (w,h), CV_32F);
    cv::Mat d1 = Mat::zeros(cv::Size (w,h), CV_32F);
    cv::Mat d2 = Mat::zeros(cv::Size (w,h), CV_32F);
    cv::Mat d3 = Mat::zeros(cv::Size (w,h), CV_32F);
    cv::Mat y1 = Mat::zeros(cv::Size (w,h), CV_32F);
    cv::Mat y2 = Mat::zeros(cv::Size (w,h), CV_32F);
    cv::Mat y3 = Mat::zeros(cv::Size (w,h), CV_32F);
    cv::Mat x1 = Mat::zeros(cv::Size (w,h), CV_32F);
    cv::Mat x2 = Mat::zeros(cv::Size (w,h), CV_32F);
    cv::Mat x3 = Mat::zeros(cv::Size (w,h), CV_32F);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            //d1 = A11
            d1.at<float>(i, j) = A11.at<float>(i, j);
            //L_12 = A12 / d1
            L_12.at<float>(i, j) = A12.at<float>(i, j) / d1.at<float>(i, j);
            //d2 = A22 - L_12 * A12
            d2.at<float>(i, j) = A22.at<float>(i, j) -  L_12.at<float>(i, j) * A12.at<float>(i, j);
            //L_13 = A13 / d1
            L_13.at<float>(i, j) = A13.at<float>(i, j) / d1.at<float>(i, j);
            //L_23 = (A23 - L_13 * A12) / d2
            L_23.at<float>(i, j) = (A23.at<float>(i, j) - L_13.at<float>(i, j)*A12.at<float>(i, j)) / d2.at<float>(i, j);
            //d3 = A33 - L_13 * A13 - L_23 * L_23 * d2
            d3.at<float>(i, j) = A33.at<float>(i, j) - L_13.at<float>(i, j)*A13.at<float>(i, j) -L_23.at<float>(i, j)*L_23.at<float>(i, j)*d2.at<float>(i, j);
            //y1 = b1
            y1.at<float>(i, j) = b[0].at<float>(i, j);
            //y2 = b2 - L_12 * y1
            y2.at<float>(i, j) = b[1].at<float>(i, j) - L_12.at<float>(i, j) * y1.at<float>(i, j);
            //y3 = b3 - L_13 * y1 - L_23 * y2
            y3.at<float>(i, j) = b[2].at<float>(i, j) - L_13.at<float>(i, j) * y1.at<float>(i, j) - L_23.at<float>(i, j) * y2.at<float>(i, j);
            //x3 = y3 / d3
            x3.at<float>(i, j) = y3.at<float>(i, j)/d3.at<float>(i, j);
            //x2 = y2 / d2 - L_23 * x3
            x2.at<float>(i, j) = y2.at<float>(i, j)/d2.at<float>(i, j) - L_23.at<float>(i, j) * x3.at<float>(i, j);
            //x1 = y1 / d1 - L_12 * x2 - L_13 * x3
            x1.at<float>(i, j) = y1.at<float>(i, j)/d1.at<float>(i, j) - L_12.at<float>(i, j) * x2.at<float>(i, j) - L_13.at<float>(i, j) * x3.at<float>(i, j);
        }
    }
    std::vector<cv::Mat> ldl3_vec;
    ldl3_vec.push_back(x1);
    ldl3_vec.push_back(x2);
    ldl3_vec.push_back(x3);
    cv::Mat ldl3;
    merge(ldl3_vec, ldl3);
    return ldl3;
    //cv::Mat L_12 = A2 / d1;
}
cv::Mat smooth_upsample(cv::Mat &X, cv::Size sz)
{
    cv::Mat XX = X.clone();
    float x[2];
    float s[2];
    x[0] = X.cols;
    x[1] = X.rows;
    s[0] = sz.width;
    s[1] = sz.height;

    float log4ratio_1 = 0.5 * log2(s[0]/x[0]);
    float log4ratio_2 = 0.5 * log2(s[1]/x[1]);
    float log4ratio = log4ratio_1 > log4ratio_2? log4ratio_1:log4ratio_2;

    int num_steps = 1 > round(log4ratio)? 1 : round(log4ratio);

    float ratio[2];
    ratio[0] = (float)sz.width / (float)X.cols;
    ratio[1] = (float)sz.height / (float)X.rows;
    float ratio_per_step[2];
    ratio_per_step[0] = x[0] * ratio[0] / (float)num_steps;
    ratio_per_step[1] = x[1] * ratio[1] / (float)num_steps;

    for (int i = 1; i < (num_steps+1); ++i) {
        cv::Size target_shape_for_step = cv::Size (round(ratio_per_step[0] * (float)i), round(ratio_per_step[1] * (float)i));
        XX = self_resize(XX, target_shape_for_step);
    }
    return XX;
}
int main() {
    Mat bgr = imread("../eval/IMG_20210309_211233.jpg",CV_32F);
    cvtColor(bgr,bgr,CV_BGR2RGB);
    Mat src = imread("../eval/IMG_20210309_211233.png",CV_32F);
    std::string pathname = "../eval/";
    std::string outname = pathname + "IMG_20210309_211233" + "_result.png";
    int W = src.cols;
    int H = src.rows;

    Mat reference = bgr.clone();

    vector<Mat> channels;
    split(src, channels);
    Mat ch1 = channels.at(0);
    Mat ch2 = channels.at(1);
    Mat mask, mask2;
    ch1.convertTo(mask, CV_32F, 1.0/255, 0);
    reference.convertTo(reference, CV_32FC3, 1.0/255, 0);
    Mat img = reference.clone();

    Mat confidence = probability_to_confidence(mask,reference);

    Mat conf = confidence.clone();
    Mat refer1 = reference.clone();
    int kernel = 256;
    cv::Mat reference_small = weighted_downsample(refer1, confidence, kernel, cv::Size(0,0));

    int small_h = reference_small.size[0];
    int small_w = reference_small.size[1];

    Mat conf1 = confidence.clone();
    cv::Mat source_small = weighted_downsample(mask, conf1, -1, cv::Size(small_w, small_h));

    std::vector<cv::Mat> outer_reference = outer_product_images(reference,reference);

    Mat conf2 = confidence.clone();
    vector<Mat> Outer_Reference = weighted_downsample(outer_reference, conf2, -1, cv::Size(small_w, small_h));

    std::vector<cv::Mat> tri_vec_out = outer_product_images(reference_small, reference_small);

//    //分离 Outer_Reference
    vector<Mat> covar;
//    split(Outer_Reference, covar);
    for (int l = 0; l < tri_vec_out.size(); ++l) {
        cv::Mat tri = tri_vec_out[l];
        cv::Mat var_tmp = Mat::zeros(tri.size(), CV_32F);
        for (int i = 0; i < tri.rows; ++i) {
            for (int j = 0; j < tri.cols; ++j) {
                var_tmp.at<float>(i, j) = Outer_Reference[l].at<float>(i, j) - tri.at<float>(i, j);
            }
        }
        covar.push_back(var_tmp);
    }

    cv::Mat ref_src = Mat::zeros(mask.size(), CV_32FC3);
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            ref_src.at<cv::Vec3f>(i, j)[0] = mask.at<float>(i, j) * reference.at<cv::Vec3f>(i, j)[0];
            ref_src.at<cv::Vec3f>(i, j)[1] = mask.at<float>(i, j) * reference.at<cv::Vec3f>(i, j)[1];
            ref_src.at<cv::Vec3f>(i, j)[2] = mask.at<float>(i, j) * reference.at<cv::Vec3f>(i, j)[2];
        }
    }
    Mat conf3 = confidence.clone();
    Mat var = weighted_downsample(ref_src, conf3, -1, cv::Size(small_w, small_h));
//
//    /* residual_small = var - reference_small * source_small[..., np.newaxis] */
    cv::Mat residual_small = Mat::zeros(cv::Size(small_w, small_h), CV_32FC3);
    for (int i = 0; i < small_h; ++i) {
        for (int j = 0; j < small_w; ++j) {
            residual_small.at<cv::Vec3f>(i, j)[0] = var.at<cv::Vec3f>(i, j)[0] - source_small.at<float>(i, j)* reference_small.at<cv::Vec3f>(i, j)[0];
            residual_small.at<cv::Vec3f>(i, j)[1] = var.at<cv::Vec3f>(i, j)[1] - source_small.at<float>(i, j)* reference_small.at<cv::Vec3f>(i, j)[1];
            residual_small.at<cv::Vec3f>(i, j)[2] = var.at<cv::Vec3f>(i, j)[2] - source_small.at<float>(i, j)* reference_small.at<cv::Vec3f>(i, j)[2];
        }
    }
//    cv::imwrite("../debug_image/residual_small_c.png",255*residual_small);
    for (int m = 0; m < covar.size(); ++m) {
        if(m == 0 || m == 3|| m == 5) {
            for (int k = 0; k < covar[0].rows; ++k) {
                for (int i = 0; i < covar[0].cols; ++i) {
                    covar[m].at<float>(k, i) = covar[m].at<float>(k, i) + 0.01 * 0.01;
                }
            }
        }
    }

    cv::Mat affine = solve_ldl3(covar, residual_small);

    cv::Mat residual = Mat::zeros(cv::Size(small_w, small_h), CV_32F);
    for (int m = 0; m < covar[0].rows; ++m) {
        for (int i = 0; i < covar[0].cols; ++i) {
            float r = affine.at<cv::Vec3f>(m, i)[0] * reference_small.at<cv::Vec3f>(m, i)[0];
            float g = affine.at<cv::Vec3f>(m, i)[1] * reference_small.at<cv::Vec3f>(m, i)[1];
            float b = affine.at<cv::Vec3f>(m, i)[2] * reference_small.at<cv::Vec3f>(m, i)[2];
            float sum = r+b+g;
            residual.at<float>(m, i) = source_small.at<float>(m, i) - sum;
        }
    }

    cv::Mat affine_modify = smooth_upsample(affine, cv::Size(W, H));
    cv::Mat residual_modify = smooth_upsample(residual, cv::Size(W, H));

    cv::Mat output = Mat::zeros(cv::Size(W, H), CV_32F);
    for (int i1 = 0; i1 < output.rows; ++i1) {
        for (int i = 0; i < output.cols; ++i) {
            float r = img.at<cv::Vec3f>(i1, i)[0] * affine_modify.at<cv::Vec3f>(i1, i)[0];
            float g = img.at<cv::Vec3f>(i1, i)[1] * affine_modify.at<cv::Vec3f>(i1, i)[1];
            float b = img.at<cv::Vec3f>(i1, i)[2] * affine_modify.at<cv::Vec3f>(i1, i)[2];
            float sum = r+g+b;
            output.at<float>(i1, i) = sum + residual_modify.at<float>(i1, i);
            if(output.at<float>(i1, i) > 1.f)
            {
                output.at<float>(i1, i) = 1.f;
            }
            if(output.at<float>(i1, i) < 0)
            {
                output.at<float>(i1, i) = 0.f;
            }
            output.at<float>(i1, i) = 255.f * output.at<float>(i1, i);
        }
    }
    cv::imwrite(outname,output);

    // bilateralFilter
    Mat out_bilf;
    bilateralFilter(output, out_bilf, 0, 20, 10);
    cv::imwrite("../eval/IMG_20210309_211233_filter.png",out_bilf);

    //cv::imwrite(outname,255 * output);


    return 0;
}


