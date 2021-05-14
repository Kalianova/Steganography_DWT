#include <opencv2/opencv.hpp>

using namespace cv;

double calc_PSNR(Mat &src, Mat &res) {
    double result = 0;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            result += pow(src.at<uchar>(i, j) - res.at<uchar>(i, j), 2);
        }
    }
    result = 10 * log10(255.0 * 255 / src.rows / src.cols * result);
    return result;
}

double calc_PCC(Mat &src, Mat &res) {
    double result = 0;
    double num = 0;
    double denom_1 = 0;
    double denom_2 = 0;
    double mean_s = mean(src)[0];
    double mean_r = mean(res)[0];
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            num += (src.at<uchar>(i, j) - mean_s) * (res.at<uchar>(i, j) - mean_r);
            denom_1 += pow(src.at<uchar>(i, j) - mean_s, 2);
            denom_2 += pow(res.at<uchar>(i, j) - mean_r, 2);
        }
    }
    return num / sqrt(denom_1 * denom_2);
}

void test_PSNR(Mat& src_image, Mat& res_image, bool multichrome) {
    double res = 0;
    std::vector<Mat> channels_src, channels_res;
    if (src_image.type() == CV_8UC3) {
        split(src_image, channels_src);
        split(res_image, channels_res);
        res = calc_PSNR(channels_src[0], channels_res[0]);
        if (multichrome) {
            res += calc_PSNR(channels_src[1], channels_res[1]);
            res += calc_PSNR(channels_src[2], channels_res[2]);
        }
        res /= 3;
    }
    std::cout << "PSNR: " << res << "  ";
}

void test_PCC(Mat& src_stego, Mat& res_stego) {
    double res = 0;
    std::vector<Mat> channels_src, channels_res;
    if (src_stego.type() == CV_8UC1) {
        res = calc_PCC(src_stego, res_stego);
    }
    else if (src_stego.type() == CV_8UC3) {
        split(src_stego, channels_src);
        split(res_stego, channels_res);
        res = calc_PCC(channels_src[0], channels_res[0]);
        res += calc_PCC(channels_src[1], channels_res[1]);
        res += calc_PCC(channels_src[2], channels_res[2]);
        res /= 3;
    }
    std::cout << "PCC: " << res << std::endl;
}

Mat GenerateKey(Size size, bool mono) {
    Mat key;
    if (mono)
        key = Mat(size, CV_8UC1);
    else
        key = Mat(size, CV_8UC3);
    randu(key, Scalar::all(0), Scalar::all(255));
    return key;
}

Mat EncryptionSecretPictureMono(Mat& mat, Mat key) {
    Mat res = Mat(mat.size(), CV_8UC1);
    if (key.size() == Size(0, 0))
        key = GenerateKey(mat.size(), true);
    bitwise_xor(mat, key, res);
    return key;
}

void DecryptionSecretPictureMono(Mat& mat, Mat key) {
    Mat res = Mat(mat.size(), CV_8UC1);
    bitwise_xor(mat, key, res);
}

Mat EncryptionSecretPicture(Mat& mat, Mat key) {
    Mat res = Mat(mat.size(), CV_8UC3);
    if (key.size() == Size(0, 0))
        key = GenerateKey(mat.size(), false);
    std::vector<Mat> channels0, channels1, channels2;
    split(mat, channels0);
    split(key, channels1);
    split(res, channels2);
    bitwise_xor(channels0[0], channels1[0], channels2[0]);
    bitwise_xor(channels0[1], channels1[1], channels2[1]);
    bitwise_xor(channels0[2], channels1[2], channels2[2]);
    merge(channels2, mat);
    merge(channels1, key);
    return key;
}

void DecryptionSecretPicture(Mat &mat, Mat key) {
    Mat res = Mat(mat.size(), CV_8UC3);
    std::vector<Mat> channels0, channels1, channels2;
    split(mat, channels0);
    split(key, channels1);
    split(res, channels2);
    bitwise_xor(channels0[0], channels1[0], channels2[0]);
    bitwise_xor(channels0[1], channels1[1], channels2[1]);
    bitwise_xor(channels0[2], channels1[2], channels2[2]);
    merge(channels2, mat);
    merge(channels1, key);
}

Mat HaarTransformStego(Mat src) {
    float c, dh, dv, dd;
    int width = src.cols;
    int height = src.rows;
    double a = 0.005;
    Mat tmp = Mat::zeros(src.rows / 2, src.cols / 2, CV_32FC1);
    int k = 0;
    for (int y = 0; y < (height >> 1); y++) {
        for (int x = 0; x < (width >> 1); x++) {
            dd = (src.at<float>(2 * y, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1)) * 0.5;
            tmp.at<float>(y, x) = dd / a;    
        }
    }
    return tmp;
}

Mat HaarTransform(Mat src, Mat stego) {
    float c, dh, dv, dd;
    int width = src.cols;
    int height = src.rows;
    float a = 0.005;
    Mat tmp = Mat::zeros(src.size(), CV_32FC1);
    int k = 0;
    for (int y = 0; y < (height >> 1); y++) {
        for (int x = 0; x < (width >> 1); x++) {
            c = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) + src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1)) * 0.5;
            tmp.at<float>(y, x) = c;

            dh = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x + 1)) * 0.5;
            tmp.at<float>(y, x + (width >> 1)) = dh;

            dv = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y + 1, 2 * x + 1)) * 0.5;
            tmp.at<float>(y + (height >> 1), x) = dv;

            tmp.at<float>(y + (height >> 1), x + (width >> 1)) = a * stego.at<float>(y, x);
        }
    }
    return tmp;
}

Mat InvHaarTransform(Mat src) {
    float c, dh, dv, dd;
    int width = src.cols;
    int height = src.rows;
    Mat dst = Mat::zeros(src.size(), CV_32FC1);;
    int k = 1;
    for (int y = 0; y < (height >> k); y++) {
        for (int x = 0; x < (width >> k); x++) {
            c = src.at<float>(y, x);
            dh = src.at<float>(y, x + (width >> k));
            dv = src.at<float>(y + (height >> k), x);
            dd = src.at<float>(y + (height >> k), x + (width >> k));
            dst.at<float>(y * 2, x * 2) = 0.5 * (c + dh + dv + dd);
            dst.at<float>(y * 2, x * 2 + 1) = 0.5 * (c - dh + dv - dd);
            dst.at<float>(y * 2 + 1, x * 2) = 0.5 * (c + dh - dv - dd);
            dst.at<float>(y * 2 + 1, x * 2 + 1) = 0.5 * (c - dh - dv + dd);
        }
    }
    return dst;
}

void convertSize(Size size, Mat &src) {
    Mat tmp;
    if (src.type() == CV_8UC3) {
        tmp = Mat::zeros(size, CV_8UC3);
        for (size_t i = 0; i < src.rows && i < size.height; i++) {
            for (size_t j = 0; j < src.cols * 3 && j < size.width * 3; j++) {
                tmp.at<uchar>(i, j) = src.at<uchar>(i, j);
            }
        }
    }
    else if (src.type() == CV_16UC3) {
        tmp = Mat::zeros(size, CV_32FC3);
        for (size_t i = 0; i < src.rows && i < size.height; i++) {
            for (size_t j = 0; j < src.cols * 3 && j < size.width * 3; j++) {
                tmp.at<float>(i, j) = src.at<float>(i, j);
            }
        }
    }
    else if (src.type() == CV_8U) {
        tmp = Mat::zeros(size, CV_8UC1);
        for (size_t i = 0; i < src.rows && i < size.height; i++) {
            for (size_t j = 0; j < src.cols && j < size.width; j++) {
                tmp.at<uchar>(i, j) = src.at<uchar>(i, j);
            }
        }
    }
    src = tmp;
}

int intTypeOfImage(Mat img) {
    if (img.type() == CV_8UC3 || img.type() == CV_8UC1)
        return 255;
    if (img.type() == CV_16UC3 || img.type() == CV_16UC1)
        return 255 * 255;
}

void resize_image(Mat& stego, Mat& image) {
    double x = 0;
    double y = 0;
    if (stego.cols > image.cols / 2)
        x = (image.cols + 0.0) / stego.cols / 2;
    if (stego.rows > image.rows / 2)
        y = (image.rows + 0.0) / stego.rows / 2;
    if (x < y && x != 0)
        resize(stego, stego, Size(stego.cols * x, stego.rows * x));
    else if (y != 0)
        resize(stego, stego, Size(stego.cols * y, stego.rows * y));
}

void monochrome(String image_name, String stego_name, String path) {
    Mat image, stego, key;
    Mat image_src = imread(path + image_name, 1);
    image_src.copyTo(image);
    Mat stego_src = imread(path + stego_name, IMREAD_GRAYSCALE);
    stego_src.copyTo(stego);
    resize_image(stego, image);
    key = EncryptionSecretPictureMono(stego, key);
    imwrite(path + "key.png", key);

    convertSize(Size(image.cols / 2, image.rows / 2), stego);
    stego.convertTo(stego, CV_32FC3, 1.0 / intTypeOfImage(stego), 0);

    //Haar Transform
    image.convertTo(image, CV_32FC3, 1.0 / 255, 0);
    std::vector<Mat> rgbChannels(3);
    split(image, rgbChannels);
    rgbChannels[0] = HaarTransform(rgbChannels[0], stego);
    merge(rgbChannels, image);

    //InvHaar Transform
    split(image, rgbChannels);
    rgbChannels[0] = InvHaarTransform(rgbChannels[0]);
    merge(rgbChannels, image);

    image.convertTo(image, CV_8UC3, 1.0 * 255, 0);
    test_PSNR(image_src, image, false);
    imwrite(path + "result_stego.png", image);
    image.convertTo(image, CV_32FC3, 1.0 / intTypeOfImage(image), 0);

    split(image, rgbChannels);
    stego = HaarTransformStego(rgbChannels[0]);
    stego.convertTo(stego, CV_8UC3, 1.0 * 255, 0);
    convertSize(Size(key.cols, key.rows), stego);
    DecryptionSecretPictureMono(stego, key);
    test_PCC(stego_src, stego);
    imwrite(path + "result_image.png", stego);
}

void multichrome(String image_name, String stego_name, String path) {
    Mat image, stego, key;
    Mat image_src = imread(path + image_name, 1);
    image_src.copyTo(image);
    Mat stego_src = imread(path + stego_name, 1);
    stego_src.copyTo(stego);
    resize_image(stego, image);
    key = EncryptionSecretPicture(stego, key);
    imwrite(path + "key.png", key);
    
    convertSize(Size(image.cols / 2, image.rows / 2), stego);
    stego.convertTo(stego, CV_32FC3, 1.0 / intTypeOfImage(stego), 0);
    std::vector<Mat> stegoRgbChannels(3);
    split(stego, stegoRgbChannels);

    //Haar Transform
    image.convertTo(image, CV_32FC3, 1.0 / intTypeOfImage(image), 0);
    std::vector<Mat> rgbChannels(3);
    split(image, rgbChannels);
    rgbChannels[0] = HaarTransform(rgbChannels[0], stegoRgbChannels[0]);
    rgbChannels[1] = HaarTransform(rgbChannels[1], stegoRgbChannels[1]);
    rgbChannels[2] = HaarTransform(rgbChannels[2], stegoRgbChannels[2]);
    merge(rgbChannels, image);

    //InvHaar Transform
    split(image, rgbChannels);
    rgbChannels[0] = InvHaarTransform(rgbChannels[0]);
    rgbChannels[1] = InvHaarTransform(rgbChannels[1]);
    rgbChannels[2] = InvHaarTransform(rgbChannels[2]);
    merge(rgbChannels, image);

    image.convertTo(image, CV_8UC3, 1.0 * 255, 0);
    test_PSNR(image_src, image, true);
    imwrite(path + "result_stego.png", image);
    image.convertTo(image, CV_32FC3, 1.0 / intTypeOfImage(image), 0);

    split(image, rgbChannels);
    rgbChannels[0] = HaarTransformStego(rgbChannels[0]);
    rgbChannels[1] = HaarTransformStego(rgbChannels[1]);
    rgbChannels[2] = HaarTransformStego(rgbChannels[2]);
    merge(rgbChannels, stego);

    stego.convertTo(stego, CV_8UC3, 1.0 * 255, 0);
    convertSize(Size(key.cols, key.rows), stego);
    DecryptionSecretPicture(stego, key);
    test_PCC(stego_src, stego);
    //Mat res;
    //cvtColor(stego, res, COLOR_BGR2GRAY);
    //imwrite(path + "result_image.png", res);
    imwrite(path + "result_image.png", stego);
}

int main(int argc, char* argv[])
{
    monochrome("pale.jpg", "src2.png", "../src_images/");
    multichrome("pale.jpg", "src2.png", "../src_images/");
	waitKey();
	return 0;
}