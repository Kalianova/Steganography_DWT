#include <opencv2/opencv.hpp>

using namespace cv;

Mat GenerateKey(Size size) {
    Mat key = Mat(size, CV_8UC3);
    randu(key, Scalar::all(0), Scalar::all(255));
    return key;
}

Mat EncryptionSecretPicture(Mat &mat) {
    Mat res = Mat(mat.size(), CV_8UC3);
    Mat key = GenerateKey(mat.size());

    std::vector<Mat> channels0;
    std::vector<Mat> channels1;
    std::vector<Mat> channels2;

    split(mat, channels0);
    Mat BlueI(channels0[0]);
    Mat GreenI(channels0[1]);
    Mat RedI(channels0[2]);

    split(key, channels1);
    Mat BlueR(channels1[0]);
    Mat GreenR(channels1[1]);
    Mat RedR(channels1[2]);

    split(res, channels2);
    Mat Blue2(channels2[0]);
    Mat Green2(channels2[1]);
    Mat Red2(channels2[2]);

    bitwise_xor(BlueI, BlueR, Blue2);
    bitwise_xor(GreenI, GreenR, Green2);
    bitwise_xor(RedI, RedR, Red2);

    merge(channels2, mat);
    merge(channels1, key);
    return key;
}


void DecryptionSecretPicture(Mat &mat, Mat key) {
    Mat res = Mat(mat.size(), CV_8UC3);

    std::vector<Mat> channels0;
    std::vector<Mat> channels1;
    std::vector<Mat> channels2;

    split(mat, channels0);
    Mat BlueI(channels0[0]);
    Mat GreenI(channels0[1]);
    Mat RedI(channels0[2]);

    split(key, channels1);
    Mat BlueR(channels1[0]);
    Mat GreenR(channels1[1]);
    Mat RedR(channels1[2]);

    split(res, channels2);
    Mat Blue2(channels2[0]);
    Mat Green2(channels2[1]);
    Mat Red2(channels2[2]);

    bitwise_xor(BlueI, BlueR, Blue2);
    bitwise_xor(GreenI, GreenR, Green2);
    bitwise_xor(RedI, RedR, Red2);

    merge(channels2, mat);
    merge(channels1, key);
}

Mat HaarTransform2(Mat src) {
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
    double a = 0.005;
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
    Mat tmp = Mat::zeros(size, CV_32FC3);
    for (size_t i = 0; i < src.rows && i < size.height; i++) {
        for (size_t j = 0; j < src.cols * 3 && j < size.width * 3; j++) {
            tmp.at<float>(i, j) = src.at<float>(i, j);
        }
    }
    src = tmp;
}

int main()
{
	Mat	image = imread("../src_images/color2.jpg", 1);
    Mat image2 = imread("../src_images/color2.jpg", 1);
	Mat	stego = imread("../src_images/color.jpg", 1);
    //resize(); for stego
    Mat key = EncryptionSecretPicture(stego);    

    stego.convertTo(stego, CV_32FC3, 1.0 / 255, 0);
    convertSize(Size(image.cols / 2, image.rows / 2), stego);
    std::vector<Mat> stegoRgbChannels(3);
    split(stego, stegoRgbChannels);

    //Haar Transform
    image.convertTo(image, CV_32FC3, 1.0 / 255, 0);
    std::vector<Mat> rgbChannels(3);
    split(image, rgbChannels);
    rgbChannels[0] = HaarTransform(rgbChannels[0], stegoRgbChannels[0]);
    rgbChannels[1] = HaarTransform(rgbChannels[1], stegoRgbChannels[1]);
    rgbChannels[2] = HaarTransform(rgbChannels[2], stegoRgbChannels[2]);
    merge(rgbChannels, image);

    //Encription into HH

    //InvHaar Transform
    split(image, rgbChannels);
    rgbChannels[0] = InvHaarTransform(rgbChannels[0]);
    rgbChannels[1] = InvHaarTransform(rgbChannels[1]);
    rgbChannels[2] = InvHaarTransform(rgbChannels[2]);
    merge(rgbChannels, image);

    split(image, rgbChannels);
    rgbChannels[0] = HaarTransform2(rgbChannels[0]);
    rgbChannels[1] = HaarTransform2(rgbChannels[1]);
    rgbChannels[2] = HaarTransform2(rgbChannels[2]);
    merge(rgbChannels, stego);

    convertSize(Size(key.cols, key.rows), stego);
    stego.convertTo(stego, CV_8UC3, 1.0 * 255, 0);
    DecryptionSecretPicture(stego, key);

    image.convertTo(image, CV_8UC3, 1.0 * 255 , 0);
        
	waitKey();
	return 0;
}