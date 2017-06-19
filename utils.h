#ifndef UTILS_H__
#define UTILS_H__
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "boost/filesystem.hpp"

#include "RECOG.h"
#include "MFMRECOG.h"
#include "filesystem.h"

using namespace boost::filesystem;
using namespace cv;

int Mat2EImage(cv::Mat &Img, EImage *&image);
int EImage2Mat(const EImage &image, cv::Mat &Img);
int Efacerect2Rect(const EFaceRect &facerect, Rect &r);

int Efeature2Matfeature(EFeature &ef, cv::Mat &f);
int Matfeature2Efeature(cv::Mat &mfeature, EFeature &efeature);
void showimg(cv::Mat &img);
void showEimg(EImage &eimg);
void showEface(EImage &eimg,EFaceRect &efr);
void showface(cv::Mat &img, EFaceRect &efr);
void showLandmarks(cv::Mat &image, cv::Rect &bbox, vector<cv::Point2f> &landmarks);

//根据原图中某个EFaceRect信息进行该处人脸的旋转
void rotateFaceOrin(cv::Mat &srcimg, EFaceRect &efr, cv::Mat &dstimg, EFaceRect &dst_efr);
/*
*图片旋转，旋转之后原图，人脸属性发生变化。
*/
int EFaceProcess_RotateOneFace(EImage & image, EFaceRect &face_rect_list, EImage *& dstImage, EFaceRect &dst_efr);

void ExpandRect(const cv::Mat& src, cv::Mat& dst, Rect rect, Rect &realPosi, Rect &posiInbf);
void adjustfaceRect(EImage &src,EFaceRect &facerect,EImage *&bigface,EFaceRect &dst_efr);
void getNormfaceInbigface(EImage &bigface,EFaceRect &efr,Rect &r);

#endif //UTILS_H__