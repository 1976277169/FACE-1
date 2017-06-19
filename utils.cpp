#include "utils.h"
#include "MFMRECOG.h"

int Mat2EImage(cv::Mat &Img, EImage *&image)
{
	if ((Img.data) && (Img.type() == CV_8UC1))
	{
		//分配内存
		image->imagedata = (uchar*)malloc(Img.rows*Img.cols*sizeof(uchar));
		for (int i = 0; i < Img.rows; i++)
		{
			for (int j = 0; j < Img.cols; j++)
			{
				uchar * val = &Img.at<uchar>(i, j);
				image->imagedata[i*Img.cols + j] = *val;
			}
		}
		IplImage tmp = IplImage(Img);
		IplImage *cvImg = &tmp;
		image->width = cvImg->width;
		image->height = cvImg->height;
		image->widthStep = cvImg->widthStep;
		image->image_size = cvImg->imageSize;

		return 0;
	}

	return -1;
}

int EImage2Mat(const EImage &image, cv::Mat &Img)
{
	if (NULL != &image)
	{
		IplImage *cvImg = cvCreateImage(cvSize(0, 0), 8, 1);
		char * tmpc = cvImg->imageData;
		cvImg->imageData = (char*)image.imagedata;
		cvImg->imageSize = image.image_size;
		cvImg->width = image.width;
		cvImg->height = image.height;
		cvImg->widthStep = image.widthStep;
		cvImg->depth = 8;
		cvImg->nChannels = 1;

		cv::Mat tImg(cvImg, 1);
		tImg.copyTo(Img);
		cvImg->imageData = tmpc;
		cvReleaseImage(&cvImg);
		return 0;
	}
	return -1;
}

int Matfeature2Efeature(cv::Mat &mfeature, EFeature &efeature)
{
	char f[256];
	for (int i = 0; i < mfeature.rows; i++)
	{
		f[i] = (char)mfeature.at<float>(i, 0);
	}
	efeature.feature = f;
	return 0;
}
int Efeature2Matfeature(EFeature &ef, cv::Mat &f)
{
	assert((ef.feature_size == f.rows) && (f.rows == 256));
	for (int i = 0; i < 256; i++)
	{
		char *featurep = (char *)ef.feature;
		char *eval = (char *)&featurep[i];
		f.at<float>(i, 0) = *eval;
	}
	return 0;
}

void showimg(cv::Mat &img)
{
	cv::imshow("img", img);
	cv::waitKey(0);
}
void showEimg(EImage &eimg)
{
	Mat img;
	EImage2Mat(eimg, img);
	showimg(img);
}
void showEface(EImage &eimg, EFaceRect &efr)
{
	Mat img;
	EImage2Mat(eimg, img);
	vector<Point> attr;
	attr.push_back(Point(efr.lefteye_x, efr.lefteye_y));
	attr.push_back(Point(efr.righteye_x, efr.righteye_y));
	attr.push_back(Point(efr.nose_x, efr.nose_y));
	attr.push_back(Point(efr.centermouth_x, efr.centermouth_y));

	Rect rect;
	rect.x = efr.left;
	rect.y = efr.top;
	rect.width = efr.right - efr.left;
	rect.height = efr.bottom - efr.top;
	Mat img3ch(img.rows, img.cols, CV_8UC3);
	cvtColor(img, img3ch, CV_GRAY2BGR);
	rectangle(img3ch, rect, Scalar(255, 0, 0), 2);
	showimg(img3ch);

	circle(img3ch, Point(efr.left, efr.top), 2, Scalar(0, 255, 0), -1);
	circle(img3ch, Point(efr.right, efr.bottom), 2, Scalar(0, 255, 0), -1);
	for (int i = 0; i < attr.size(); i++)   //画出形状
	{
		circle(img3ch, attr[i], 2, Scalar(0, 255, 0), -1);
	}

	showimg(img3ch);
}
void showface(Mat &img, EFaceRect &efr)
{
	vector<Point> attr;
	attr.push_back(Point(efr.lefteye_x, efr.lefteye_y));
	attr.push_back(Point(efr.righteye_x, efr.righteye_y));
	attr.push_back(Point(efr.nose_x, efr.nose_y));
	attr.push_back(Point(efr.centermouth_x, efr.centermouth_y));

	Rect rect;
	rect.x = efr.left;
	rect.y = efr.top;
	rect.width = efr.right - efr.left;
	rect.height = efr.bottom - efr.top;
	Mat img3ch(img.rows, img.cols, CV_8UC3);
	cvtColor(img, img3ch, CV_GRAY2BGR);
	rectangle(img3ch, rect, Scalar(255, 0, 0), 2);
	//showimg(img3ch);

	circle(img3ch, Point(efr.left, efr.top), 2, Scalar(0, 255, 0), -1);
	circle(img3ch, Point(efr.right, efr.bottom), 2, Scalar(0, 255, 0), -1);
	for (int i = 0; i < attr.size(); i++)   //画出形状
	{
		circle(img3ch, attr[i], 2, Scalar(0, 255, 0), -1);
	}

	showimg(img3ch);
}
void showLandmarks(cv::Mat &image, Rect &bbox, vector<Point2f> &landmarks) {
	Mat img;
	image.copyTo(img);
	rectangle(img, bbox, Scalar(0, 0, 255), 2);
	for (int i = 0; i < landmarks.size(); i++) {
		Point2f &point = landmarks[i];
		circle(img, point, 2, Scalar(0, 255, 0), -1);
	}
	imshow("landmark", img);
	waitKey(0);
}

void rotateFaceOrin(Mat &srcimg, EFaceRect &efr, Mat &dstimg, EFaceRect &dst_efr)
{
	assert((dstimg.rows == srcimg.rows) && (dstimg.cols == srcimg.cols) && (dstimg.type() == srcimg.type()));

	//旋转时，原图整体不变，只是人脸区域那一块发生旋转。该块区域为frect
	Rect frect(efr.left, efr.top, efr.right - efr.left, efr.bottom - efr.top);
	Mat dsttmp;
	Rect bf;   //记录大脸dsttmp实际取到的位置realgetR
	Rect realposi;
	ExpandRect(srcimg, dsttmp, frect, bf, realposi);  //将原人脸区域frect扩展20%，得到dsttmp大脸，在dsttmp上进行旋转。
	//showimg(dsttmp);

	const double PIE = CV_PI;
	Point A = Point(efr.lefteye_x, efr.lefteye_y);
	Point B = Point(efr.righteye_x, efr.righteye_y);

	double angle = 180 * atan((B.y - A.y) / (double)(B.x - A.x + 1e-12)) / PIE;  //角度制
	double scale = 1.0;
	double cita = atan((B.y - A.y) / (double)(B.x - A.x + 1e-12));//弧度制

	vector<Point> attr;  //6个属性点在大脸中的坐标
	attr.push_back(Point(dsttmp.cols * 4 / 18.0, dsttmp.rows * 4 / 18.0));
	attr.push_back(Point(dsttmp.cols * 14 / 18.0, dsttmp.rows * 14 / 18.0));
	attr.push_back(Point(efr.lefteye_x - frect.x + dsttmp.cols * 4 / 18.0, efr.lefteye_y - frect.y + dsttmp.rows * 4 / 18.0));
	attr.push_back(Point(efr.righteye_x - frect.x + dsttmp.cols * 4 / 18.0, efr.righteye_y - frect.y + dsttmp.rows * 4 / 18.0));
	attr.push_back(Point(efr.nose_x - frect.x + dsttmp.cols * 4 / 18.0, efr.nose_y - frect.y + dsttmp.rows * 4 / 18.0));
	attr.push_back(Point(efr.centermouth_x - frect.x + dsttmp.cols * 4 / 18.0, efr.centermouth_y - frect.y + dsttmp.rows * 4 / 18.0));

	for (int i = 0; i < attr.size(); i++)   //转存到旋转后的形状pA2
	{
		double dis = (double)sqrt(pow((double)attr[i].x - dsttmp.cols / 2.0, 2)
			+ pow((double)attr[i].y - dsttmp.rows / 2.0, 2));
		double cita0;
		if (attr[i].x > dsttmp.cols / 2.0)  //1,4象限
		{
			cita0 = -atan((double)(attr[i].y - dsttmp.rows / 2.0) / (attr[i].x - dsttmp.cols / 2.0 + 1e-12));
		}
		else if ((attr[i].x < dsttmp.cols / 2.0) && (attr[i].y <= dsttmp.rows / 2.0)) //2象限
		{
			cita0 = PIE - atan((attr[i].y - dsttmp.rows / 2.0) / (attr[i].x - dsttmp.cols / 2.0 + 1e-12));
		}
		else if ((attr[i].x < dsttmp.cols / 2.0) && (attr[i].y > dsttmp.rows / 2.0))  //3象限
		{
			cita0 = -1 * PIE - atan((attr[i].y - dsttmp.rows / 2.0) / (attr[i].x - dsttmp.cols / 2.0 + 1e-12));
		}
		else if (attr[i].x == dsttmp.cols / 2.0)    //垂直于y轴
		{
			if (attr[i].y > dsttmp.rows / 2.0)
			{
				cita0 = -1 * PIE / 2.0;
			}
			else if (attr[i].y <= dsttmp.rows / 2.0)
			{
				cita0 = PIE / 2.0;
			}
		}
		else{ cout << "error" << endl; }
		if ((((int)((dsttmp.cols / 2.0) + dis*cos(cita0 + cita))) >= 0) && ((int)(((dsttmp.cols / 2.0) + dis*cos(cita0 + cita))) <= dsttmp.cols - 1))
		{
			attr[i].x = (int)((dsttmp.cols / 2.0) + dis*cos(cita0 + cita));
		}
		else if (((int)(((dsttmp.cols / 2.0) + dis*cos(cita0 + cita)))) < 0)
		{
			attr[i].x = 0;
		}
		else if ((((int)((dsttmp.cols / 2.0) + dis*cos(cita0 + cita)))) >= dsttmp.cols)
		{
			attr[i].x = dsttmp.cols - 1;
		}
		else
		{
			cout << "error with x cordinate= " << attr[i].x << endl;
		}

		if ((((int)((dsttmp.rows / 2.0) - dis*sin(cita0 + cita))) >= 0) && (((int)((dsttmp.rows / 2.0) - dis*sin(cita0 + cita))) <= dsttmp.rows - 1))
		{
			attr[i].y = (int)((dsttmp.rows / 2.0) - dis*sin(cita0 + cita));
		}
		else if ((((int)((dsttmp.rows / 2.0) - dis*sin(cita0 + cita))) < 0))
		{
			attr[i].y = 0;
		}
		else if ((int)(((dsttmp.rows / 2.0) - dis*sin(cita0 + cita))) >= dsttmp.rows)
		{
			attr[i].y = dsttmp.rows - 1;
		}
		else
		{
			cout << "error with y cordinate= " << attr[i].y << endl;
		}
	}
	cv::Mat rot_mat(2, 3, CV_32FC1);
	Point center = Point(dsttmp.cols / 2, dsttmp.rows / 2);
	//std::cout<<"angle= "<<angle<<endl;
	rot_mat = getRotationMatrix2D(center, angle, scale);

	Mat rottmp(dsttmp.rows, dsttmp.cols, dsttmp.type());
	cv::warpAffine(dsttmp, rottmp, rot_mat, dsttmp.size());

	//在大脸上完成旋转,观测旋转后各信息是否正确
	//由于旋转后，人脸框的左上和右下两点发生变化，会变成一个长方形。要将其拓宽为正方形。
	int oriW = attr[1].x - attr[0].x;
	int oriH = attr[1].y - attr[0].y;
	int W_H = max(oriW, oriH);

	attr[0].x -= (W_H - oriW) / 2;
	attr[0].y -= (W_H - oriH) / 2;
	attr[1].x += (W_H - oriW) / 2;
	attr[1].y = attr[0].y + (attr[1].x - attr[0].x); //强制使左上和右下之间构成正方形
	assert(attr[1].x - attr[0].x == attr[1].y - attr[0].y);
	//showimg(rottmp);
	//EFaceRect tm;
	//tm.track_no = efr.track_no;
	//tm.left = attr[0].x;
	//tm.top = attr[0].y;
	//tm.right = attr[1].x;
	//tm.bottom = attr[1].y;
	//tm.lefteye_x = attr[2].x;
	//tm.lefteye_y = attr[2].y;
	//tm.righteye_x = attr[3].x;
	//tm.righteye_y = attr[3].y;
	//tm.nose_x = attr[4].x;
	//tm.nose_y = attr[4].y;
	//tm.centermouth_x = attr[5].x;
	//tm.centermouth_y = attr[5].y;
	//tm.facial_score = efr.facial_score;
	//tm.type = efr.type;
	//showface(rottmp,tm);

	//将大脸放回原图,由于bf是大脸实际能取到的原图中的区域，
	//realposi是bf区域在大脸中的位置（当未出现越界时，bf和大脸一样大）
	rottmp(realposi).copyTo(dstimg(bf));

	//showimg(dstimg);
	// Result  从大脸坐标系变换到原图坐标系
	int x_shift = bf.x - realposi.x;
	int y_shift = bf.y - realposi.y;

	dst_efr.track_no = efr.track_no;


	dst_efr.left = attr[0].x + x_shift;
	dst_efr.top = attr[0].y + y_shift;
	dst_efr.right = attr[1].x + x_shift;
	dst_efr.bottom = attr[1].y + y_shift;
	//x_shift和y_shift可能不同。需要再次变为正方形
	int W_H2 = max(dst_efr.right - dst_efr.left, dst_efr.bottom - dst_efr.top);
	dst_efr.left -= (W_H2 - (dst_efr.right - dst_efr.left)) / 2;
	dst_efr.top -= (W_H2 - (dst_efr.bottom - dst_efr.top)) / 2;
	dst_efr.right += (W_H2 - (dst_efr.right - dst_efr.left)) / 2;
	dst_efr.bottom += (W_H2 - (dst_efr.bottom - dst_efr.top)) / 2;

	dst_efr.lefteye_x = attr[2].x + x_shift;
	dst_efr.lefteye_y = attr[2].y + y_shift;
	dst_efr.righteye_x = attr[3].x + x_shift;
	dst_efr.righteye_y = attr[3].y + y_shift;
	dst_efr.nose_x = attr[4].x + x_shift;
	dst_efr.nose_y = attr[4].y + y_shift;
	dst_efr.centermouth_x = attr[5].x + x_shift;
	dst_efr.centermouth_y = attr[5].y + y_shift;
	dst_efr.facial_score = efr.facial_score;
	dst_efr.type = efr.type;

	//showface(dstimg,dst_efr);

}

int EFaceProcess_RotateOneFace(EImage & image, EFaceRect &face_rect_list,
	EImage *&dstImage, EFaceRect &dst_efr)
{
	Mat src(image.height, image.width, CV_8UC1);
	Mat dst(image.height, image.width, CV_8UC1);
	EImage2Mat(image, src);
	src.copyTo(dst);
	rotateFaceOrin(src, face_rect_list, dst, dst_efr);
	//showimg(dst);
	dstImage->imagedata = (uchar *)malloc(dst.rows*dst.cols*sizeof(uchar));
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			uchar * val = &dst.at<uchar>(i, j);
			dstImage->imagedata[i*dst.cols + j] = *val;
		}
	}
	IplImage tmp1 = IplImage(dst);
	IplImage *cvImg = &tmp1;
	//IplImage *cvImg = &IplImage(dst);

	dstImage->width = cvImg->width;
	dstImage->height = cvImg->height;
	dstImage->widthStep = cvImg->widthStep;
	dstImage->image_size = cvImg->imageSize;

	return 0;
}

int Efacerect2Rect(const EFaceRect &facerect, Rect &r)
{
	r.x = facerect.left;
	r.y = facerect.top;
	r.width = facerect.right - facerect.left;
	r.height = facerect.bottom - facerect.top;

	return 0;
}
void ExpandRect(const Mat& src, Mat& dst, Rect rect, Rect &realPosi, Rect &posiInbf)
{

	// 归一化人脸和矩形框
	const Mat& m = src;
	Mat expanded;
	Mat tmp;
	m.copyTo(tmp);
	//rectangle(tmp, rect, Scalar(0, 0, 255));
	//imshow("face", tmp);
	//waitKey(0);

	const double kPercentX = 0.4;
	const double kPercentY = 0.4;

	// 矩形框扩展后的点
	int left, top, right, bottom;
	left = rect.x - rect.width*kPercentX;
	top = rect.y - rect.height*kPercentY;
	right = rect.x + rect.width + rect.width*kPercentX;
	bottom = rect.y + rect.height + rect.height*kPercentY;
	// 实际图像中能够扩展到的点
	int real_left, real_top, real_right, real_bottom;
	real_left = max(0, left);
	real_top = max(0, top);
	real_right = min(right, m.cols - 1);
	real_bottom = min(bottom, m.rows - 1);
	// 新图像中的点
	int inner_left, inner_top, inner_right, inner_bottom;
	inner_left = real_left - left;
	inner_top = real_top - top;
	inner_right = real_right - left;
	inner_bottom = real_bottom - top;
	// 复制扩展后人脸区域到新图像
	int rows = bottom - top + 1;
	int cols = right - left + 1;
	expanded = Mat::zeros(rows, cols, m.type());
	Rect r1(inner_left, inner_top, inner_right - inner_left + 1, inner_bottom - inner_top + 1);
	Rect r2(real_left, real_top, real_right - real_left + 1, real_bottom - real_top + 1);

	//cout << "m\n" << r2 << endl;
	//cout << "expanded\n" << r1 << endl;
	//cout << expanded.size() << endl;

	m(r2).copyTo(expanded(r1));

	dst = expanded;
	realPosi = r2;
	posiInbf = r1;
}

void adjustfaceRect(EImage &src, EFaceRect &facerect, EImage *&bigface, EFaceRect &dst_efr)
{
	double eup = 0.60;
	double eleft = 0.60;
	double eright = 0.60;
	double edown = 0.60;

	int width = facerect.right - facerect.left;
	int height = facerect.bottom - facerect.top;

	// 矩形框扩展后的点
	int left, top, right, bottom;
	left = facerect.left - width*eleft;
	top = facerect.top - height*eup;
	right = facerect.left + width + width*eright;
	bottom = facerect.top + height + height*edown;

	// 实际图像中能够扩展到的点
	int real_left, real_top, real_right, real_bottom;
	real_left = max(0, left);
	real_top = max(0, top);
	real_right = min(right, src.width - 1);
	real_bottom = min(bottom, src.height - 1);
	// 新图像中的点
	int inner_left, inner_top, inner_right, inner_bottom;
	inner_left = real_left - left;
	inner_top = real_top - top;
	inner_right = real_right - left;
	inner_bottom = real_bottom - top;
	// 复制扩展后人脸区域到新图像
	int rows = bottom - top + 1;
	int cols = right - left + 1;
	int RC = min(rows, cols);  //防止rows cols差1
	Mat tmp = Mat::zeros(RC, RC, CV_8UC1);
	int WH = min(inner_right - inner_left + 1, inner_bottom - inner_top + 1);
	Rect r1(inner_left, inner_top, WH, WH);
	Rect r2(real_left, real_top, WH, WH);

	//cout << "m\n" << r2 << endl;
	//cout << "expanded\n" << r1 << endl;
	//cout << expanded.size() << endl;
	Mat srctmp;
	EImage2Mat(src, srctmp);
	srctmp(r2).copyTo(tmp(r1));

	bigface->imagedata = (uchar *)malloc(tmp.rows*tmp.cols*sizeof(uchar));
	for (int i = 0; i < tmp.rows; i++)
	{
		for (int j = 0; j < tmp.cols; j++)
		{
			uchar * val = &tmp.at<uchar>(i, j);
			bigface->imagedata[i*tmp.cols + j] = *val;
		}
	}
	IplImage tmp3 = IplImage(tmp);
	IplImage *cvImg = &tmp3;
	//IplImage *cvImg = &IplImage(tmp);

	bigface->width = cvImg->width;
	bigface->height = cvImg->height;
	bigface->widthStep = cvImg->widthStep;
	bigface->image_size = cvImg->imageSize;


	//showimg(rottmp);
	//EFaceRect tm;
	//tm.track_no = efr.track_no;
	//tm.left = attr[0].x;
	//tm.top = attr[0].y;
	//tm.right = attr[1].x;
	//tm.bottom = attr[1].y;
	//tm.lefteye_x = attr[2].x;
	//tm.lefteye_y = attr[2].y;
	//tm.righteye_x = attr[3].x;
	//tm.righteye_y = attr[3].y;
	//tm.nose_x = attr[4].x;
	//tm.nose_y = attr[4].y;
	//tm.centermouth_x = attr[5].x;
	//tm.centermouth_y = attr[5].y;
	//tm.facial_score = efr.facial_score;
	//tm.type = efr.type;
	//showface(rottmp,tm);

	//将大脸放回原图,由于bf是大脸实际能取到的原图中的区域，
	//realposi是bf区域在大脸中的位置（当未出现越界时，bf和大脸一样大）
	Rect realposi = r2;
	Rect bf = r1;
	//showimg(dstimg);
	// Result  从大脸坐?晗当浠坏皆甲晗?
	int x_shift = bf.x - realposi.x;
	int y_shift = bf.y - realposi.y;

	dst_efr.track_no = facerect.track_no;


	dst_efr.left = facerect.left + x_shift;
	dst_efr.top = facerect.top + y_shift;
	dst_efr.right = facerect.right + x_shift;
	dst_efr.bottom = facerect.bottom + y_shift;
	//x_shift和y_shift可能不同。需要再次变为正方形
	int W_H2 = max(dst_efr.right - dst_efr.left, dst_efr.bottom - dst_efr.top);
	dst_efr.left -= (W_H2 - (dst_efr.right - dst_efr.left)) / 2;
	dst_efr.top -= (W_H2 - (dst_efr.bottom - dst_efr.top)) / 2;
	dst_efr.right += (W_H2 - (dst_efr.right - dst_efr.left)) / 2;
	dst_efr.bottom = dst_efr.top + (dst_efr.right - dst_efr.left);  //强制变成正方形

	dst_efr.lefteye_x = facerect.lefteye_x + x_shift;
	dst_efr.lefteye_y = facerect.lefteye_y + y_shift;
	dst_efr.righteye_x = facerect.righteye_x + x_shift;
	dst_efr.righteye_y = facerect.righteye_y + y_shift;
	dst_efr.nose_x = facerect.nose_x + x_shift;
	dst_efr.nose_y = facerect.nose_y + y_shift;
	dst_efr.centermouth_x = facerect.centermouth_x + x_shift;
	dst_efr.centermouth_y = facerect.centermouth_y + y_shift;
	dst_efr.facial_score = facerect.facial_score;
	dst_efr.type = facerect.type;
}

void getNormfaceInbigface(EImage &bigface, EFaceRect &efr, Rect &r)
{
	//showEface(bigface, efr);
	//归一化到需要的大小和位置
	int ec_y = efr.lefteye_y;
	int ec_mc_y = efr.centermouth_y - ec_y;

	int newWH = (int)(ec_mc_y*(1 + 80 / 48.0));
	r.width = newWH;
	r.height = newWH;
	r.y = efr.lefteye_y - ec_mc_y * 40 / 48.0;
	r.x = (bigface.width - r.width) / 2;
}