#include "MFMRECOG.h"
#include "utils.h"
#include <boost/algorithm/string.hpp>
#include "fastCluster.h"
#include "common.h"

using namespace boost::algorithm;

int MFMRECOG::E_Init(const char *modulepath)
{
	string kccpath = string(modulepath) + "/face.xml";
	string kccppath = string(modulepath) + "/pface.xml";
	string cfvnetpath = string(modulepath) + "/facevalidnet.prototxt";
	string cfvmodelpath = string(modulepath) + "/facevalidnet_iter_450000.caffemodel";
	string lderpath = string(modulepath) + "/deeplandmark";
	//string prototxtpath = string(modulepath) + "/LightenedCNN_B.prototxt"; 
	string prototxtpath = string(modulepath) + "/LCNN_NET.prototxt"; 
	//string caffemodelpath = string(modulepath) + "/_iter_4005800.caffemodel";
	string caffemodelpath = string(modulepath) + "/LightenedCNN_B.caffemodel";
	if (!FileSystem::isExists(string(modulepath)))
	{
		cerr << "module path is invalid!" << endl;
		return -1;
	}
	else if ((!FileSystem::isExists(kccpath)) || (!FileSystem::isExists(kccppath))
		|| (!FileSystem::isExists(cfvnetpath)) || (!FileSystem::isExists(cfvmodelpath))
		||(!FileSystem::isExists(lderpath)) || (!FileSystem::isExists(prototxtpath))
		|| (!FileSystem::isExists(caffemodelpath)))
	{
		cerr << "check your kccpath or kccppath or cfvnetpath or cfvmodelpath or lderpath or facerecog's proto or model." << endl;
		return -1;
	}
	//初始化M的模型
	kcc = CascadeClassifier(kccpath);
	kccp = CascadeClassifier(kccppath);
	cfv =new CaffeFaceValidator(cfvnetpath, cfvmodelpath);
	lder.LoadModel(lderpath);
	//protonet = string(modulepath) + "/DeepFacenet.prototxt";
	protonet = prototxtpath;
	//caffemodel = string(modulepath) + "/DeepFacenet_iter.caffemodel";
	caffemodel = caffemodelpath;
	fe = new featureExer(protonet, caffemodel);

	return 0;
}

int MFMRECOG::E_UnInit()
{
	delete cfv;
	delete fe;

	return 0;
}

int MFMRECOG::EFaceProcess_Facedetect(const EImage& image, int& list_size,
	std::vector<EFaceRect> &face_rect_list, int method)
{
	cv::Mat img;
	EImage2Mat(image,img);
	Mat gray, gray2;
	assert(img.data != NULL);
	if (img.type() == CV_8UC3) 
	{
		cvtColor(img, gray, CV_BGR2GRAY);
		cvtColor(img, gray2, CV_BGR2GRAY);
	}
	else if (img.type() == CV_8UC1) 
	{
		img.copyTo(gray);
		img.copyTo(gray2);
	}
	else
	{
		return -1;
	}

	int flag = 1;
	int norm_width, norm_height;
	if (((gray.rows > 600) && (gray.rows <= 1200)) || ((gray.cols > 600) && (gray.cols <= 1200)))
	{
		flag = 2;
	}
	else if (((gray.rows > 1200) && (gray.rows <= 2400)) || ((gray.cols > 1200) && (gray.cols <= 2400)))
	{
		flag = 3;
	}
	else if ((gray.rows > 2400) || (gray.cols > 2400))
	{
		flag = 5;
	}

	norm_width = gray.cols / flag;
	norm_height = gray.rows / flag;

	resize(gray,gray,Size(norm_width,norm_height));

	equalizeHist(gray, gray);

	vector<Rect> rects;
	if (method == 0)
	{
		kcc.detectMultiScale(gray,
			rects,
			1.1,
			2,
			0 | CV_HAAR_SCALE_IMAGE,
			Size(50, 50),
			Size(gray.cols , gray.rows ));
	}
	else if (method==1)
	{
		kccp.detectMultiScale(gray,
			rects,
			1.2,
			2,
			0 | CV_HAAR_SCALE_IMAGE,
			Size(30, 30),
			Size(2 * gray.cols / 3, 2 * gray.rows / 3));
	}
	else
	{
		cerr << "method must be 0 or 1" << endl;
	}

	if (flag > 1)
	{
		for (int i = 0; i < rects.size(); i++)
		{
			rects[i].x *= flag;
			rects[i].y *= flag;
			rects[i].width *= flag;
			rects[i].height *= flag;
		}
	}

	list_size = rects.size();
	//对每一张人脸，记录其面部属性
	vector<Point2f> landmarks;
	for (int i = 0; i < rects.size(); i++)
	{
		//使用caffe判断是否为人脸
		int j = 0;
		bool result = true;
		float sc = 0;
		Mat imgforcaffe;
		gray2(rects[i]).convertTo(imgforcaffe, CV_32F, 1.0 / 256.0);
		cfv->validate(imgforcaffe*256.0, result, sc);
		if (result)
		{
			//如果是人脸，则提取相应属性
			
			EFaceRect tmpfrect;

			tmpfrect.track_no = j;
			j++;
			tmpfrect.left = rects[i].x;
			tmpfrect.top = rects[i].y;
			tmpfrect.right = rects[i].x + rects[i].width;
			tmpfrect.bottom = rects[i].y + rects[i].height;
			BBox bbox_ = BBox(rects[i]).subBBox(0.1, 0.9, 0.2, 1);
			landmarks = lder.DetectLandmark(gray2, bbox_);
			//showLandmarks(gray, bbox_.rect, landmarks);
			tmpfrect.lefteye_x = (int)landmarks[0].x;
			tmpfrect.lefteye_y = (int)landmarks[0].y;
			tmpfrect.righteye_x = (int)landmarks[1].x;
			tmpfrect.righteye_y = (int)landmarks[1].y;
			tmpfrect.nose_x = (int)landmarks[2].x;
			tmpfrect.nose_y = (int)landmarks[2].y;
			tmpfrect.centermouth_x = (int)((landmarks[3].x + landmarks[4].x) / 2);
			tmpfrect.centermouth_y = (int)((landmarks[3].y + landmarks[4].y) / 2);

			tmpfrect.facial_score = sc;
			tmpfrect.type = 0;
			face_rect_list.push_back(tmpfrect);
		}
		else
		{
			list_size--;
		}
	}
	return 0;
}

int MFMRECOG::EFaceProcess_RotateOneFace(EImage & image, EFaceRect &face_rect_list, 
	EImage *&dstImage, EFaceRect &dst_efr)
{
	Mat src(image.height, image.width, CV_8UC1);
	Mat dst(image.height, image.width, CV_8UC1);
	EImage2Mat(image,src);
	src.copyTo(dst);
	rotateFaceOrin(src,face_rect_list,dst,dst_efr);

	dstImage->imagedata = (uchar *)malloc(dst.rows*dst.cols*sizeof(uchar));
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			uchar * val = &dst.at<uchar>(i, j);
			dstImage->imagedata[i*dst.cols + j] = *val;
		}
	}
	IplImage tmp = IplImage(dst);
	IplImage *cvImg = &tmp;

	dstImage->width = cvImg->width;
	dstImage->height = cvImg->height;
	dstImage->widthStep = cvImg->widthStep;
	dstImage->image_size = cvImg->imageSize;
	
	return 0;
}

int MFMRECOG::EFaceProcess_Getface(EImage & image, EFaceRect &facerect, EImage *&face)
{
	//获得人脸是为了提取特征，因此函数内部涉及旋转和归一化
	//对图片中的这张脸，根据eyes进行原图旋转
	//showEimg(image);
	struct tagEImage dsttmp = { NULL, 0, 0, 0, 0 };
	EImage *dstimg = &dsttmp;  //旋转后的图片
	EFaceRect dst_efr;  //旋转后图片中该人脸的属性
	EFaceProcess_RotateOneFace(image, facerect, dstimg, dst_efr);
	//showEimg(*dstimg);
	struct tagEImage bftmp = { NULL, 0, 0, 0, 0 };
	EImage *bigface = &bftmp;  //旋转后的图片
	EFaceRect fr_bigface;
	adjustfaceRect(*dstimg, dst_efr, bigface, fr_bigface);
	//showEface(*bigface,fr_bigface);
	assert(bigface->width==bigface->height);
	//showEimg(*bigface);
	//showEface(*dstimg,dst_efr);
	Rect r; //获得所需要的人脸区域在bigface中的位置
	getNormfaceInbigface(*bigface,fr_bigface,r);
	//showEimg(*bigface);
	//showEface(*dstimg,dst_efr);
	Mat img(bigface->height, bigface->width, CV_8UC1);
	EImage2Mat(*bigface, img);
	Mat tmpface(r.height, r.width, CV_8UC1);
	img(r).copyTo(tmpface); 
	//showimg(tmpface);
	face->imagedata = (uchar *)malloc(tmpface.rows*tmpface.cols*sizeof(uchar));
	for (int i = 0; i < tmpface.rows; i++)
	{
		for (int j = 0; j < tmpface.cols; j++)
		{
			uchar * val = &tmpface.at<uchar>(i, j);
			face->imagedata[i*tmpface.cols + j] = *val;
		}
	}
	IplImage tmp = IplImage(tmpface);
	IplImage *cvImg = &tmp;

	face->width = cvImg->width;
	face->height = cvImg->height;
	face->widthStep = cvImg->widthStep;
	face->image_size = cvImg->imageSize;

	EFaceProcess_FreeImage(dstimg);
	EFaceProcess_FreeImage(bigface);
	return 0;
}

int MFMRECOG::EFaceProcess_GetFeatureParam(int &feature_size)
{
	feature_size = 256;
	return 0;
}

int MFMRECOG::EFaceProcess_GetFaceFeature(EImage &image, EFaceRect &facerect,
	EFeature &feature)
{
	Mat face;
	EFeature efeature;
	EImage tmpface = { NULL, 0, 0, 0, 0 };
	EImage *eface = &tmpface;
	EFaceProcess_Getface(image, facerect, eface);
	EImage2Mat(*eface, face);

	EFaceProcess_getFacefeature(face, feature);
	EFaceProcess_FreeImage(eface);
	return 0;
}

int MFMRECOG::EFaceProcess_ReadImage(const std::string &imgpath, EImage *eimg)
{
	Mat img = imread(imgpath,0);
	struct tagEImage t = {NULL,0,0,0,0};
	EImage *eimgtmp = &t;
	Mat2EImage(img,eimgtmp);
	eimg->imagedata = eimgtmp->imagedata;
	eimg->height = eimgtmp->height;
	eimg->width = eimgtmp->width;
	eimg->image_size = eimgtmp->image_size;
	eimg->widthStep = eimgtmp->widthStep;

	return 0;
}

int MFMRECOG::EFaceProcess_SaveImage(const EImage& eimg, std::string &savepath)
{

	Mat img;
	EImage2Mat(eimg,img);
	imwrite(savepath,img);

	return 0;
}

int MFMRECOG::EFaceProcess_FreeImage(EImage *image)
{
	free(image->imagedata);
	image->imagedata = NULL;
	return 0;
}

int MFMRECOG::EFaceProcess_FeatureCompare(const EFeature& query_feature,
	const EFeature& ref_feature, double& similarity)
{
	Mat f(256, 1, CV_32FC1);
	Mat r(256, 1, CV_32FC1);
	Efeature2Mfeature(query_feature,f);
	Efeature2Mfeature(ref_feature,r);
	cosSimilarity(f,r,similarity);
	return 0;
}

int MFMRECOG::EFaceProcess_GetFaceFeature(cv::Mat &face, EFeature &feature)
{
	Mat mfeature(256, 1, CV_32FC1);
	assert(face.type() == CV_8UC1);
	resize(face, face, cvSize(128, 128));
	face.convertTo(face, CV_32FC1, 1.0 / 256.0);
	fe->extractfeature(face, mfeature);

	Mfeature2Efeature(mfeature, feature);
	return 0;
}
int MFMRECOG::EFaceProcess_getFacefeature(Mat &face, EFeature &efeature)
{
	Mat mfeature(256, 1, CV_32FC1);
	assert(face.type()==CV_8UC1);
	resize(face, face, cvSize(128, 128));
	face.convertTo(face, CV_32FC1, 1.0 / 256.0);
	fe->extractfeature(face, mfeature);

	Mfeature2Efeature(mfeature, efeature);
	return 0;
}


int MFMRECOG::compareFace(Mat &queryface, Mat &refface, double &similarity)
{
	assert((queryface.type()==CV_8UC1)&&(refface.type()==CV_8UC1));
	Mat q(queryface.rows, queryface.cols, CV_8UC1);
	Mat r(refface.rows, refface.cols, CV_8UC1);
	queryface.copyTo(q);
	refface.copyTo(r);
	resize(q, q, cvSize(128, 128));
	q.convertTo(q, CV_32FC1, 1.0 / 256.0);
	
	resize(r, r, cvSize(128, 128));
	r.convertTo(r, CV_32FC1, 1.0 / 256.0);

	Mat qf(256, 1, CV_32FC1);
	Mat rf(256, 1, CV_32FC1);
	fe->extractfeature(q, qf);
	fe->extractfeature(r, rf);
    imshow("q", q);
	imshow("r",r);
	waitKey(0);
 	cout << qf << endl;
	cout << rf << endl;
	cosSimilarity(qf,rf,similarity);

	return 0;
	
}
int MFMRECOG::cosSimilarity(Mat &q, Mat &r, double &similarity)
{
	assert((q.rows==r.rows)&&(q.cols==r.cols));
	double fenzi = q.dot(r);
	double fenmu = sqrt(q.dot(q)) * sqrt(r.dot(r));
	similarity = fenzi/fenmu;

	return 0;
}

int MFMRECOG::ouSimilarity(Mat &q, Mat &r, double &similarity)
{
	assert((q.rows==r.rows)&&(q.cols==r.cols));

	similarity = (q - r).dot(q - r)/((q.dot(q))*(r.dot(r)));
	return 0;
}


int MFMRECOG::Efeature2Mfeature(const EFeature &ef, Mat &mf)
{
	assert(mf.type()==CV_32FC1);
	int N = ef.feature_size;
	float *pt = (float*)ef.feature;

	for (int i = 0; i < N; i++)
	{
		float *val = &pt[i];
		mf.at<float>(i, 0) = (float)(*val);
	}

	return 0;
	
}

int MFMRECOG::Mfeature2Efeature(Mat &mf, EFeature &ef)
{
	assert(ef.feature_size = mf.rows*mf.cols);
	int N = ef.feature_size;
	float *pt = (float *)ef.feature;
	for (int i = 0; i < N; i++)
	{
		float *val = (float *)(&(mf.at<float>(i, 0)));
		pt[i] = *val;
	}
	return 0;
}

int  MFMRECOG::EFaceProcess_Landmark(cv::Mat &gray, cv::Rect &r, EFaceRect& efr)
{
	vector<Point2f> landmarks;
	BBox bbox_ = BBox(r).subBBox(0.1, 0.9, 0.2, 1);
	landmarks = lder.DetectLandmark(gray, bbox_);

	efr.left = r.x;
	efr.top = r.y;
	efr.right = efr.left + r.width;
	efr.bottom = efr.top + r.height;

	efr.lefteye_x = (int)landmarks[0].x;
	efr.lefteye_y = (int)landmarks[0].y;
	efr.righteye_x = (int)landmarks[1].x;
	efr.righteye_y = (int)landmarks[1].y;
	efr.nose_x = (int)landmarks[2].x;
	efr.nose_y = (int)landmarks[2].y;
	efr.centermouth_x = (int)((landmarks[3].x + landmarks[4].x) / 2);
	efr.centermouth_y = (int)((landmarks[3].y + landmarks[4].y) / 2);

	efr.facial_score = 1.0;
	efr.type = 0;

	return 0;
}

int MFMRECOG::EFaceProcess_GetDist(const vector<CFace>& cfaces, Mat &dist)
{
	int Numface = cfaces.size();
	dist = Mat::zeros(Numface, Numface, CV_64FC1);
	for (int i = 0; i < Numface - 1; i++)
	{
		for (int j = i + 1; j < Numface; j++)
		{
			double simij;
			EFaceProcess_FeatureCompare(cfaces[i].facefeature, cfaces[j].facefeature, simij);
			dist.at<double>(i, j) = 1 - simij;
			dist.at<double>(j, i) = 1 - simij;  //相似度越高，距离越小
		}
	}
	return 0;
}

int MFMRECOG::EFaceProcess_CLUST(const std::vector<CFace> &cfaces, std::vector<datapoint> &result)
{
	Mat dist;
	EFaceProcess_GetDist(cfaces,dist);
	fastClust(dist, result);
	return 0;
}