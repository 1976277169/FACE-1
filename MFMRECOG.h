#ifndef MFMRECOG_H__
#define MFMRECOG_H__

#include <opencv2/objdetect/objdetect.hpp>
#include "featureExByCaffe.h"
#include "RECOG.h"
#include "faceAlign.hpp"
#include "caffeFaceVal.h"

class MFMRECOG :public RECOG
{
public:
	//MFMRECOG();
	//~MFMRECOG();

	int E_Init(const char *modulepath) override;
	int E_UnInit() override;
	int EFaceProcess_Facedetect(const EImage& image, int& list_size,
		std::vector<EFaceRect> &face_rect_list, int method = 0) override;
	//EImage* EFaceProcess_Getface(EImage & image, EFaceRect &facerect)override;
	/*
	*获得指定位置处的人脸截图，内存在函数内部分配，需使用FreeImage进行释放
	*/

	/*
	*图片旋转，旋转之后原图，人脸属性发生变化。
	*/
	int EFaceProcess_RotateOneFace(EImage & image,EFaceRect &face_rect_list,EImage *& dstImage,EFaceRect &dst_efr);

	int EFaceProcess_Getface(EImage & image, EFaceRect &facerect, EImage *&face)override;

	int EFaceProcess_GetFeatureParam(int &feature_size) override;
	int EFaceProcess_GetFaceFeature(EImage &image, EFaceRect &facerect,
		EFeature &feature) override;

	int EFaceProcess_ReadImage(const std::string &imgpath, EImage *eimg) override;

	int EFaceProcess_SaveImage(const EImage& eimg, std::string &savepath) override;

	int EFaceProcess_FreeImage(EImage *image) override;

	int EFaceProcess_FeatureCompare(const EFeature& query_feature,
		const EFeature& ref_feature, double& similarity)override;
	
	int EFaceProcess_GetFaceFeature(cv::Mat &face, EFeature &feature) override;
	int EFaceProcess_getFacefeature(cv::Mat &face, EFeature &efeature);
	int compareFace(cv::Mat &queryface, cv::Mat &refface, double &similarity);
	int cosSimilarity(cv::Mat &q, cv::Mat &r, double &similarity);
	int ouSimilarity(cv::Mat &q, cv::Mat &r, double &similarity);
	int Efeature2Mfeature(const EFeature &ef, cv::Mat &mf);
	int Mfeature2Efeature(cv::Mat &mf, EFeature &ef);

	int EFaceProcess_Landmark(cv::Mat &gray, cv::Rect &r, EFaceRect& efr) override;

	int EFaceProcess_GetDist(const std::vector<CFace>& cfaces, cv::Mat &dist)override;

	int EFaceProcess_CLUST(const std::vector<CFace> &cfaces, std::vector<datapoint> &result) override;

public:
	CascadeClassifier kcc;
	CascadeClassifier kccp;
	CaffeFaceValidator *cfv;
	Landmarker lder;
	string protonet;
	string caffemodel;
	featureExer *fe;

};


#endif //MFMRECOG_H__