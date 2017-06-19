#ifndef RECOG_H__
#define RECOG_H__
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include "mysqlCPP.h"
/**
* @brief 图像信息
*/
typedef struct tagEImage
{
	/** 图像数据 */
	unsigned char *imagedata;
	/** 图像数据大小 */
	int image_size;
	/** 图像宽 */
	int width;
	/** 图像高 */
	int height;
	/** 图像行大小 */
	int widthStep;
}EImage;

/**
* @brief 人脸信息
*/
typedef struct tagEFaceRect
{
	/** 人脸的跟踪号 */
	int track_no;

	/** 人脸矩形框左上角的x坐标 */
	int left;
	/** 人脸矩形框左上角的y坐标 */
	int top;
	/** 人脸矩形框右下角的x坐标 */
	int right;
	/** 人脸矩形框右下角的y坐标 */
	int bottom;
	/** 人脸左眼的x坐标 */
	int lefteye_x;
	/** 人脸左眼的y坐标 */
	int lefteye_y;
	/** 人脸右眼的x坐标 */
	int righteye_x;
	/** 人脸右眼的y坐标 */
	int righteye_y;
	/** 人脸鼻子的x坐标 */
	int nose_x;
	/** 人脸鼻子的y坐标 */
	int nose_y;
	/** 人脸嘴巴的x坐标 */
	int centermouth_x;
	/** 人脸嘴巴的y坐标 */
	int centermouth_y;

	float facial_score;
	int type;
}EFaceRect;

/**
* @brief 人脸特征信息
*/
typedef struct tagEFeature
{
	/** 人脸特征指针 */
	void *feature;
	/** 特征大小 */
	int feature_size;
}EFeature;

struct CFace
{
	std::string srcpath;
	EFaceRect facerect;
	EFeature facefeature;
	std::string facepath;
	int facelabel;
	bool clustCenter;
};
struct datapoint
{
	int label;
	bool clustcenter;
};

class RECOG
{
public:
	RECOG()=default;
	~RECOG()=default;

	/*
	*初始化，初始化系统所需的模型等。
	*/
	virtual int E_Init(const char *modelpath)=0;
	virtual int E_UnInit() = 0;

	/**
	* 图片人脸检测
	* @param [in] image  待检测图片
	* @param [out] list_size 人脸个数
	* @param [out] face_rect_list 人脸位置列表
	* @param [in] method 人脸检测方法（0正脸检测，1侧脸检测）
	* @return 返回是否成功
	*  0 表示成功
	*  -1 表示失败
	*/
	virtual int EFaceProcess_Facedetect(const EImage& image, int& list_size,
		std::vector<EFaceRect> &face_rect_list, int method = 0) = 0;
	
	/*
	*依据人脸框，对图片中的人脸进行关键点标注。
	*/
	virtual int EFaceProcess_Landmark(cv::Mat &gray, cv::Rect &r, EFaceRect& efr) = 0;

	/*
	*获得图片中的某张人脸的标准化人脸，经历了旋正和标准化的过程。
	*该人脸的内存在函数内分配，调用之后要使用EFaceProcess_FreeImage释放face的内存。
	*/
	virtual int EFaceProcess_Getface(EImage & image, EFaceRect &facerect, EImage *&face) = 0;

	/*
	*获得人脸特征的维度
	*/
	virtual int EFaceProcess_GetFeatureParam(int &feature_size) = 0;

	
	/*
	*获取原图中某个人脸的特征，需事先分配好特征内存。
	*/
	virtual int EFaceProcess_GetFaceFeature(EImage &image, EFaceRect &facerect,EFeature &feature) = 0;

	/*
	*获得Mat类型图像的特征，需要事先分配好特征的内存。
	*face必须是灰度图片
	*/
	virtual int EFaceProcess_GetFaceFeature(cv::Mat &face, EFeature &feature) = 0;

	///**
	//* 特征比对 比较两个特征的相似度
	//* @param [in] query_feature 特征1信息
	//* @param [in] ref_feature 特征2信息
	//* @param [out] similarity相似度
	//* @return 返回是否成功
	//*  0 表示成功
	//*  -1 表示失败
	//*/
	virtual int EFaceProcess_FeatureCompare(const EFeature& query_feature,
		const EFeature& ref_feature, double& similarity) = 0;

	virtual int EFaceProcess_ReadImage(const std::string &imgpath, EImage *eimg) = 0;

	virtual int EFaceProcess_SaveImage(const EImage& eimg, std::string &savepath) = 0;
	/*
	*释放图片的内存空间
	*/
	virtual int EFaceProcess_FreeImage(EImage *image) = 0;

	virtual int EFaceProcess_GetDist(const std::vector<CFace>& cfaces, cv::Mat &dist) = 0;

	/*
	*根据特征池计算两两之间的距离矩阵
	*/	
	virtual int EFaceProcess_CLUST(const std::vector<CFace> &cfaces, std::vector<datapoint> &result) = 0;
public:

};

#endif //RECOG_H__