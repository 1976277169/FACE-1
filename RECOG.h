#ifndef RECOG_H__
#define RECOG_H__
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include "mysqlCPP.h"
/**
* @brief ͼ����Ϣ
*/
typedef struct tagEImage
{
	/** ͼ������ */
	unsigned char *imagedata;
	/** ͼ�����ݴ�С */
	int image_size;
	/** ͼ��� */
	int width;
	/** ͼ��� */
	int height;
	/** ͼ���д�С */
	int widthStep;
}EImage;

/**
* @brief ������Ϣ
*/
typedef struct tagEFaceRect
{
	/** �����ĸ��ٺ� */
	int track_no;

	/** �������ο����Ͻǵ�x���� */
	int left;
	/** �������ο����Ͻǵ�y���� */
	int top;
	/** �������ο����½ǵ�x���� */
	int right;
	/** �������ο����½ǵ�y���� */
	int bottom;
	/** �������۵�x���� */
	int lefteye_x;
	/** �������۵�y���� */
	int lefteye_y;
	/** �������۵�x���� */
	int righteye_x;
	/** �������۵�y���� */
	int righteye_y;
	/** �������ӵ�x���� */
	int nose_x;
	/** �������ӵ�y���� */
	int nose_y;
	/** ������͵�x���� */
	int centermouth_x;
	/** ������͵�y���� */
	int centermouth_y;

	float facial_score;
	int type;
}EFaceRect;

/**
* @brief ����������Ϣ
*/
typedef struct tagEFeature
{
	/** ��������ָ�� */
	void *feature;
	/** ������С */
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
	*��ʼ������ʼ��ϵͳ�����ģ�͵ȡ�
	*/
	virtual int E_Init(const char *modelpath)=0;
	virtual int E_UnInit() = 0;

	/**
	* ͼƬ�������
	* @param [in] image  �����ͼƬ
	* @param [out] list_size ��������
	* @param [out] face_rect_list ����λ���б�
	* @param [in] method ������ⷽ����0������⣬1������⣩
	* @return �����Ƿ�ɹ�
	*  0 ��ʾ�ɹ�
	*  -1 ��ʾʧ��
	*/
	virtual int EFaceProcess_Facedetect(const EImage& image, int& list_size,
		std::vector<EFaceRect> &face_rect_list, int method = 0) = 0;
	
	/*
	*���������򣬶�ͼƬ�е��������йؼ����ע��
	*/
	virtual int EFaceProcess_Landmark(cv::Mat &gray, cv::Rect &r, EFaceRect& efr) = 0;

	/*
	*���ͼƬ�е�ĳ�������ı�׼�������������������ͱ�׼���Ĺ��̡�
	*���������ڴ��ں����ڷ��䣬����֮��Ҫʹ��EFaceProcess_FreeImage�ͷ�face���ڴ档
	*/
	virtual int EFaceProcess_Getface(EImage & image, EFaceRect &facerect, EImage *&face) = 0;

	/*
	*�������������ά��
	*/
	virtual int EFaceProcess_GetFeatureParam(int &feature_size) = 0;

	
	/*
	*��ȡԭͼ��ĳ�������������������ȷ���������ڴ档
	*/
	virtual int EFaceProcess_GetFaceFeature(EImage &image, EFaceRect &facerect,EFeature &feature) = 0;

	/*
	*���Mat����ͼ�����������Ҫ���ȷ�����������ڴ档
	*face�����ǻҶ�ͼƬ
	*/
	virtual int EFaceProcess_GetFaceFeature(cv::Mat &face, EFeature &feature) = 0;

	///**
	//* �����ȶ� �Ƚ��������������ƶ�
	//* @param [in] query_feature ����1��Ϣ
	//* @param [in] ref_feature ����2��Ϣ
	//* @param [out] similarity���ƶ�
	//* @return �����Ƿ�ɹ�
	//*  0 ��ʾ�ɹ�
	//*  -1 ��ʾʧ��
	//*/
	virtual int EFaceProcess_FeatureCompare(const EFeature& query_feature,
		const EFeature& ref_feature, double& similarity) = 0;

	virtual int EFaceProcess_ReadImage(const std::string &imgpath, EImage *eimg) = 0;

	virtual int EFaceProcess_SaveImage(const EImage& eimg, std::string &savepath) = 0;
	/*
	*�ͷ�ͼƬ���ڴ�ռ�
	*/
	virtual int EFaceProcess_FreeImage(EImage *image) = 0;

	virtual int EFaceProcess_GetDist(const std::vector<CFace>& cfaces, cv::Mat &dist) = 0;

	/*
	*���������ؼ�������֮��ľ������
	*/	
	virtual int EFaceProcess_CLUST(const std::vector<CFace> &cfaces, std::vector<datapoint> &result) = 0;
public:

};

#endif //RECOG_H__