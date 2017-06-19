#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "filesystem.h"
#include "common.h"
#include "utils.h"
#include "fastCluster.h"
#include "demo.h"

using namespace std;
using namespace cv;

int newClust()
{
	google::InitGoogleLogging(" ");
	FLAGS_alsologtostderr = false;
	//1，初始化：识别模型，加载配置文件
	char buff[256];
	sprintf(buff, "initialize begins");
	logger(buff);

	loadConfig("config.txt");
	string model = Config["model"];
	RECOG *mr = new MFMRECOG();
	mr->E_Init(model.c_str());

	sprintf(buff, "initialize ends");
	logger(buff);

	//2，读入待聚类的图片集，提取每张人脸的特征，得到feature池。（vector<EFeature>）
	string massDir = Config["massDir"];
	if (!FileSystem::isExists(massDir))
	{
		sprintf(buff, "imgFolder does not exist!");
		logger(buff);
		return -1;
	}
	vector<string> filenames;
	FileSystem::readDir(massDir, "jpg", filenames);

	string faceFolder = Config["faceFolder"];
	vector<string> facePaths;
	vector<CFace> cfaces;

	string classFolder = Config["classFolder"];   //存储聚类结果的文件目录
	char buff_out[256];
	sprintf(buff_out, "%s", classFolder.c_str());
	FileSystem::makeDir(buff_out);

	int kFeatureSize;
	mr->EFaceProcess_GetFeatureParam(kFeatureSize);//获得特征大小
	for (int i = 0; i < filenames.size(); i++)
	{
		int list_size;
		vector<EFaceRect> face_lists;
		string filename = filenames[i];
		string file = massDir + "/" + filename;

		struct tagEImage t = { NULL, 0, 0, 0, 0 };
		EImage *srceimg = &t;
		mr->EFaceProcess_ReadImage(file, srceimg);
		Mat src = imread(file);
		if (srceimg->width < 50 || srceimg->height < 50)
		{
			mr->EFaceProcess_FreeImage(srceimg);//释放原图
			continue;
		}
		else
		{
			/*检测人脸*/
			mr->EFaceProcess_Facedetect(*srceimg, list_size, face_lists, 0);
			logger("detecting " + filename + "  face num = " + to_string(list_size));
			for (int j = 0; j < list_size; j++)
			{
				CFace cface;
				cface.srcpath = file;  //1，图片原路径
				cface.facerect = face_lists[j];  //2,人脸的信息
				EFeature feature;
				feature.feature_size = kFeatureSize;
				feature.feature = (float*)malloc(feature.feature_size*sizeof(float)); //分配特征内存

				/*提取特征*/
				logger("extracting face feature");
				try
				{
					mr->EFaceProcess_GetFaceFeature(*srceimg, face_lists[j], feature);
					logger("get feature successfully!");
					cface.facefeature = feature;   //3，人脸特征
				}
				catch (...)
				{
					logger("get feature error!");
					continue;
				}
				Mat tmp;
				Rect r1, r2;
				Rect onefacerect;
				Efacerect2Rect(face_lists[j], onefacerect);
				ExpandRect(src, tmp, onefacerect, r1, r2);
				Mat expandface = tmp(r2);

				sprintf(buff, "%s", faceFolder.c_str());
				FileSystem::makeDir(buff);

				sprintf(buff, "%s/%s_%03d.jpg", faceFolder.c_str(), filename.substr(0, max(filename.find_last_of(".jpg"), filename.find_last_of(".JPG")) - 3).c_str(), j + 1);
				facePaths.push_back(buff);
				imwrite(buff, expandface);
				cface.facepath = buff;  //4,人脸存储的路径
				cface.facelabel = -1;   //5，人脸类别，待确定
				cface.clustCenter = false;  //6,是否为聚类中心，待确定

				cfaces.push_back(cface);
			}
		}
		mr->EFaceProcess_FreeImage(srceimg);//释放原图
	}

	//4，调用fastCluster聚类算法进行聚类，输出每个feature所属的类别，即每张人脸的类别。
	/*人脸聚类*/
	vector<datapoint> result;
	mr->EFaceProcess_CLUST(cfaces, result);
	//将聚类结果写入文件系统。
	for (int i = 0; i < result.size(); i++)
	{
		Mat face = imread(cfaces[i].facepath.c_str());  //取出人脸
		string dirinclass = classFolder + "/" + to_string(result[i].label);
		if (!FileSystem::isExists(dirinclass))
		{
			FileSystem::makeDir(dirinclass);
		}
		string savepath;
		savepath = dirinclass + "/" + cfaces[i].facepath.substr(cfaces[i].facepath.find_last_of("/") + 1);
		imwrite(savepath, face);
	}

	//释放cfaces中的feature内存。
	for (int i = 0; i < cfaces.size(); i++)
	{
		free(cfaces[i].facefeature.feature);
	}
	return 0;
}
