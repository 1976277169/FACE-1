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
	//1����ʼ����ʶ��ģ�ͣ����������ļ�
	char buff[256];
	sprintf(buff, "initialize begins");
	logger(buff);

	loadConfig("config.txt");
	string model = Config["model"];
	RECOG *mr = new MFMRECOG();
	mr->E_Init(model.c_str());

	sprintf(buff, "initialize ends");
	logger(buff);

	//2������������ͼƬ������ȡÿ���������������õ�feature�ء���vector<EFeature>��
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

	string classFolder = Config["classFolder"];   //�洢���������ļ�Ŀ¼
	char buff_out[256];
	sprintf(buff_out, "%s", classFolder.c_str());
	FileSystem::makeDir(buff_out);

	int kFeatureSize;
	mr->EFaceProcess_GetFeatureParam(kFeatureSize);//���������С
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
			mr->EFaceProcess_FreeImage(srceimg);//�ͷ�ԭͼ
			continue;
		}
		else
		{
			/*�������*/
			mr->EFaceProcess_Facedetect(*srceimg, list_size, face_lists, 0);
			logger("detecting " + filename + "  face num = " + to_string(list_size));
			for (int j = 0; j < list_size; j++)
			{
				CFace cface;
				cface.srcpath = file;  //1��ͼƬԭ·��
				cface.facerect = face_lists[j];  //2,��������Ϣ
				EFeature feature;
				feature.feature_size = kFeatureSize;
				feature.feature = (float*)malloc(feature.feature_size*sizeof(float)); //���������ڴ�

				/*��ȡ����*/
				logger("extracting face feature");
				try
				{
					mr->EFaceProcess_GetFaceFeature(*srceimg, face_lists[j], feature);
					logger("get feature successfully!");
					cface.facefeature = feature;   //3����������
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
				cface.facepath = buff;  //4,�����洢��·��
				cface.facelabel = -1;   //5��������𣬴�ȷ��
				cface.clustCenter = false;  //6,�Ƿ�Ϊ�������ģ���ȷ��

				cfaces.push_back(cface);
			}
		}
		mr->EFaceProcess_FreeImage(srceimg);//�ͷ�ԭͼ
	}

	//4������fastCluster�����㷨���о��࣬���ÿ��feature��������𣬼�ÿ�����������
	/*��������*/
	vector<datapoint> result;
	mr->EFaceProcess_CLUST(cfaces, result);
	//��������д���ļ�ϵͳ��
	for (int i = 0; i < result.size(); i++)
	{
		Mat face = imread(cfaces[i].facepath.c_str());  //ȡ������
		string dirinclass = classFolder + "/" + to_string(result[i].label);
		if (!FileSystem::isExists(dirinclass))
		{
			FileSystem::makeDir(dirinclass);
		}
		string savepath;
		savepath = dirinclass + "/" + cfaces[i].facepath.substr(cfaces[i].facepath.find_last_of("/") + 1);
		imwrite(savepath, face);
	}

	//�ͷ�cfaces�е�feature�ڴ档
	for (int i = 0; i < cfaces.size(); i++)
	{
		free(cfaces[i].facefeature.feature);
	}
	return 0;
}
