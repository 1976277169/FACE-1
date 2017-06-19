//#include <cassert>
//#include <caffe/caffe.hpp>
//#include <boost/algorithm/string.hpp>
//#include "featureExByCaffe.h"
//#include "filesystem.h"
//#include <iostream>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include "RECOG.h"
//#include "MFMRECOG.h"
//#include "utils.h"
//#include "demo.h"
//
//using namespace cv;
//using namespace std;
//using namespace caffe;
//using namespace boost::algorithm;
//int faceveri()
//{
//	google::InitGoogleLogging(" ");
//	FLAGS_alsologtostderr = false;
//	RECOG *r = new MFMRECOG();
//	string m = "../Models";
//	r->E_Init(m.c_str());
//	//读取pos_pair和neg_pair
//	ifstream pos_pair("pos.txt",ios::in);
//	ifstream neg_pair("neg.txt",ios::in);
//	vector<array<int, 2>> pos;
//	vector<array<int, 2>> neg;
//	if ((pos_pair.is_open())&&(neg_pair.is_open()))
//	{
//		string posline,negline;
//		while (getline(pos_pair, posline))
//		{
//			int I1,I2;
//			vector<string> linevector;
//			boost::algorithm::split(linevector, posline, is_any_of(" "));
//			assert(linevector.size() == 2);
//			I1 = atoi(linevector[0].c_str());
//			I2 = atoi(linevector[1].c_str());
//			array<int, 2> postmp;
//			postmp.at(0) = I1;
//			postmp.at(1) = I2;
//			pos.push_back(postmp);
//		}
//
//		while (getline(neg_pair, negline))
//		{
//			int E1, E2;
//			vector<string> linevector;
//			boost::algorithm::split(linevector, negline, is_any_of(" "));
//			assert(linevector.size() == 2);
//			E1 = atoi(linevector[0].c_str());
//			E2 = atoi(linevector[1].c_str());
//			array<int, 2> negtmp;
//			negtmp.at(0) = E1;
//			negtmp.at(1) = E2;
//			neg.push_back(negtmp);
//		}
//		assert((pos.size() == 3000) && (neg.size() == 3000));
//	}
//	else
//	{
//		cerr << "cannot load pos_pair.txt or neg_pair.txt" << endl;
//		return -1;
//	}
//
//	pos_pair.close();
//	neg_pair.close();
//
//	//读取lfw_path全集
//	ifstream lfwpath("lfw_path.txt",ios::in);
//	vector<string> lfwimg;
//	vector<string> lfwperson;
//	if (lfwpath.is_open())
//	{
//		string line;
//		while (getline(lfwpath, line))
//		{
//			string picname;
//			vector<string> linevector;
//			boost::algorithm::split(linevector, line, is_any_of("/"));
//			picname = linevector[linevector.size()-1];
//			lfwimg.push_back(picname);
//			lfwperson.push_back(linevector[linevector.size() - 2]);
//		}
//	
//	}
//	else
//	{
//		cerr << "cannot find lfw_path.txt" << endl;
//		return -1;
//	}
//	lfwpath.close();
//	int num = 3000;  //只各取num对进行测试
//	vector<array<string, 2>> intraTest(num);
//	vector<array<string, 2>> extraTest(num);
//	vector<array<string, 2>> intrap(num);
//	vector<array<string, 2>> extrap(num);
//	for (int i = 0; i < num; i++)
//	{
//		intraTest[i].at(0) = lfwimg[pos[i].at(0) - 1];
//		intraTest[i].at(1) = lfwimg[pos[i].at(1) - 1];
//		intrap[i].at(0) = lfwperson[pos[i].at(0) - 1];
//		intrap[i].at(1) = lfwperson[pos[i].at(1) - 1];
//
//		extraTest[i].at(0) = lfwimg[neg[i].at(0) - 1];
//		extraTest[i].at(1) = lfwimg[neg[i].at(1) - 1];
//		extrap[i].at(0) = lfwperson[neg[i].at(0) - 1];
//		extrap[i].at(1) = lfwperson[neg[i].at(1) - 1];
//	}
//
//	int right_intra = 0;
//	int right_extra = 0;
//	ofstream itestsresult("iPCA_cos.txt");
//	ofstream etestsresult("ePCA_cos.txt");
//	int Inum = 0;
//	int Enum = 0;
//	//存放一对照片的地址
//	string dir = "../lfw-deepfunneled"; //lfw全集，经过旋正的图集
//	cv::Mat Apic, Bpic, Cpic, Dpic;
//	//下面读入图片，进行检脸+标点+提特征+比对
//
//	for (int p = 0; p < num; p++)
//	{
//		string Apicpath = dir + "/" + intrap[p].at(0) + "/" + intraTest[p].at(0);
//		string Bpicpath = dir + "/" + intrap[p].at(1) + "/" + intraTest[p].at(1);
//
//		string Cpicpath = dir + "/" + extrap[p].at(0) + "/" + extraTest[p].at(0);
//		string Dpicpath = dir + "/" + extrap[p].at(1) + "/" + extraTest[p].at(1);
//		string AAA = "a.jpg";
//		if ((!FileSystem::isExists(Apicpath)) || (!FileSystem::isExists(Bpicpath)) ||
//			(!FileSystem::isExists(Cpicpath)) || (!FileSystem::isExists(Dpicpath)))
//		{
//			cout << "file not found" << endl;
//			return -1;
//		}
//		Apic = imread(Apicpath, 0);
//		Bpic = imread(Bpicpath, 0);
//		Cpic = imread(Cpicpath, 0);
//		Dpic = imread(Dpicpath, 0);
//
//		Rect facerect(68,68,114,114);
//		//showimg(Apic);
//		struct tagEImage tmpA = { NULL, 0, 0, 0, 0 };
//		struct tagEImage tmpB = { NULL, 0, 0, 0, 0 };
//		struct tagEImage tmpC = { NULL, 0, 0, 0, 0 };
//		struct tagEImage tmpD = { NULL, 0, 0, 0, 0 };
//		EImage *eA = &tmpA;
//		EImage *eB = &tmpB;
//		EImage *eC = &tmpC;
//		EImage *eD = &tmpD;
//		r->EFaceProcess_ReadImage(Apicpath, eA);
//		r->EFaceProcess_ReadImage(Bpicpath, eB);
//		r->EFaceProcess_ReadImage(Cpicpath, eC);
//		r->EFaceProcess_ReadImage(Dpicpath, eD);
//		assert(eA->imagedata != NULL);
//		int listsizeA, listsizeB, listsizeC, listsizeD;
//		vector<EFaceRect> frlA, frlB, frlC, frlD;
//		r->EFaceProcess_Facedetect(*eA, listsizeA, frlA, 0);
//		r->EFaceProcess_Facedetect(*eB, listsizeB, frlB, 0);
//		r->EFaceProcess_Facedetect(*eC, listsizeC, frlC, 0);
//		r->EFaceProcess_Facedetect(*eD, listsizeD, frlD, 0);
//
//		EFeature Afeature;
//		Afeature.feature = (float*)malloc(256 * sizeof(float)); //分配特征内存
//		EFeature Bfeature;
//		Bfeature.feature = (float*)malloc(256 * sizeof(float)); //分配特征内存
//
//		EFaceRect Aefr, Befr, Cefr, Defr;
//		bool Anorm = false;
//		bool Bnorm = false;
//		bool Cnorm = false;
//		bool Dnorm = false;
//		if ((listsizeA == 1) && (frlA[0].left <= 80) && (frlA[0].left >= 40) && (frlA[0].bottom >= 160) && (frlA[0].bottom <= 220)&&(frlA[0].top<=80)&&(frlA[0].top>=40)&&(frlA[0].right>160)&&(frlA[0].right<=220))
//		{
//			Anorm = true;
//		}
//		if ((listsizeB == 1) && (frlB[0].left <= 80) && (frlB[0].left >= 40) && (frlB[0].bottom >= 160) && (frlB[0].bottom <= 220) && (frlB[0].top <= 80) && (frlB[0].top >= 40) && (frlB[0].right>160) && (frlB[0].right <= 220))
//		{
//			Bnorm = true;
//		}
//		if ((listsizeC == 1) && (frlC[0].left <= 80) && (frlC[0].left >= 40) && (frlC[0].bottom >= 160) && (frlC[0].bottom <= 220) && (frlC[0].top <= 80) && (frlC[0].top >= 40) && (frlC[0].right>160) && (frlC[0].right <= 220))
//		{
//			Cnorm = true;
//		}
//		if ((listsizeD == 1) && (frlD[0].left <= 80) && (frlD[0].left >= 40) && (frlD[0].bottom >= 160) && (frlD[0].bottom <= 220) && (frlD[0].top <= 80) && (frlD[0].top >= 40) && (frlD[0].right>160) && (frlD[0].right <= 220))
//		{
//			Dnorm = true;
//		}
//
//		if (!Anorm)
//		{
//			r->EFaceProcess_Landmark(Apic, facerect, Aefr);
//			r->EFaceProcess_GetFaceFeature(*eA, Aefr, Afeature);
//			r->EFaceProcess_FreeImage(eA);
//		}
//		else
//		{
//			//double start = cv::getTickCount();
//			//for (int i = 0; i < 1000; i++)
//			//{
//			//	r->EFaceProcess_GetFaceFeature(*eA, frlA[0], Afeature);
//			//}
//			//double extractf_cost = (cv::getTickCount() - start) / cv::getTickFrequency();
//			r->EFaceProcess_GetFaceFeature(*eA, frlA[0], Afeature);
//			r->EFaceProcess_FreeImage(eA);
//		}
//
//		if (!Bnorm)
//		{
//			r->EFaceProcess_Landmark(Bpic, facerect, Befr);
//			r->EFaceProcess_GetFaceFeature(*eB, Befr, Bfeature);
//			r->EFaceProcess_FreeImage(eB);
//		}
//		else
//		{
//			r->EFaceProcess_GetFaceFeature(*eB, frlB[0], Bfeature);
//			r->EFaceProcess_FreeImage(eB);
//		}
//
//		double simAB = 0.0;
//		r->EFaceProcess_FeatureCompare(Afeature, Bfeature, simAB);
//		itestsresult << simAB << "\n";
//		if (simAB > 0.299)
//		{
//			right_intra++;
//		}
//		Inum++;
//		free(Afeature.feature);
//		free(Bfeature.feature);
//
//		EFeature Cfeature;
//		Cfeature.feature = (float*)malloc(256 * sizeof(float)); //分配特征内存
//		EFeature Dfeature;
//		Dfeature.feature = (float*)malloc(256 * sizeof(float)); //分配特征内存
//
//		if (!Cnorm)
//		{
//			r->EFaceProcess_Landmark(Cpic, facerect, Cefr);
//			r->EFaceProcess_GetFaceFeature(*eC, Cefr, Cfeature);
//			r->EFaceProcess_FreeImage(eC);
//		}
//		else
//		{
//			r->EFaceProcess_GetFaceFeature(*eC, frlC[0], Cfeature);
//			r->EFaceProcess_FreeImage(eC);
//		}
//
//		if (!Dnorm)
//		{
//			r->EFaceProcess_Landmark(Dpic, facerect, Defr);
//			r->EFaceProcess_GetFaceFeature(*eD, Defr, Dfeature);
//			r->EFaceProcess_FreeImage(eD);
//		}
//		else
//		{
//			r->EFaceProcess_GetFaceFeature(*eD, frlD[0], Dfeature);
//			r->EFaceProcess_FreeImage(eD);
//		}
//		double simCD = 0.0;
//		r->EFaceProcess_FeatureCompare(Cfeature, Dfeature, simCD);
//		free(Cfeature.feature);
//		free(Dfeature.feature);
//		etestsresult << simCD << "\n";
//		if (simCD < 0.299)
//		{
//			right_extra++;
//		}
//		Enum++;
//
//		if (p % 200 == 100)
//		{
//			cout << "verified " << p << " positive pairs and " << p << " negtive pairs. ";
//			cout << "TPR : " << 100 * right_intra / (double)Inum << " %  ." ;
//			cout << "FAR : " << 100 * (Enum - right_extra) / (double)Enum << " %" << endl;
//		}
//	}
//	
//	itestsresult.close();
//	etestsresult.close();
//	cout << "verified over ";
//	cout << "TPR : " << 100 * right_intra / (double)Inum << " %  .";
//	cout << "FAR : " << 100 * (Enum - right_extra) / (double)Enum << " % ." << endl;
//	cout << "average : "<<(right_intra + right_extra) / (double)(Inum + Enum) << endl;
//
//	//根据intra和extra两个文档画出曲线
//	ifstream intra_r("iPCA_cos.txt", ios::in);
//	ifstream extra_r("ePCA_cos.txt", ios::in);
//	vector<double> sim_In, sim_Ex;
//	if ((intra_r.is_open()) && (extra_r.is_open()))
//	{
//		string line_in, line_ex;
//		while (getline(intra_r, line_in) && getline(extra_r, line_ex))
//		{
//			double sim_in = atof(line_in.c_str());
//			double sim_ex = atof(line_ex.c_str());
//			sim_In.push_back(sim_in);
//			sim_Ex.push_back(sim_ex);
//		}
//	}
//	intra_r.close();
//	extra_r.close();
//	assert(sim_In.size() == sim_Ex.size());
//	int NN = sim_In.size();
//	ofstream accurate("accurate.txt");
//	accurate << "th   par   nrr   aver" << endl;
//
//	for (int i = 0; i < 1000; i++)
//	{
//		int pa = 0;
//		int nr = 0;
//		double th = i / 1000.0;
//		for (int m = 0; m < NN; m++)
//		{
//			if (sim_In[m]>th)
//			{
//				pa++;
//			}
//			if (sim_Ex[m] < th)
//			{
//				nr++;
//			}
//		}
//
//		//统计当前th下的准确率
//		double par = pa / (double)(NN);
//		double nrr = nr / (double)(NN);
//		double aver = (par + nrr) / 2;
//		accurate << th << "  " << par << "  " << nrr << "  " << aver << endl;
//	}
//	accurate.close();
//	return 0;
//
//}
