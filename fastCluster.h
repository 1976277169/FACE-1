/*
快速聚类算法
输入：N张人脸中，任意两张之间的距离d(i,j)
输出：m个人物的簇心（中心人脸），以及每一张人脸所属的类别（簇心）

Author：zhao
Date：2016年1月4日 20:34:56
*/
#include <iostream>
#include <vector>
#include <opencv/cxcore.hpp>
#include "RECOG.h"
//#include <armadillo>

struct node
{
	double rho;
	int idx;
};

//struct datapoint
//{
//	int label;
//	bool clustcenter;
//	//int element;
//	//int nc;
//	//int nh;
//};

struct cluster
{
	int classid;
	int centerid;
	int nelement;
	int ncore;
	int nhalo;
	double centerrho;
	std::vector<int> elements;
};

bool comp(node x, node y);
bool comprho(cluster x,cluster y);
void fillval(std::vector<double> &a, double &val);
void fillval(std::vector<int> &a, int &val);
//double compareMatFromMatlab(char* fp, mat& data);
//double compareVecFromMatlab(char* fp, vector<double>& data);
//void Mat2mat(cv::Mat &Matdata, mat &matdata);
//double compareVecIntFromMatlab(char* fp, vector<int>& data);
//void Mat2matInt(cv::Mat &Matdata, mat &matdata);

double getaverNeighrate(const cv::Mat &dist);

double getDc(cv::Mat &dist,double& percent);
void calculateRho(cv::Mat &dist,double &dc,std::vector<double>& rho);
void sortRho(std::vector<double>& rho, std::vector<double>& sorted_rho, 
	std::vector<int>& ordrho);
void calculateDelta(cv::Mat& dist, std::vector<double>& rho, 
	std::vector<double>& sorted_rho, std::vector<int>& ordrho, 
	std::vector<double>& delta, std::vector<int>& nneigh);

void fastClust(cv::Mat &dist, std::vector<datapoint>& clustResult);

