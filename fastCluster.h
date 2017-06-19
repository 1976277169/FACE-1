/*
���پ����㷨
���룺N�������У���������֮��ľ���d(i,j)
�����m������Ĵ��ģ��������������Լ�ÿһ��������������𣨴��ģ�

Author��zhao
Date��2016��1��4�� 20:34:56
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

