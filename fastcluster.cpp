#include <iostream>
#include <fstream>
#include "fastCluster.h"

using namespace std;
using namespace cv;
//using namespace arma;
bool comp(node x, node y)
{
	return x.rho > y.rho;
}
bool comprho(cluster x, cluster y)
{
	return x.centerrho > y.centerrho;
}
void fillval(vector<int> &a, int &val)
{
	for (int i = 0; i < a.size(); i++)
	{
		a[i] = val;
	}
}

void fillval(vector<double> &a, double &val)
{
	for (int i = 0; i < a.size(); i++)
	{
		a[i] = val;
	}
}

double getaverNeighrate(const Mat &dist)
{
	int N = dist.rows;
	int nneigh = 0;
	for (int i = 0; i < N-1; i++)
	{
		for (int j = i + 1; j < N; j++)
		{
			if (dist.at<double>(i, j) < 0.4)
			{
				nneigh++;
			}
		}
	}
	double percents = 100.0*nneigh / (double)(N*(N - 1) / 2);
	return percents;
}

double getDc(cv::Mat &dist, double& percent)
{
	int Num = dist.rows;
	int N = Num*(Num - 1) / 2;  //���о��������
	if ((((int)percent) >= 100) || (((int)percent) <= 0))
	{
		cerr << "the percent must be [1-100],2-5 maybe ok." << endl;
	}
	cout << "average percentage of neighbours(hard code) :  " << percent << "% ." << endl;

	//��ýضϲ���dc,�Ƽ�ֵ��ʹ��ƽ��ÿ������ھ���Ϊ����������1%-2%
	int position = (int)(N*percent / 100);
	cv::Mat alldis(1, N, CV_64FC1);

	for (int i = 0; i < Num - 1; i++)
	{
		for (int j = i + 1; j < Num; j++)
		{
			alldis.at<double>(0, (1 + 2 * (Num - 1) - i)*i / 2 + (j - i) - 1) = dist.at<double>(i, j);
		}
	}
	vector<double> sda = (vector<double>)alldis;
	sort(sda.begin(), sda.end());  //��������
	double dc = sda[position];
	cout << "Computing Rho with gaussian kernal of radius(dc) : " << dc << endl;
	return dc;
}

void calculateRho(cv::Mat &dist, double &dc, std::vector<double>& rho)
{
	int Num = dist.rows;
	assert(rho.size() == Num);
	double bdouble = 0.0;
	fillval(rho, bdouble);
	//Gaussian kernal
	for (int i = 0; i < Num - 1; i++)
	{
		for (int j = i + 1; j < Num; j++)
		{
			double distmp = dist.at<double>(i, j);
			rho[i] = rho[i] + exp(-pow(distmp / dc, 2));
			rho[j] = rho[j] + exp(-pow(distmp / dc, 2));
		}
	}  
}

void sortRho(vector<double>& rho, vector<double>& sorted_rho, vector<int>& ordrho)
{
	assert((sorted_rho.size() == rho.size()) && (ordrho.size()==rho.size()));
	//���ֲ��ܶ�rho[Num] ���ս������У��������±�ŵĴ���
	vector<node> _rho(rho.size());
	for (int i = 0; i < rho.size(); i++)
	{
		_rho[i].rho = rho[i];
		_rho[i].idx = i;
	}
	sort(_rho.begin(), _rho.end(), comp);//comp����ָ�������ǽ���

	for (int i = 0; i < _rho.size(); i++)
	{
		sorted_rho[i] = _rho[i].rho;
		ordrho[i] = _rho[i].idx;
	}
}
void calculateDelta(cv::Mat& dist, vector<double>& rho, vector<double>& sorted_rho, vector<int>& ordrho, vector<double>& delta, vector<int>& nneigh)
{
	assert((delta.size() == rho.size()) && (nneigh.size() == rho.size()));
	delta[ordrho[0]] = -1.0;
	nneigh[ordrho[0]] = 0;
	double maxd, mind;
	minMaxIdx(dist, &mind, &maxd);  //ȡ�����ľ���ֵ maxd.
	for (int ii = 1; ii < rho.size(); ii++)
	{
		delta[ordrho[ii]] = maxd;
		for (int jj = 0; jj < ii; jj++)
		{
			//cout << "dist.at<double>(ordrho["<<ii<<"], ordrho["<<jj<<"]) =dist["<<ordrho[ii]<<","<<ordrho[jj]<<"] = "<<dist.at<double>(ordrho[ii], ordrho[jj]) << endl;
			if (dist.at<double>(ordrho[ii], ordrho[jj]) < delta[ordrho[ii]])
			{
				delta[ordrho[ii]] = dist.at<double>(ordrho[ii], ordrho[jj]);
				nneigh[ordrho[ii]] = ordrho[jj];
			}
		}
	}

	cv::Mat delta_Mat = cv::Mat(delta);   //������delta��ֵ��Mat����ȡ���ֵ��
	double mindel, maxdel;
	minMaxIdx(delta_Mat, &mindel, &maxdel);
	delta[ordrho[0]] = maxdel;  //delta[Num]��ֵ���

	/*��rho[Num]  �� delta[Num] ����gamma[Num].*/
	vector<int> ind(rho.size());
	vector<double> gamma(rho.size());
	for (int i = 0; i < rho.size(); i++)
	{
		ind[i] = i;
		gamma[i] = rho[i] * delta[i];
	}
}
void fastClust(cv::Mat &dist, vector<datapoint>& clustResult)
{
	assert(dist.rows == dist.cols);
	int Num = dist.rows;
	double percent = getaverNeighrate(dist); //ָ��ƽ���ھ����İٷֱ�

  	double dc = getDc(dist, percent);

	vector<double> rho(Num);
	calculateRho(dist,dc,rho);

	vector<double> rho_sorted(Num);
	vector<int> ordrho(Num);
	sortRho(rho, rho_sorted, ordrho);   //��������

	vector<double> delta(Num);
	vector<int> nneigh(Num);
	calculateDelta(dist,rho,rho_sorted,ordrho,delta,nneigh);

	vector<double> delta_sort(Num);
	delta_sort = delta;
	sort(delta_sort.begin(), delta_sort.end());
	/*��ʼ����*/
	double maxdel = delta[ordrho[0]];
	double rhomin = rho_sorted[rho_sorted.size()*max((100-percent*3),30.0)/100] ;  //ָ��rhomin����deltamin
	double deltamin = delta_sort[ordrho.size()*(min(percent * 4, 30.0)) / 100];
	int NCLUST = 0;
	vector<int> cl(Num);  //��ʼ��cl[Num]��ȫΪ-1
	int bint = -1;
	fillval(cl, bint);
	vector<int> icl;
	for (int i = 0; i < Num; i++)
	{
		if ((rho[i]>rhomin) && (delta[i] > deltamin))
		{
			cl[i] = NCLUST;  //��i�������������NCLUST
			icl.push_back(i);  //��NCLUST����������������i.
			NCLUST++;
		}
	}
	//���ҳ����еľ������ġ�
	cout << "NUMBER OF CLUSTERS : " << NCLUST << endl;

	/*��ʼ������������ķ��䣬�������е�����*/
	cout << "Performing assignation" << endl;
	for (int i = 0; i < Num; i++)
	{
		if (cl[ordrho[i]] == -1)
		{
			//����Ѱ�Ҿ�������ʱ��ֻ�д����ĵ�cl�Ų���-1.��ÿһ���Ǵ����ĵ�����
			//�����Ϊ����nneigh����𣬼�ĳ�������ĵ����
			cl[ordrho[i]] = cl[nneigh[ordrho[i]]];
		}
	}
	/*halo��������������֮�����µĲ��ּ�Ϊ��nneigh����Щ���ݵ㡪������*/
	vector<int> halo(Num);
	for (int i = 0; i < Num; i++)
	{
		halo[i] = cl[i];
	}

	vector<double> bord_rho(NCLUST);
	if (NCLUST > 0)
	{
		for (int i = 0; i < NCLUST; i++)
		{
			bord_rho[i] = 0;
		}

		for (int i = 0; i < Num - 1; i++)
		{
			for (int j = i + 1; j < Num; j++)
			{
				if ((cl[i] != cl[j]) && (dist.at<double>(i, j) <= dc))
				{
					//i��j�����ͬ����������ľ���С�ڽضϾ��롣
					double rho_aver = (rho[i] + rho[j]) / 2.0;
					if (rho_aver>bord_rho[cl[i]])
					{
						//i��������bord_rho��ƽ���ֲ��ܶ�С,��bord_rho���ŵ�rho_aver.
						bord_rho[cl[i]] = rho_aver;
					}
					if (rho_aver>bord_rho[cl[j]])
					{
						//ͬ��������j��������bord_rho.
						bord_rho[cl[j]] = rho_aver;
					}
				}
			}
		}

		for (int i = 0; i < Num; i++)
		{
			if (rho[i] < bord_rho[cl[i]])
			{
				halo[i] = -1;  //��Ⱥ�� -1
			}
		}
	}

	//����ÿ������nc��nh
	vector<cluster> allclust(NCLUST);
	vector<int> nc(NCLUST);
	vector<int> nh(NCLUST);
	int ling = 0;
	fillval(nc, ling);
	fillval(nh, ling);
	for (int i = 0; i < NCLUST; i++)
	{
		int ncc = 0;
		int nhh = 0;
		for (int j = 0; j < Num; j++)
		{
			//ͳ��ÿ��������������������halo������
			if (cl[j] == i)
			{
				ncc++;
			}
			if (halo[j] == i)
			{
				nhh++;
			}
		}
		nc[i] = ncc;
		nh[i] = nhh;
		cout << "CLUSTER : " << i << "  CENTER: " << icl[i] << " ELEMENT: " << nc[i]
			<< "  CORE: " << nh[i] << "  HALO: " << nc[i] - nh[i] << endl;

	}

	for (int i = 0; i < NCLUST; i++)
	{
		allclust[i].classid = i;
		allclust[i].centerid = icl[i];
		allclust[i].nelement = nc[i];
		allclust[i].ncore = nh[i];
		allclust[i].nhalo = nc[i] - nh[i];
		allclust[i].centerrho = rho[icl[i]];
	}
	for (int i = 0; i < Num; i++)
	{
		int nclass = cl[i];
		allclust[nclass].elements.push_back(i);
	}
	//�Ծ��������ж��ι��࣬����rho���н�������
	//vector<cluster> newallclust = allclust;
	//sort(newallclust.begin(),newallclust.end(),comprho);
	//sort(allclust.begin(), allclust.end(), comprho);
	//�ҳ����е���ͼƬ �ࡣM*N����
	ofstream MNsimi("MNsimi.txt");
	vector<cluster> onepicclust;
	for (int i = 0; i < NCLUST; i++)
	{
		if (allclust[i].nelement == 1)
		{
			onepicclust.push_back(allclust[i]);
		}
	}
	int Monepiccluster = onepicclust.size();
	cout << " �������" << Monepiccluster << endl;
	for (int i = 0; i < Monepiccluster; i++)
	{
		int maxid = -1;
		double maxsimi = 0.0;
		int index = onepicclust[i].classid;
		//int start = onepicclust[i].centerid;
		for (int j = 0; j < NCLUST; j++)  //�벻�ǵ���i�������������Ƚ�
		{
			double dist_mn = 1 - dist.at<double>(allclust[index].centerid, allclust[j].centerid);
			MNsimi << dist_mn << " ";

			if ((dist_mn > maxsimi) && (dist_mn<0.9))
			{
				maxsimi = dist_mn;
				maxid = j;
			}
		}

		if (maxsimi > 0.55)
		{
			//maxsimi����0.5����Ϊ��Ҫ�ϲ�
			for (int n = 0; n < allclust[index].elements.size(); n++)
			{
				allclust[maxid].nhalo += 1;
				allclust[maxid].elements.push_back(allclust[index].elements[n]);
				cl[allclust[index].elements[n]] = cl[allclust[maxid].centerid];
			}
			//allclust[maxid].nhalo += 1;
			//cl[onepicclust[i].centerid] = cl[allclust[maxid].centerid];
			cout << "���� " << allclust[index].classid << "�������� " << allclust[index].centerid
				<< "���� " << allclust[index].elements.size()<<" ������"
				<< "����� " << maxid << "(���� " << allclust[index].elements.size()<<"������)"
				<< "�ϲ� ������������ " << allclust[maxid].centerid << endl;
			icl[allclust[index].classid] = icl[allclust[maxid].classid];
			allclust[index].centerid = allclust[maxid].centerid;  //�ı���������������id

		}
		MNsimi << "\n";
	}
	MNsimi.close();

	ofstream dist_center_ij("dist_center_ij.txt");

	//dist_center_ij << "dc = " << dc << "\n";
	for (int i = 0; i < NCLUST; i++)
	{
		for (int j = 0; j < NCLUST; j++)
		{
			double dist_clust_ij = 1 - dist.at<double>(allclust[i].centerid, allclust[j].centerid);
			dist_center_ij << dist_clust_ij;
			dist_center_ij << "  ";
		}
		dist_center_ij << "\n";
	}
	dist_center_ij.close();

	clustResult.clear();
	//vector<datapoint> clustresult(Num);
	for (int i = 0; i < Num; i++)
	{
		datapoint cresult;
		cresult.label = cl[i];
		cresult.clustcenter = false;
		clustResult.push_back(cresult);
	}
	for (int i = 0; i < NCLUST; i++)
	{
		int centerID = icl[i];
		clustResult[centerID].clustcenter = true;
	}

}