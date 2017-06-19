#include <iostream>
#include <cstring>
#include "demo.h"

using namespace std;
int main(int argc, char* argv[])
{
	argc = 2;
	argv[1] = "faceveri";
	if (argc != 2)
	{
		cerr << "need argument" << endl;
		return 0;
	}
	if (strcmp(argv[1], "faceveri") == 0)
	{
		return faceveri();
	}
	else if (strcmp(argv[1], "faceclust") == 0)
	{
		return newClust();
	}
	else
	{
		cerr << "argument is wrong" << endl;
		return 0;
	}
}