// LungSegmentation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include<opencv2/opencv.hpp>

#include "Segmentation.h"

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
	Segmentation segmentation;
	
	std::cout << "Hello World!\n";
	//load image
	cout << "loading lung images" << endl;

	Mat inputImg = imread(argv[1]);
	if (!inputImg.data) {
		cerr << "Cannot read input image" << endl;
		exit(-1);
	}

	
	//imshow("Image1", inputImg);

	segmentation.PreProcessing_ruleBased(inputImg);

	waitKey(0);
	return 0;
}


