#pragma once
#ifndef _SEGMENTATION_
#define _SEGMENTATION_

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


class Segmentation {
public:
	void PreProcessing_ruleBased(Mat inputImg);
	void PreProcessing(Mat inputImg);
};




#endif // !_SEGMENTATION_

