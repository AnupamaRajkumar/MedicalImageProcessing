#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "Segmentation.h"

using namespace std;
using namespace cv;

RNG rng(12345);

void Segmentation::PreProcessing_ruleBased(Mat inputImg) {

	/*step 1: Resixe the image to 512 x 512*/
	Mat dst = Mat::zeros(512, 512, inputImg.type()); 
	resize(inputImg, dst, dst.size(), 0, 0);
	imshow("resized", dst);

	/*convert image to single channel*/
	if (inputImg.channels() != 1) {
		
		cvtColor(dst, dst, COLOR_BGR2GRAY);
	}

	/*step 2: Filtering to smooth out noise*/
	int kSize = 3;
	Mat kernel = Mat(kSize, kSize, dst.type());				
	Mat filtImg = Mat::zeros(dst.size(), dst.type());				
	GaussianBlur(dst, filtImg, kernel.size(), 0, 0);

	/*step 3: Histogram Equalisation*/
	equalizeHist(filtImg, filtImg);
;

	/*step 4: Thresholding*/
	double maxVal = 255;
	double thresh = 127;			
	Mat binImg = filtImg.clone();
	threshold(filtImg, binImg, thresh, maxVal, THRESH_BINARY);
	imshow("Binary", binImg);

	//step 5: extract edges
	Mat edgeImg = binImg.clone();
	Canny(binImg, edgeImg, 0, 2, 3);
	//imshow("Canny", edgeImg);

	//step 6: Finding contours
	vector<vector<Point>> contours;
	vector<vector<Point>> contoursLargest;
	findContours(edgeImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<pair<double,int>> periLength;
	for (int cnt = 0; cnt < contours.size(); cnt++) {
		pair<double, int> peri;
		peri.first = arcLength(contours[cnt], true);
		peri.second = cnt;		
		periLength.push_back(peri);
	}
	sort(periLength.begin(), periLength.end(), greater<>());

	for (int cnt = 0; cnt < periLength.size()/2; cnt++) {
		int cntIdx = periLength[cnt].second;
		contoursLargest.push_back(contours[cntIdx]);
	}

	for (int cnt = 0; cnt < contoursLargest.size(); cnt++) {									
		drawContours(edgeImg, contoursLargest, cnt, (0,0,255), 2);
	}	
	//imshow("Contours", edgeImg); 

	/*for (int cnt = 0; cnt < contoursLargest.size(); cnt++) {											
		fillConvexPoly(edgeImg, contoursLargest[cnt], Scalar(255, 255, 255), 8);
	}*/
	//imshow("Filled", edgeImg);


	/*step 7 : morphological operations - opening to smooth out edges*/
	Mat openImg = edgeImg.clone();
	morphologyEx(edgeImg, openImg, MORPH_OPEN, kernel, Point(-1, -1), 1);
	//imshow("Opened", openImg);

	/*cropping lower abdomen area*/
	int offset_x = 60;
	int offset_y = 70;

	cv::Rect roi;
	roi.x = 30;
	roi.y = 10;
	roi.width = openImg.size().width - (offset_x);
	roi.height = openImg.size().height - (offset_y);

	cv::Mat crop = openImg(roi);
	cv::imshow("crop", crop);

	waitKey(0);
}
