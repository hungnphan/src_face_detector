//
// filter.cpp:
//
//	*** NOTICE: the functions are arranged in chronological order with the main function at the end ***
//
//
// Author: Long Pham
// Project: ITS
// Licensed to: Computer Vision & Image Processing Lab, International University, VNU-HCM
//
// Created: 28/10/2015 by Long Pham
// Last Modified: 28/10/2015 by Long Pham 
// 



#include "stdafx.h"
#include "filter.h"



///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor() / Destructor() / Initialize() / Reset()
///////////////////////////////////////////////////////////////////////////////////////////////////////////
Filter::Filter(void) : min_vehicle_size_(50.0),
					   max_vehicle_size_(50.0) {}


Filter::Filter(double min_vehicle_size = 50.0, double max_vehicle_size = 50.0) {

	min_vehicle_size_ = min_vehicle_size;
	max_vehicle_size_ = max_vehicle_size;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////
// createBinaryForeground()
//	- Convert the foreground image to binary
//	- Apply morphology to enhance the foreground image
///////////////////////////////////////////////////////////////////////////////////////////////////////////
void Filter::createBinaryForeground(const Mat& foreground, Mat& binary_foreground) {

	// Apply closing morphological transformation
	Mat gray;
	cvtColor(foreground, gray, CV_BGR2GRAY);
	//imshow("Gray", gray);
	morphologyEx(gray, gray, MORPH_CLOSE, Mat());

	// Fill in holes on foreground objects
	Mat result = Mat::zeros(gray.size(), CV_8U);

	vector<vector<Point>> contours;
	findContours(gray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	for (size_t i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i]);
		if (area > 5) {
			drawContours(result, contours, (int) i, 255, -1);
		}
	}

	

	// Convert to binary image
	threshold(result, binary_foreground, 0, 255, CV_THRESH_BINARY);

}