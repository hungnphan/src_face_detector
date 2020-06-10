//
// filter.h:
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



#ifndef FILTER_H
#define FILTER_H

//#include "stdafx.h"



///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
class Filter {

public:
	Filter(void);
	Filter(double min_vehicle_size, double max_vehicle_size);

	// Create the binary image of foreground, also enhance foreground objects
	void createBinaryForeground(const Mat& foreground, Mat& binary_foreground);



protected:
	double min_vehicle_size_;
	double max_vehicle_size_;

};


#endif