//
// mixture_of_gaussian.h:
//
//	*** NOTICE: the functions are arranged in chronological order with the main function at the end ***
//
//
// Author: Long Pham
// Project: ITS
// Licensed to: Computer Vision & Image Processing Lab, International University, VNU-HCM
//
// Created: 19/10/2015 by Tien Nguyen
// Last Modified: 26/10/2015 by Long Pham 
// 



#ifndef MIXTURE_OF_GAUSSIAN_H
#define MIXTURE_OF_GAUSSIAN_H

//#include "stdafx.h"
#include "background_model.h"



///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
#define NUMBEROFGAUSSIAN 5
#define LEARNINGRATE 0.02
#define THRESHOLD 2.5
#define BGTHRESHOLD 0.6
#define INITIALVAR 600.0
*/

#define NUMBEROFGAUSSIAN 4
#define LEARNINGRATE 0.025
#define THRESHOLD 2.5
#define BGTHRESHOLD 0.6
#define INITIALVAR 400.0



class MOG : public BackgroundModel {

public:
	MOG(int, int);
	~MOG();

	void setMOGParameter(int, int, double*);



protected:
	MOGDATA2* m_pMOG;
	int* m_pK; // number of distributions per pixel
	void Init();
	void Update();

	double m_alpha;
	double m_threshold;
	double m_noise;
	double m_T;

};

#endif