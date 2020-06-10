//
// background_model.h:
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



#ifndef BACKGROUND_MODEL_H
#define BACKGROUND_MODEL_H

//#include "stdafx.h"



///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
typedef struct tagDBLRGB{
	double Red;
	double Green;
	double Blue;
} DBLRGB;


typedef struct tagBYTERGB{
	unsigned char Red;
	unsigned char Green;
	unsigned char Blue;
} BYTERGB;


typedef struct tagMOGDATA{
	DBLRGB mu;
	DBLRGB var;
	double w;
	double sortKey;
} MOGDATA;


typedef struct tagMOGDATA2{
	DBLRGB mu;
	double var;
	double w;
	double sortKey;
} MOGDATA2;



///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
class BackgroundModel {

public:
	BackgroundModel(int m_width, int m_height);
	virtual ~BackgroundModel();

	void initializeModel(Mat* image);
	void updateModel(Mat* image);
	virtual void setBackgroundModelParameter();

	virtual Mat* getSource();
	virtual Mat* getForeground();
	virtual Mat* getBackground();


protected:
	const int m_width;
	const int m_height;

	virtual void Init() = 0;
	virtual void Update() = 0;

	void erodeForeground();
	void dilateForeground();

	Mat* m_SourceImage;
	Mat* m_BackgroundImage;
	Mat* m_ForegroundImage;

};

#endif