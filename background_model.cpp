//
// background_model.cpp:
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



#include "stdafx.h"
#include "background_model.h"



///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
BackgroundModel::BackgroundModel(int width, int height) : m_width(width), m_height(height) {

	m_SourceImage = new Mat(Mat::zeros(height, width, CV_8UC3));
	m_BackgroundImage = new Mat(Mat::zeros(height, width, CV_8UC3));
	m_ForegroundImage = new Mat(Mat::zeros(height, width, CV_8UC3));

	BYTERGB *prgbSrc = (BYTERGB*)m_SourceImage->data;
	BYTERGB *prgbBG = (BYTERGB*)m_BackgroundImage->data;
	BYTERGB *prgbFG = (BYTERGB*)m_ForegroundImage->data;

	for (int k = 0; k<m_width*m_height; k++) {

		prgbSrc->Red = prgbBG->Red = prgbFG->Red = 0;
		prgbSrc->Green = prgbBG->Green = prgbFG->Green = 0;
		prgbSrc->Blue = prgbBG->Blue = prgbFG->Blue = 0;

		prgbSrc++;
		prgbBG++;
		prgbFG++;
	}
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
BackgroundModel::~BackgroundModel() {

	if (m_SourceImage != NULL) delete m_SourceImage;
	if (m_BackgroundImage != NULL) delete m_BackgroundImage;
	if (m_ForegroundImage != NULL) delete m_ForegroundImage;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
void BackgroundModel::initializeModel(Mat* image) {

	*m_SourceImage = image->clone();
	Init();
	return;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
void BackgroundModel::updateModel(Mat* image) {

	*m_SourceImage = image->clone();
	Update();
	//erodeFG();
	//dilateFG();
	return;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
Mat* BackgroundModel::getSource() {
	return m_SourceImage;
}


Mat* BackgroundModel::getForeground() {
	return m_ForegroundImage;
}


Mat* BackgroundModel::getBackground() {
	return m_BackgroundImage;
}


void BackgroundModel::setBackgroundModelParameter() {
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
void BackgroundModel::erodeForeground() {

	Mat ErodeImage(m_width, m_height, CV_8UC3);
	ptrdiff_t inc = 1;

	BYTERGB *prgbFG = (BYTERGB*)m_ForegroundImage->data;
	BYTERGB *prgbErode = (BYTERGB*)ErodeImage.data;

	for (int j = 1; j<m_height - 1; j++) {

		BYTERGB *prgbFGm = prgbFG + static_cast<ptrdiff_t>(j*m_width + 1);
		BYTERGB *prgbFGt = prgbFGm - static_cast<ptrdiff_t>(m_width);
		BYTERGB *prgbFGb = prgbFGm + static_cast<ptrdiff_t>(m_width);
		BYTERGB *prgbPixel = prgbErode + static_cast<ptrdiff_t>(j*m_width + 1);

		for (int i = 1; i<m_width - 1; i++) {

			prgbPixel->Red = prgbPixel->Green = prgbPixel->Blue =
				(prgbFGm->Green & ((prgbFGt - inc)->Green & prgbFGt->Green & (prgbFGt + inc)->Green &
				(prgbFGm - inc)->Green & (prgbFGm + inc)->Green &
				(prgbFGb - inc)->Green & prgbFGb->Green & (prgbFGb + inc)->Green));

			prgbFGm++;
			prgbFGt++;
			prgbFGb++;
			prgbPixel++;
		}
	}

	*m_ForegroundImage = ErodeImage.clone();
	return;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
void BackgroundModel::dilateForeground() {

	Mat DilateImage(m_width, m_height, CV_8UC3);
	ptrdiff_t inc = 1;

	BYTERGB *prgbFG = (BYTERGB*)m_ForegroundImage->data;
	BYTERGB *prgbDilate = (BYTERGB*)DilateImage.data;

	for (int j = 1; j<m_height - 1; j++) {

		BYTERGB *prgbFGm = prgbFG + static_cast<ptrdiff_t>(j*m_width + 1);
		BYTERGB *prgbFGt = prgbFGm - static_cast<ptrdiff_t>(m_width);
		BYTERGB *prgbFGb = prgbFGm + static_cast<ptrdiff_t>(m_width);
		BYTERGB *prgbPixel = prgbDilate + static_cast<ptrdiff_t>(j*m_width + 1);

		for (int i = 1; i<m_width - 1; i++) {

			prgbPixel->Red = prgbPixel->Green = prgbPixel->Blue =
				(prgbFGm->Green | ((prgbFGt - inc)->Green | prgbFGt->Green | (prgbFGt + inc)->Green |
				(prgbFGm - inc)->Green | (prgbFGm + inc)->Green |
				(prgbFGb - inc)->Green | prgbFGb->Green | (prgbFGb + inc)->Green));

			prgbFGm++;
			prgbFGt++;
			prgbFGb++;
			prgbPixel++;
		}
	}

	*m_ForegroundImage = DilateImage.clone();
	return;
}