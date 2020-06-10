//
// mixture_of_gaussian.cpp:
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
#include "mixture_of_gaussian.h"



///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
MOG::MOG(int width, int height) : BackgroundModel(width, height) {
	m_alpha = LEARNINGRATE;
	m_threshold = THRESHOLD*THRESHOLD;
	m_noise = INITIALVAR;
	m_T = BGTHRESHOLD;
	m_pMOG = new MOGDATA2[NUMBEROFGAUSSIAN*m_width*m_height];
	m_pK = new int[m_width*m_height];

	MOGDATA2 *pMOG = m_pMOG;
	int *pK = m_pK;

	for (int i = 0; i < m_width*m_height; i++) {
		for (int k = 0; k < NUMBEROFGAUSSIAN; k++) {
			pMOG->mu.Red = 0.0;
			pMOG->mu.Green = 0.0;
			pMOG->mu.Blue = 0.0;
			pMOG->var = 0.0;
			pMOG->w = 0.0;
			pMOG->sortKey = 0.0;
			pMOG++;
		}
		pK[i] = 0;
	}
}


MOG::~MOG() {
	delete[] m_pMOG;
	delete[] m_pK;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
void MOG::setMOGParameter(int id, int value, double* rvalue) {
	double dvalue = (double)value / 255.0;

	switch (id) {
	case 0:
		m_threshold = 100.0*dvalue*dvalue;
		*rvalue = 10.0*dvalue;
		break;
	case 1:
		*rvalue = m_T = dvalue;
		break;
	case 2:
		*rvalue = m_alpha = dvalue*dvalue*dvalue;
		break;
	case 3:
		*rvalue = m_noise = 100.0*dvalue;;
		break;
	}
	return;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
void MOG::Init() {
	MOGDATA2 *pMOG = m_pMOG;
	BYTERGB *prgbSrc = (BYTERGB*)m_SourceImage->data;
	int *pK = m_pK;

	for (int i = 0; i < m_width*m_height; i++) {
		pMOG[0].mu.Red = prgbSrc->Red;
		pMOG[0].mu.Green = prgbSrc->Green;
		pMOG[0].mu.Blue = prgbSrc->Blue;
		pMOG[0].var = m_noise;
		pMOG[0].w = 1.0;
		pMOG[0].sortKey = pMOG[0].w / pMOG[0].var;
		pK[i] = 1;

		pMOG += NUMBEROFGAUSSIAN;
		prgbSrc++;
	}
	return;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
void MOG::Update() {
	int kBG;
	int *pK = m_pK;

	MOGDATA2 *pMOG = m_pMOG;
	BYTERGB *pRGBSrc = (BYTERGB*)m_SourceImage->data;
	BYTERGB *pRGBBG = (BYTERGB*)m_BackgroundImage->data;
	BYTERGB *pRGBFG = (BYTERGB*)m_ForegroundImage->data;

	for (int i = 0; i < m_width*m_height; i++) {
		double srcR = (double)pRGBSrc->Red;
		double srcG = (double)pRGBSrc->Green;
		double srcB = (double)pRGBSrc->Blue;

		// Find matching distribution
		int kHit = -1;
		for (int k = 0; k < pK[i]; k++) {

			// Mahalanobis distance 
			double dr = srcR - pMOG[k].mu.Red;
			double dg = srcG - pMOG[k].mu.Green;
			double db = srcB - pMOG[k].mu.Blue;
			double d2 = dr*dr + dg*dg + db*db;
			//double d2 = dr*dr + dg*dg + db*db;

			if (d2 < m_threshold*pMOG[k].var) {
				kHit = k;
				break;
			}
		}

		// Adjust parameters
		if (kHit != -1) {		// matching distribution found
			for (int k = 0; k < pK[i]; k++) {
				if (k == kHit) {
					pMOG[k].w = pMOG[k].w + m_alpha*(1.0f - pMOG[k].w);
					double d;

					d = srcR - pMOG[k].mu.Red;
					if (d*d > DBL_MIN) {
						pMOG[k].mu.Red += m_alpha*d;
					}

					d = srcG - pMOG[k].mu.Green;
					if (d*d > DBL_MIN) {
						pMOG[k].mu.Green += m_alpha*d;
					}

					d = srcB - pMOG[k].mu.Blue;
					if (d*d > DBL_MIN) {
						pMOG[k].mu.Blue += m_alpha*d;
					}

					d = (srcR - pMOG[k].mu.Red)*(srcR - pMOG[k].mu.Red) - pMOG[k].var;
					if (d*d > DBL_MIN) {
						pMOG[k].var += m_alpha*d;
					}

					d = (srcG - pMOG[k].mu.Green)*(srcG - pMOG[k].mu.Green) - pMOG[k].var;
					if (d*d > DBL_MIN) {
						pMOG[k].var += m_alpha*d;
					}

					d = (srcB - pMOG[k].mu.Blue)*(srcB - pMOG[k].mu.Blue) - pMOG[k].var;
					if (d*d > DBL_MIN) {
						pMOG[k].var += m_alpha*d;
					}

					pMOG[k].var = (std::max)(pMOG[k].var, m_noise);
				}
				else {
					pMOG[k].w = (1.0 - m_alpha)*pMOG[k].w;
				}
			}
		}
		else {				// no match found... create new one
			if (pK[i] < NUMBEROFGAUSSIAN) {
				pK[i]++;
			}

			kHit = pK[i] - 1;
			if (pK[i] == 1) {
				pMOG[kHit].w = 1.0;
			}
			else {
				pMOG[kHit].w = LEARNINGRATE;
			}

			pMOG[kHit].mu.Red = srcR;
			pMOG[kHit].mu.Green = srcG;
			pMOG[kHit].mu.Blue = srcB;
			pMOG[kHit].var = m_noise;
		}

		// Normalize weights
		double wsum = 0.0;
		for (int k = 0; k < pK[i]; k++) {
			wsum += pMOG[k].w;
		}

		double wfactor = 1.0 / wsum;
		for (int k = 0; k < pK[i]; k++) {
			pMOG[k].w *= wfactor;
			pMOG[k].sortKey = pMOG[k].w / pMOG[k].var;
		}

		// Sort distributions
		for (int k = 0; k < kHit; k++) {
			if (pMOG[kHit].sortKey > pMOG[k].sortKey) {
				std::swap(pMOG[kHit], pMOG[k]);
				break;
			}
		}

		// Determine background distributions
		wsum = 0.0;
		for (int k = 0; k < pK[i]; k++) {
			wsum += pMOG[k].w;
			if (wsum > m_T) {
				kBG = k;
				break;
			}
		}

		if (kHit > kBG) {
			// detecting shadow
			bool isShadow = false;
			double tw = 0.0;
			double numerator, denominator;
			for (int k = 0; k<pK[i]; k++)
			{

				numerator = pMOG[k].mu.Red*srcR + pMOG[k].mu.Green*srcG + pMOG[k].mu.Blue*srcB;
				denominator = pMOG[k].mu.Red*pMOG[k].mu.Red + pMOG[k].mu.Green*pMOG[k].mu.Green + pMOG[k].mu.Blue*pMOG[k].mu.Blue;

				if (denominator == 0) break;
				double a = numerator / denominator;

				tw += pMOG[k].w;

				if (a <= 1 && a >= 0.45)
				{

					double dist2 = 0.0;

					double dR = a*pMOG[k].mu.Red - srcR;
					double dG = a*pMOG[k].mu.Green - srcG;
					double dB = a*pMOG[k].mu.Blue - srcB;
					dist2 = dR*dR + dG*dG + dB*dB;

					if (dist2 < 16 * (pMOG[k].var)*a*a)
					{
						isShadow = true;
						break;
					}
				}

				if (tw > m_T)
				{
					isShadow = false;
					break;
				}

			}

			//pRGBFG->Red = pRGBFG->Green = pRGBFG->Blue = isShadow ? 128 : 255;
			pRGBFG->Red = pRGBFG->Green = pRGBFG->Blue = isShadow ? 5 : 255;

		}
		else
			pRGBFG->Red = pRGBFG->Green = pRGBFG->Blue = 0;




		pRGBBG->Red = (unsigned char)pMOG[0].mu.Red;
		pRGBBG->Green = (unsigned char)pMOG[0].mu.Green;
		pRGBBG->Blue = (unsigned char)pMOG[0].mu.Blue;

		if (pRGBFG->Red == 0 || pRGBFG->Green == 0 || pRGBFG->Blue == 0)
			if (abs(pRGBBG->Red - srcR) > 20 || abs(pRGBBG->Green - srcG) > 20 || abs(pRGBBG->Blue - srcB) > 20) {
				pRGBFG->Red = pRGBFG->Green = pRGBFG->Blue = 255;
			}
			else {
				pRGBFG->Red = pRGBFG->Green = pRGBFG->Blue = 0;
			}

			pMOG += NUMBEROFGAUSSIAN;
			pRGBSrc++;
			pRGBBG++;
			pRGBFG++;
	}
	return;
}