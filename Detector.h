#pragma once
#include "stdafx.h"



#define CAFFE

class detector
{
public:
	Mat input;
	CascadeClassifier cas;
	void init();
	void detect(Mat &roiRegion, Mat &roiFore);

	void detectFaceOpenCVDNN(Mat &frameOpenCVDNN);

	vector<Rect> getFaces();

private:
	const std::string caffeConfigFile = "./deploy.prototxt";
	const std::string caffeWeightFile = "./res10_300x300_ssd_iter_140000_fp16.caffemodel";
	const std::string tensorflowConfigFile = "./opencv_face_detector.pbtxt";
	const std::string tensorflowWeightFile = "./opencv_face_detector_uint8.pb";
	const std::string faceLBP = "lbpcascade_frontalface.xml";
	const std::string faceHaar = "haarcascade_frontalface_alt2.xml";


	vector<Rect> faces;
	Net net;
	CascadeClassifier face;
	


};
