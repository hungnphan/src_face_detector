
#include "stdafx.h"
#include "background_model.h"
#include "mixture_of_gaussian.h"
#include "filter.h"
//#include "Detector.h"


#include <opencv2/ml.hpp>
#include <opencv2/dnn.hpp>
//#include <dlib/opencv.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
//#include <dlib/image_processing/frontal_face_detector.h>
//#include <dlib/image_processing/render_face_detections.h>
//#include <dlib/image_processing.h>
//#include <dlib/gui_widgets.h>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


#include <opencv2/xphoto/white_balance.hpp>

//#include <dlib/opencv.h>
//#include <dlib/image_processing.h>
//#include <dlib/dnn.h>
//#include <dlib/data_io.h>
//using namespace dlib;

#define bug(x) cout << #x << " = " << x << endl;
#define show(x) imshow(#x,x);
using namespace cv;
using namespace std;



// new variables
BackgroundModel* bg_model;
Filter* filter_;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 177.0, 123.0);






Mat gammaCorrection(const Mat &img, const double gamma_)
{
	CV_Assert(gamma_ >= 0);
	//! [changing-contrast-brightness-gamma-correction]
	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);

	Mat res = img.clone();
	LUT(img, lookUpTable, res);
	//! [changing-contrast-brightness-gamma-correction]

	//hconcat(img, res, img_gamma_corrected);
	return res;
	//imshow("Gamma correction", img_gamma_corrected);
}

using namespace cv::dnn;
int total_faces = 0;

const size_t inWidth1 = 50;
const size_t inHeight1 = 50;
const double inScaleFactor1 = 0.95;
const float confidenceThreshold1 = 0.7;
const cv::Scalar meanVal1(104.0, 177.0, 123.0);
//const cv::Scalar meanVal1(26.0, 44.25, 30.75);

#define CAFFE

const std::string caffeConfigFile = "./models/deploy.prototxt";
const std::string caffeWeightFile = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel";

const std::string tensorflowConfigFile = "./models/opencv_face_detector.pbtxt";
const std::string tensorflowWeightFile = "./models/opencv_face_detector_uint8.pb";
int cntFace = 0;

void detectFaceOpenCVDNN(Net net, Mat &frameOpenCVDNN,bool &flat, Rect &roiF)
{
	int frameHeight = frameOpenCVDNN.rows;
	int frameWidth = frameOpenCVDNN.cols;
#ifdef CAFFE
	cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor1, cv::Size(inWidth1, inHeight1), meanVal1, false, false);
#else
	cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor1, cv::Size(inWidth1, inHeight1), meanVal1, true, false);
#endif

	net.setInput(inputBlob, "data");
	cv::Mat detection = net.forward("detection_out");

	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	//double ans;
	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);
		if (confidence > confidenceThreshold1)
		{
			//ans = confidence;
			int x1 = max(static_cast<int>(detectionMat.at<float>(i, 3)* frameWidth), 0);
			int y1 = max(static_cast<int>(detectionMat.at<float>(i, 4)* frameHeight), 0);
			int x2 = min(static_cast<int>(detectionMat.at<float>(i, 5)* frameWidth), frameWidth - 5);
			int y2 = min(static_cast<int>(detectionMat.at<float>(i, 6)* frameHeight), frameHeight - 5);
			cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);
			++total_faces;
			flat = true;			
			roiF = Rect(cv::Point(x1, y1), cv::Point(x2, y2));
		}
	}

}

/////////////////////////////////////////////
// main program
/////////////////////////////////////////////
int main() {

#ifdef CAFFE
	Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
#else
	Net net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);
#endif

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	// Load face detection and pose estimation models.
	//frontal_face_detector detector = get_frontal_face_detector();
	//shape_predictor pose_model;
	//dlib::shape_predictor predictor_;
	//deserialize("shape_predictor_5_face_landmarks.dat") >> predictor_;
	Mat inputImage, backgroundImage, foregroundImage, filter_foregroundImage, original;
	Filter* filter_;
	BackgroundModel* bg_model;
	//detector detect;
	//detect.init();
	Size resolution(320, 240);
	
	Rect roi;
	roi = Rect(Point(60, 20),
		Point(550, 390));


	string video[4];
	video[0] = "P1L_S1_C1";
	video[1] = "P2E_S5_C2";
	video[2] = "P2L_S1_C1";
	video[3] = "P2L_S5_C2";
	for (int v = 0; v < 4; ++v) {

		total_faces = 0;
		// Initialize Video Capture Stream
		VideoCapture cap(video[v] + ".avi");


		cout << roi.tl() << " " << roi << endl;

		CascadeClassifier face;
		face.load("./models/lbpcascade_frontalface.xml");

		// Testing the Video Stream (?)
		// If OK then execute continue
		// If Fail then stop the program
		if (cap.grab() == false)
		{
			cout << "Cannot read the video !\n";
			cerr << "Error" << endl;
			waitKey(0);
			//return 0;
		}
		else
		{
			cout << "Read the video !\n";
			cap.retrieve(inputImage);
			//resize(inputImage, inputImage, Size(640, 480));
			//resolution = inputImage.size();
			resize(inputImage, inputImage, resolution, 0, 0, CV_INTER_LINEAR);
		}



		// Initialize the Background Subtraction Model
		bg_model = new MOG(resolution.width, resolution.height);
		bg_model->initializeModel(&inputImage);
		filter_ = new Filter(100, 100);


		// Initialize Showing Windows For Result Images (maybe unnecessary)
		//namedWindow("inputImage", WINDOW_NORMAL);
		//namedWindow("backgroundImage", WINDOW_NORMAL);
		//namedWindow("filter_foregroundImage", WINDOW_NORMAL);
		//moveWindow("inputImage", 0, 0);
		//moveWindow("backgroundImage", 0, 360);
		//moveWindow("filter_foregroundImage", 320, 0);


		///////////////////////////////////////////////
		// Main work process
		///////////////////////////////////////////////

		int index_image = 0;
		
		Mat image, src, src_gray;
		Mat grad;
		
		


		int cnt = 0;
		int numFrame = 0;


		clock_t tStart = clock();
		while (true)
		{
			cap >> inputImage;
			if (inputImage.empty())
			{
				break;
			}

			resize(inputImage, inputImage, Size(640, 480));
			++numFrame;
			original = inputImage.clone();

			resize(inputImage, inputImage, resolution, 0, 0, CV_INTER_LINEAR);
			//Calculate the background, foreground, filtered-foreground image
			bg_model->updateModel(&inputImage);

			if (cnt <= 80)
			{
				++cnt;
				continue;
			}



			// I. Background Substraction //

			Mat* background_img_t = bg_model->getBackground();
			Mat* foreground_img_t = bg_model->getForeground();

			backgroundImage = background_img_t->clone();								// background_img in BGR
			foregroundImage = foreground_img_t->clone();								// foreground_img in BGR

			filter_->createBinaryForeground(foregroundImage, filter_foregroundImage);	// binary_foreground in CV_8U	
			resize(filter_foregroundImage, filter_foregroundImage, Size(640, 480));
			show(filter_foregroundImage);

			// get roi region in the original Image //
			Mat roiRegion = original(roi);
			show(roiRegion);

			// get roi region in the filter foreground Image  //
			Mat roiFore = filter_foregroundImage(roi);

			
			// Filter moving objects // 
			Mat object;
			bitwise_and(roiRegion, roiRegion, object, roiFore);
		
			object = gammaCorrection(object, 0.8);



		

			// Find contours for Proposal Region Detection //
			std::vector<Rect> faceCandidates;
			std::vector<std::vector<Point> > contours;
			std::vector<Rect> boundRect;
			std::vector<Vec4i> hierarchy;
			/// Find contours
			findContours(roiFore, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
			std::vector< std::vector<Point> > contours_poly(contours.size());
			Mat drawing = Mat::zeros(roiFore.size(), CV_8UC3);
			for (int i = 0; i < contours.size(); i++)
			{
				approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
				if (boundingRect(Mat(contours_poly[i])).area() >= 30*30)
				{
					boundRect.push_back(boundingRect(Mat(contours_poly[i])));
				}
			}



			// II. Proposal Region Detection //
			// Local Binary Pattern OpenCV 3.4
			for (int i = 0; i < boundRect.size(); ++i)
			{

				Mat faceRoi = object(boundRect[i]);
				cvtColor(faceRoi, faceRoi, CV_RGB2GRAY);
				
				// Histogram equalization //
				equalizeHist(faceRoi, faceRoi);
				std::vector<Rect> faces;
				face.detectMultiScale(faceRoi, faces, 1.1, 3, 0, Size(35, 35));

				for (int j = 0; j < faces.size(); ++j)
				{
					faceCandidates.push_back(Rect(Point(boundRect[i].x + faces[j].x, boundRect[i].y + faces[j].y),
						Point(boundRect[i].x + faces[j].x + faces[j].width, boundRect[i].y + faces[j].y + faces[j].height)));
				}
			}
			

			// III. Face Detection using SSD - ResNet10 // 
			for (int i = 0; i < faceCandidates.size(); ++i)
			{
				Mat temp = roiRegion(faceCandidates[i]);
				 //int x1, y1;
				 double confidence = 0;
				 Point top, bottom;
				 Rect roiF;
				 bool check = false;
				 detectFaceOpenCVDNN(net, temp, check, roiF);
			}
			

			show(original);
			++cnt;

			// [Run Option 1] Output frame-by-frame
			//waitKey(0);

			// [Run Option 2] Output continously a sequence of frame
			// if key press = SPACE, then pause the procedure
			// if key press = ESC, then stop the procedure
			char c = waitKey(1);
			if (c == 27) break;
			if (c == ' ') while (waitKey(0) != ' ');


		}
		printf("Time taken: %.5fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
		cout << total_faces << endl;
		cout << numFrame << endl;
	}
	//video.release();
	waitKey(0);

	return 0;
}



