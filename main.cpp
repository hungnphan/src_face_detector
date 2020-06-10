
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

/*
dlib::rectangle openCVRectToDlib(cv::Rect r)
{
	return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

cv::Rect dlibRectangleToOpenCV(dlib::rectangle r)
{
	return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}


void dlib_point2cv_Point(full_object_detection& S, std::vector<cv::Point>& L, double& scale)
{
	for (unsigned int i = 0; i < S.num_parts(); ++i)
	{
		L.push_back(cv::Point(S.part(i).x()*(1 / scale), S.part(i).y()*(1 / scale)));
	}
}

*/


void mergeOverlappingBoxes(std::vector<cv::Rect> &inputBoxes, cv::Mat &image, std::vector<cv::Rect> &outputBoxes)
{
	cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1); // Mask of original image
	cv::Size scaleFactor(20, 20); // To expand rectangles, i.e. increase sensitivity to nearby rectangles. Doesn't have to be (10,10)--can be anything
	for (int i = 0; i < inputBoxes.size(); i++)
	{
		cv::Rect box = inputBoxes.at(i) + scaleFactor;
		cv::rectangle(mask, box, cv::Scalar(255), CV_FILLED); // Draw filled bounding boxes on mask
	}

	std::vector<std::vector<cv::Point>> contours;
	// Find contours in mask
	// If bounding boxes overlap, they will be joined by this function call
	cv::findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	for (int j = 0; j < contours.size(); j++)
	{
		outputBoxes.push_back(cv::boundingRect(contours.at(j)));
	}
}


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

const std::string caffeConfigFile = "./deploy.prototxt";
const std::string caffeWeightFile = "./res10_300x300_ssd_iter_140000_fp16.caffemodel";

const std::string tensorflowConfigFile = "./opencv_face_detector.pbtxt";
const std::string tensorflowWeightFile = "./opencv_face_detector_uint8.pb";
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
	//roi.tl().x = 25;
	//roi.tl.y = 6;
	//roi.width = 634;
	//roi.height = 438;
	roi = Rect(Point(60, 20),
		Point(550, 390));
	//roi = Rect(Point(1,1), Point(636, 475));

	string video[4];
	video[0] = "P1L_S1_C1";
	video[1] = "P2E_S5_C2";
	video[2] = "P2L_S1_C1";
	video[3] = "P2L_S5_C2";
	for (int v = 0; v < 4; ++v) {

		total_faces = 0;
		// Initialize Video Capture Stream
		VideoCapture cap(video[v] + ".avi");

		
		vector<Rect> roiTrackings;

		//VideoCapture cap("https://root:Daicoviet12345????@hcmiucvip.com:8081/axis-media/media.amp");
	
	
		//VideoCapture cap(0);
		cout << roi.tl() << " " << roi << endl;
		CascadeClassifier face;
		
		//
		//std::vector<Rect> test;
		//face.detectMultiScale(inputImage, test, 1.1, 3, 0, Size(30, 30));
		//ofstream  file("out.txt");

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
			resize(inputImage, inputImage, Size(640, 480));
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
		
		int ksize = 3;
		int scale = 1;
		int delta = 0;
		int ddepth = CV_16S;
		int key = 0;
		Mat roiMat;
		int cnt = 0;

		
		int numFrame = 0;

		//int cnt_faces = 0;
		int faces_prev = 0;

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
			
			//bitwise_and(inputIe.clone();.
			//imshow("samplemage, inputImage, sample, ROImask);
			//inputImage = sampl", sample);
			//GammaCorrection(inputImage, inputImage, 0.7);
			original = inputImage.clone();

			resize(inputImage, inputImage, resolution, 0, 0, CV_INTER_LINEAR);
			//Calculate the background, foreground, filtered-foreground image
			bg_model->updateModel(&inputImage);

			if (cnt <= 80)
			{
				++cnt;
				continue;
			}

		//	cout << original.size() << endl;
			Mat* background_img_t = bg_model->getBackground();
			Mat* foreground_img_t = bg_model->getForeground();

			backgroundImage = background_img_t->clone();								// background_img in BGR
			foregroundImage = foreground_img_t->clone();								// foreground_img in BGR

			filter_->createBinaryForeground(foregroundImage, filter_foregroundImage);	// binary_foreground in CV_8U	
		//	cout << filter_foregroundImage.type() << endl;
			resize(filter_foregroundImage, filter_foregroundImage, Size(640, 480));
			//	show(inputImage);
			//show(filter_foregroundImage);



			//face.load("lbpcascade_frontalface.xml");
			face.load("haarcascade_frontalface_alt2.xml");
			//cout << roiTrackings.size() << endl;
			for (int i = 0; i < roiTrackings.size(); ++i)
			{
				Mat getROI = original(roiTrackings[i]);
				cvtColor(getROI, getROI, CV_RGB2GRAY);
				equalizeHist(getROI, getROI);
				//show(faceRoi);
				std::vector<Rect> faces;
				face.detectMultiScale(getROI, faces, 1.1, 3, 0, Size(30, 30));
				cout << "faces tracking: " << faces.size() << endl;
				if (faces.size() > 0) {
					for (int j = 0; j < faces.size(); ++j)
					{

						rectangle(original, Rect(Point(roiTrackings[i].x + faces[j].x, roiTrackings[i].y + faces[j].y),
							Point(roiTrackings[i].x + faces[j].x + faces[j].width, roiTrackings[i].y + faces[j].y + faces[j].height)), cv::Scalar(0, 255, 0), 2, 4);
						int x1 = max(1, int(roiTrackings[i].x + faces[j].x - faces[j].width*0.2));
						int y1 = max(1, int(roiTrackings[i].y + faces[j].y - faces[j].height*0.2));
						int x2 = min(635, int(roiTrackings[i].x + faces[j].x + faces[j].width + faces[j].width * 0.2));
						int y2 = min(475, int(roiTrackings[i].y + faces[j].y + faces[j].height + faces[j].height * 0.2));
						
						Rect temp = Rect(Point(roiTrackings[i].x + faces[j].x, roiTrackings[i].y + faces[j].y),
							Point(roiTrackings[i].x + faces[j].x + faces[j].width, roiTrackings[i].y + faces[j].y + faces[j].height));
						roiTrackings[i] = Rect(Point(x1, y1), Point(x2, y2));
						//cout << temp << endl;
						//cout << filter_foregroundImage.size() << endl;
						Mat roi = filter_foregroundImage(temp);
						//show(roi);
						///show(original(temp));
						roi.setTo(Scalar(0, 0, 0));
					}
				}
				else
				{
					if (roiTrackings.size() > 0) {
						roiTrackings.erase(roiTrackings.begin() + i);
						--i;
					}
				}

			}
		//	cout << "Done " << endl;

			show(filter_foregroundImage);

			face.load("lbpcascade_frontalface.xml");
			//face.load("haarcascade_frontalface_alt2.xml");
			Mat roiRegion = original(roi);
			show(roiRegion);
			Mat roiFore = filter_foregroundImage(roi);
			//show(roiFore);
			
			Mat dst;
			int dilation_size = 2;
			int erosion_size = 2;

			
			


			//detect.detect(roiRegion, roiFore);
			//imwrite("G:/Project1/Project1/result_SSD/" + video[v] + "/S2/" + to_string(cnt) + "_AT_Mor.jpg", roiFore);
			
			//imshow(window_name, roiFore);
			
			
			//imwrite("G:/Project1/Project1/result_SSD/" + video[v] + "/S1/" + to_string(cnt) + "_BG_1.jpg", roiFore);
			Mat object;
			bitwise_and(roiRegion, roiRegion, object, roiFore);
			//Mat printStep1;
			//bitwise_and(original, original, printStep1, filter_foregroundImage);
		
			object = gammaCorrection(object, 0.8);
			
			//show(object);
			//resize(object, object, Size(640, 480));
			//resize(object, object, Size(640, 480));
			//imwrite("G:/Project1/Project1/result_SSD/" + video[v] + "/S1/" + to_string(cnt) + "_BG.jpg", printStep1);
		//	imwrite("G:/Project1/Project1/result_SSD/" + videm[v] + "/S1/" + to_string(cnt) + ".jpg", original);
			std::vector<Rect> faceCandidates;
			//std::vector<Mat> faceCandidates;
		

			Mat canny_output;
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

			for (int i = 0; i < boundRect.size(); ++i)
			{
				//cv::rectangle(object, boundRect[i], 150);
				Mat faceRoi = object(boundRect[i]);
				//show(faceRoi);
				cvtColor(faceRoi, faceRoi, CV_RGB2GRAY);
				equalizeHist(faceRoi, faceRoi);
				//show(faceRoi);
				std::vector<Rect> faces;
				face.detectMultiScale(faceRoi, faces, 1.1, 3, 0, Size(35, 35));
				//cnt_faces = faces.size();

				for (int j = 0; j < faces.size(); ++j)
				{
					//++total_faces;
					//cv::rectangle(original, Rect(Point(roi.x +  boundRect[i].x + faces[j].x,roi.y + boundRect[i].y + faces[j].y),
					//	Point(roi.x + boundRect[i].x + faces[j].x + faces[j].width, roi.y + boundRect[i].y + faces[j].y + faces[j].height)), cv::Scalar(255, 0, 0), 2, 4);
					faceCandidates.push_back(Rect(Point(boundRect[i].x + faces[j].x, boundRect[i].y + faces[j].y),
						Point(boundRect[i].x + faces[j].x + faces[j].width, boundRect[i].y + faces[j].y + faces[j].height)));
				}
			}
			
		//	cout <<"faceCandidates " << faceCandidates.size() << endl;
			for (int i = 0; i < faceCandidates.size(); ++i)
			{
				//	double scale = 1.0;
				Mat temp = roiRegion(faceCandidates[i]);
				 //int x1, y1;
				 double confidence = 0;
				 Point top, bottom;
				 Rect roiF;
				 bool check = false;
				 detectFaceOpenCVDNN(net, temp, check, roiF);
				 /*
				 if (check == true) 
				 {
					 Rect roiTracking;
					 Point newtl, newbr;
					 newtl.x = std::max(0, roi.x + faceCandidates[i].x + (int)(roiF.tl().x));
					 newtl.y = std::max(0, roi.y + faceCandidates[i].y + (int)(roiF.tl().y));

					 newtl.x = std::min(638, roi.x + faceCandidates[i].x + (int)(roiF.br().x));
					 newtl.y = std::min(478, roi.y + faceCandidates[i].y + (int)(roiF.br().y));


					 int x1 = max(0, int(roi.x + faceCandidates[i].x + roiF.x - roiF.width*0.5));
					 int y1 = max(0, int(roi.y + faceCandidates[i].y + roiF.y - roiF.height*0.5));
					 int x2 = min(635, int(roi.x + faceCandidates[i].x + roiF.x + roiF.width + roiF.width*0.5));
					 int y2 = min(475, int(roi.y + faceCandidates[i].y + roiF.y + roiF.height + roiF.height * 0.5));
					 roiTracking = Rect(Point(x1, y1), Point(x2, y2));
					 roiTrackings.push_back(roiTracking);
				 }*/
				 ///rectangle(original, Rect(Point(x1, y1), Point(x2, y2)), Scalar(0, 0, 0));
			}
			
			//show(object);
			Mat compare;
			show(original);
			//imwrite("G:/Project1/Project1/result_SSD/" + video[v] + "/S3/" + to_string(cnt) + ".jpg", original);
			++cnt;
			//imwrite("result_compare/P2E_S5_C1/" + to_string(++cnt) + ".jpg", compare);

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



