#include "Detector.h"




void detector::init()
{
	this->cnt = 0;

	 net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
	 LBP = LBPHFaceRecognizer::create();
	 LBP->read("P1L_S1_C1.xml");

	 //this->facemark = FacemarkLBF::create();
	// this->facemark->loadModel("lbfmodel.yaml");
	 this->face.load("lbpcascade_frontalface.xml");
	 this->roiTrackings.clear();
	 this->IDs.clear();
}





void detectFaceOpenCVDNN(Net net, Mat &frameOpenCVDNN, bool &flat, Rect &roiF)
{
	
	//int total_faces = 0;
	const size_t inWidth1 = 65;
	const size_t inHeight1 = 65;
	const double inScaleFactor1 = 0.95;
	const float confidenceThreshold1 = 0.7;
	const cv::Scalar meanVal1(104.0, 177.0, 123.0);

	int frameHeight = frameOpenCVDNN.rows;
	int frameWidth = frameOpenCVDNN.cols;

	cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor1, cv::Size(inWidth1, inHeight1), meanVal1, false, false);


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
			//++total_faces;
			flat = true;
			roiF = Rect(cv::Point(x1, y1), cv::Point(x2, y2));
		}
	}
	
}

Mat detector::getfaces(Mat image, vector<Mat> &outputFaces)
{
	
	Size resolution(320, 240);
	Rect roi;
	roi = Rect(Point(20, 20),
		Point(600, 390));
	
	

	if (cnt == 0)
	{
		resize(image, image, Size(640, 480));
		//resolution = inputImage.size();
		resize(image, image, resolution, 0, 0, CV_INTER_LINEAR);
		bg_model = new MOG(resolution.width, resolution.height);
		bg_model->initializeModel(&image);
		filter_ = new Filter(100, 100);
		++cnt;
		return image;
	}

	original = image.clone();

	resize(image, image, resolution, 0, 0, CV_INTER_LINEAR);
	//Calculate the background, foreground, filtered-foreground image
	bg_model->updateModel(&image);

	if (cnt <= 80)
	{
		++cnt;
		return image;
	}

	Mat* background_img_t = bg_model->getBackground();
	Mat* foreground_img_t = bg_model->getForeground();

	backgroundImage = background_img_t->clone();								// background_img in BGR
	foregroundImage = foreground_img_t->clone();								// foreground_img in BGR

	filter_->createBinaryForeground(foregroundImage, filter_foregroundImage);	// binary_foreground in CV_8U	
//	cout << filter_foregroundImage.type() << endl;
	resize(filter_foregroundImage, filter_foregroundImage, Size(640, 480));

	//face.load("lbpcascade_frontalface.xml");
	
	for (int i = 0; i < roiTrackings.size(); ++i)
	{
		Mat getROI = original(roiTrackings[i]);
		cvtColor(getROI, getROI, CV_RGB2GRAY);
		equalizeHist(getROI, getROI);
		//show(faceRoi);
		std::vector<Rect> faces;
		face.detectMultiScale(getROI, faces, 1.1, 3, 0, Size(30, 30));
		vector< vector<Point2f> > landmarks;

	    
		cout << "faces tracking: " << faces.size() << endl;
		if (faces.size() > 0) {
			for (int j = 0; j < faces.size(); ++j)
			{
				
				if (IDs[i] == 0) {
					Mat faceR = getROI(faces[0]).clone();
					imshow("FaceR", faceR);
					resize(faceR, faceR, Size(50, 50));
					int ID;
					double conf;
					LBP->predict(faceR, ID, conf);

					if (conf <= 135) {
						IDs[i] = ID;
						//cout << ID << endl;
						Mat IDP = imread("F:/getTrainImage/x64/Release/data/" + to_string(ID + 1) + ".jpg");
						imshow("IDP", IDP);

						cout << conf << endl;

					}
					else
					{
						IDs[i] = 1000;
						cout << "unknow" << "" << conf << endl;
					}

				}
				string outname = "";
				if (IDs[i] == 1000)
				{
					outname = "unknown";
				}
				else outname = to_string(IDs[i]);
				putText(original, outname, Point(roiTrackings[i].x + faces[j].x, roiTrackings[i].y + faces[j].y), 3, 2, cv::Scalar(0, 255, 0));
					
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
				IDs.erase(IDs.begin() + i);
				--i;
			}
		}
		

	}
	
	//face.load("lbpcascade_frontalface.xml");

	Mat roiRegion = original(roi);   
	//(roiRegion);
	Mat roiFore = filter_foregroundImage(roi);
	Mat dst;
	int dilation_size = 2;
	int erosion_size = 2;
	Mat object;
	bitwise_and(roiRegion, roiRegion, object, roiFore);
	std::vector<Rect> faceCandidates;
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
		if (boundingRect(Mat(contours_poly[i])).area() >= 30 * 30)
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
			//::rectangle(printStep1, Rect(Point(roi.x +  boundRect[i].x + faces[j].x,roi.y + boundRect[i].y + faces[j].y),
			//	Point(roi.x + boundRect[i].x + faces[j].x + faces[j].width, roi.y + boundRect[i].y + faces[j].y + faces[j].height)), cv::Scalar(0, 255, 0), 2, 4);
			faceCandidates.push_back(Rect(Point(boundRect[i].x + faces[j].x, boundRect[i].y + faces[j].y),
				Point(boundRect[i].x + faces[j].x + faces[j].width, boundRect[i].y + faces[j].y + faces[j].height)));
		}
	}
	//imwrite("F:/studentResearch/studentResearch/Result/" + video[v] + "/S2/" + to_string(cnt) + ".jpg", printStep1);
	//show(printStep1);
	//cout <<"faceCandidates " << faceCandidates.size() << endl;
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
		
		if (check == true)
		{
			Rect roiTracking;
			Point newtl, newbr;
			newtl.x = std::max(0, roi.x + faceCandidates[i].x + (int)(roiF.tl().x));
			newtl.y = std::max(0, roi.y + faceCandidates[i].y + (int)(roiF.tl().y));

			newbr.x = std::min(635, roi.x + faceCandidates[i].x + (int)(roiF.br().x));
			newbr.y = std::min(475, roi.y + faceCandidates[i].y + (int)(roiF.br().y));


			int x1 = max(0, int(roi.x + faceCandidates[i].x + roiF.x - roiF.width*0.5));
			int y1 = max(0, int(roi.y + faceCandidates[i].y + roiF.y - roiF.height*0.5));
			int x2 = min(635, int(roi.x + faceCandidates[i].x + roiF.x + roiF.width + roiF.width*0.5));
			int y2 = min(475, int(roi.y + faceCandidates[i].y + roiF.y + roiF.height + roiF.height * 0.5));
			cout << x1 << " " << y1 << " " << x2 << " " << y2 << endl;
			roiTracking = Rect(Point(x1, y1), Point(x2, y2));
			roiTrackings.push_back(roiTracking);
			IDs.push_back(0);
			outputFaces.push_back(temp(roiF));
		}
		
		
		///rectangle(original, Rect(Point(x1, y1), Point(x2, y2)), Scalar(0, 0, 0));
	}
	//tt_opencvHaar = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	//fpsOpencvHaar = 1 / tt_opencvHaar;
	//putText(original, format("OpenCV HAAR ; FPS = %.16f", tt_opencvHaar), Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 4);
	//show(object);
	//Mat compare;
	//imshow("ori", original);
	++cnt;

	return original;
	//imwrite("G:/Project1/Project1/result_SSD/" + video[v] + "/S3/" + to_string(cnt) + ".jpg", original);
	

	//Mat i;

	//return i;

}