// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

//#include "targetver.h"

#include <stdio.h>
#include <tchar.h>



// TODO: reference additional headers your program requires here
///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Include C++ standard libraries /////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Input-output streams
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip> 
#include <tchar.h>
#include <cstdio>

// Data structures
#include <vector>
#include <map>
#include <string>
#include <regex>
#include <queue>
#include <algorithm>
#include <bits/stdc++.h>
// Multi-threads
#include <thread> // (C++ 11)

// Math
#define _USE_MATH_DEFINES
#include <cmath>
#include <cmath>

// Time
#include <chrono> // (C++ 11)
#include <ctime>


///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Include Windows libraries //////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <windows.h>
#include <Windows.h>
#include <direct.h>
#include "Shlwapi.h"



///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Include OpenCV libraries ///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
// OpenCV 3.0
//#include <cv.h>
#include <opencv2\highgui\highgui_c.h>
#include <opencv2\imgproc.hpp>
#include <opencv2\video.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\tracking.hpp>
#include <opencv2/xphoto/white_balance.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/dnn.hpp>
//#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <dlib/image_processing/frontal_face_detector.h>
//#include <dlib/image_processing/render_face_detections.h>
//#include <dlib/image_processing.h>
//#include <dlib/gui_widgets.h>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>



using namespace cv;
using namespace std;
using namespace cv::dnn;
using namespace cv::face;



const Scalar Red = Scalar(34, 68, 228);
const Scalar LightRed = Scalar(101, 74, 254);
const Scalar Green = Scalar(97, 204, 66);
const Scalar LightGreen = Scalar(0, 255, 127);
const Scalar ArmyGreen = Scalar(75, 170, 30);
const Scalar Blue = Scalar(225, 151, 60);
const Scalar DarkBlue = Scalar(255, 146, 1);
const Scalar Emerald = Scalar(152, 188, 53);
const Scalar White = Scalar(241, 240, 237);
const Scalar LightGray = Scalar(96, 71, 51);
const Scalar Gray = Scalar(50, 50, 50);
const Scalar Yellow = Scalar(1, 210, 254);