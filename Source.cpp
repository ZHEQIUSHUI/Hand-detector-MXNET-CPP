#define _CRT_SECURE_NO_WARNINGS

#pragma comment(lib,"opencv_world410.lib")
#pragma comment(lib,"libmxnet.lib")

#include <opencv2\opencv.hpp>
#include <iostream>
#include "HandDetector.h"
using namespace std;
using namespace cv;
int main()
{
	HandDetector detector(true);
	detector.Loadmodel("E:\\PyCode", "faster_rcnn_resnet50_v1b_custom");//ssd_512_mobilenet1.0_custom


	VideoCapture video(0);
	Mat src;
	while (true)
	{
		video >> src;
		if (!src.data)
		{
			cout << "video data is emyty" << endl;
			return -1;
		}
		vector<mx_float> scores;
		vector<Rect> bboxes;
		detector.detect(src, scores, bboxes,1.0);

		for (size_t i = 0; i < bboxes.size(); i++)
		{
			rectangle(src, bboxes[i], Scalar(0, 255, 0), 1);
			putText(src, to_string(scores[i]), bboxes[i].br(), FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 255, 0), 1);
		}
		imshow("result", src);
		if (waitKey(1) == 27)
		{
			break;
		}
	}
	return 0;

}