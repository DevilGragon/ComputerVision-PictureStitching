#include "stdafx.h"
#include "fast_feature_detect.h"

/*
 * @function fast_feature_detect FASTÃÿ’˜Ã·»°
 * @return null
 */
void fast_feature_detect()
{
	Mat src_img, keyPoints_img;
	src_img = imread("2222.png", 1);
	cvtColor(src_img, keyPoints_img, CV_BGR2GRAY);
	vector<KeyPoint> keyPoints;
	printf("Please wait...");
	FAST(keyPoints_img, keyPoints, 40);
	drawKeypoints(keyPoints_img, keyPoints, src_img, Scalar::all(-1), DrawMatchesFlags::DRAW_OVER_OUTIMG);
	imshow("Feature Detect", src_img);
	//imwrite("FastFeatureDetection.jpg", src_img);
	waitKey(0);
}