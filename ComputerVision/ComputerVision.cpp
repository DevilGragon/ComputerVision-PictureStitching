// ComputerVision.cpp : 定义控制台应用程序的入口点。
//

// DIPhomework.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"
#include "ComputerVision.h"
#include "fast_feature_detect.h"
#include "image_stitching.h"
#include "surf_feature_detect_bruteforce_RANSAC_Homography.h"
#include "surf_feature_detect_flann_Homography.h"
#include "surf_feature_detect_bruteforce_RANSAC_Homography_stitching.h"

void main()
{
 	char* string1 = "Matches1";
	char* string2 = "Matches2";
	//fast_feature_detect();
	//Mat SourceImg1 = imread("obj.jpg", 1);
	//Mat SourceImg2 = imread("scene.jpg", 1);
	Mat SourceImg3 = imread("obj.jpg", 1);
	Mat SourceImg4 = imread("scene.jpg", 1);
	Mat imageMatches, stitching_img;
	//surf_feature_detect_Homography(SourceImg1, SourceImg2, imageMatches1, string1);
	//surf_feature_detect_bruteforce_RANSAC_Homography_stitching(SourceImg3, SourceImg4, stitching_img, imageMatches, string2);
	surf_feature_detect_flann_Homography(SourceImg3, SourceImg4, imageMatches, string2);
	//img_stitching();
	waitKey(0);
}
