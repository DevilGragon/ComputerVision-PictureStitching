#include "stdafx.h"
#include "surf_feature_detect_flann_Homography.h"

/*
 * @function surf_feature_detect_Homography SURF特征提取及匹配，Homography透视变换
 * @return null
 * @method 
 */
void surf_feature_detect_flann_Homography(Mat SourceImg, Mat SceneImg, Mat imageMatches, char* string)
{

	vector<KeyPoint> keyPoints1, keyPoints2;
	SurfFeatureDetector detector(400);
	detector.detect(SourceImg, keyPoints1); //标注原图特征点
	detector.detect(SceneImg, keyPoints2); //标注场景图特征点

	SurfDescriptorExtractor surfDesc;
	Mat SourceImgDescriptor, SceneImgDescriptor;
	surfDesc.compute(SourceImg, keyPoints1, SourceImgDescriptor); //描述原图surf特征点
	surfDesc.compute(SceneImg, keyPoints2, SceneImgDescriptor); //描述场景图surf特征点

	FlannBasedMatcher matcher;
	vector<DMatch> matches;
	matcher.match(SourceImgDescriptor, SceneImgDescriptor, matches);

	double max_dist = 0;
	double min_dist = 100;
	for(int i = 0; i < SourceImgDescriptor.rows; i++)
	{
		double dist = matches[i].distance;
		if(dist < min_dist)
			min_dist = dist;
		if(dist > max_dist)
			max_dist = dist;
	}
	printf("--Max dist : %f \n", max_dist);
	printf("--Min dist : %f \n", min_dist);

	vector<DMatch> good_matches;
	for(int i = 0; i < SourceImgDescriptor.rows; i++)
	{
		if(matches[i].distance < 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	drawMatches(SourceImg, keyPoints1, SceneImg, keyPoints2, good_matches, imageMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow(string, imageMatches);

	vector<Point2f> obj;
	vector<Point2f> scene;
	for(unsigned int i = 0; i < good_matches.size(); i++)
	{
		obj.push_back(keyPoints1[good_matches[i].queryIdx].pt);
		scene.push_back(keyPoints2[good_matches[i].trainIdx].pt);
	}
	Mat H = findHomography(obj, scene, CV_RANSAC);
	vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0);
	obj_corners[1] = cvPoint(SourceImg.cols, 0);
	obj_corners[2] = cvPoint(SourceImg.cols, SourceImg.rows);
	obj_corners[3] = cvPoint(0, SourceImg.rows);
	vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H);

	line(imageMatches, scene_corners[0] + Point2f(SourceImg.cols, 0), scene_corners[1] + Point2f(SourceImg.cols, 0), Scalar(0, 0, 255), 4);
	line(imageMatches, scene_corners[1] + Point2f(SourceImg.cols, 0), scene_corners[2] + Point2f(SourceImg.cols, 0), Scalar(0, 0, 255), 4);
	line(imageMatches, scene_corners[2] + Point2f(SourceImg.cols, 0), scene_corners[3] + Point2f(SourceImg.cols, 0), Scalar(0, 0, 255), 4);
	line(imageMatches, scene_corners[3] + Point2f(SourceImg.cols, 0), scene_corners[0] + Point2f(SourceImg.cols, 0), Scalar(0, 0, 255), 4);
	imshow(string, imageMatches);
}