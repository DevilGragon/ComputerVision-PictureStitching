#include "stdafx.h"
#include "surf_feature_detect_bruteforce_RANSAC_Homography.h"

/*
 * @function surf_feature_detect_RANSAC SURF特征提取及匹配，RANSAC错误点消除以及物体标记
 * @return null
 * @method SURF feature detector
 * @method SURF descriptor
 * @method findFundamentalMat RANSAC错误去除
 * @method findHomography 寻找透视变换矩阵
 */
void surf_feature_detect_bruteforce_RANSAC_Homography(Mat SourceImg, Mat SceneImg, Mat imageMatches, char* string)
{
	vector<KeyPoint> keyPoints1, keyPoints2;
	SurfFeatureDetector detector(400);
	detector.detect(SourceImg, keyPoints1); //标注原图特征点
	detector.detect(SceneImg, keyPoints2); //标注场景图特征点

	SurfDescriptorExtractor surfDesc;
	Mat SourceImgDescriptor, SceneImgDescriptor;
	surfDesc.compute(SourceImg, keyPoints1, SourceImgDescriptor); //描述原图surf特征点
	surfDesc.compute(SceneImg, keyPoints2, SceneImgDescriptor); //描述场景图surf特征点

	//计算两张图片的特征点匹配数
	BruteForceMatcher<L2<float>>matcher;
	vector<DMatch> matches;
	matcher.match(SourceImgDescriptor, SceneImgDescriptor, matches);
	std::nth_element(matches.begin(), matches.begin() + 29 ,matches.end());
	matches.erase(matches.begin() + 30, matches.end());

	//FLANN匹配检测算法
	//vector<DMatch> matches;
	//DescriptorMatcher *pMatcher = new FlannBasedMatcher;
	//pMatcher->match(SourceImgDescriptor, SceneImgDescriptor, matches);
	//delete pMatcher;

	//keyPoints1 图1提取的关键点
	//keyPoints2 图2提取的关键点
	//matches 关键点的匹配
	int ptCount = (int)matches.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);
	Point2f pt;
	for(int i = 0; i < ptCount; i++)
	{
		pt = keyPoints1[matches[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = keyPoints2[matches[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}
	Mat m_Fundamental;
	vector<uchar> m_RANSACStatus;
	m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);
	int OutlinerCount = 0;
	for(int i = 0; i < ptCount; i++)
	{
		if(m_RANSACStatus[i] == 0)
		{
			OutlinerCount++;
		}
	}

	// 计算内点
	vector<Point2f> m_LeftInlier;
	vector<Point2f> m_RightInlier;
	vector<DMatch> m_InlierMatches;

	// 上面三个变量用于保存内点和匹配关系
	int InlinerCount = ptCount - OutlinerCount;
	m_InlierMatches.resize(InlinerCount);
	m_LeftInlier.resize(InlinerCount);
	m_RightInlier.resize(InlinerCount);
	InlinerCount = 0;
	for (int i=0; i<ptCount; i++)
	{
		if (m_RANSACStatus[i] != 0)
		{
			m_LeftInlier[InlinerCount].x = p1.at<float>(i, 0);
			m_LeftInlier[InlinerCount].y = p1.at<float>(i, 1);
			m_RightInlier[InlinerCount].x = p2.at<float>(i, 0);
			m_RightInlier[InlinerCount].y = p2.at<float>(i, 1);
			m_InlierMatches[InlinerCount].queryIdx = InlinerCount;
			m_InlierMatches[InlinerCount].trainIdx = InlinerCount;
			InlinerCount++;
		}
	}

	// 把内点转换为drawMatches可以使用的格式
	vector<KeyPoint> key1(InlinerCount);
	vector<KeyPoint> key2(InlinerCount);
	KeyPoint::convert(m_LeftInlier, key1);
	KeyPoint::convert(m_RightInlier, key2);

	//显示计算F过后的内点匹配
	drawMatches(SourceImg, key1, SceneImg, key2, m_InlierMatches, imageMatches);
	//drawKeypoints(SourceImg, key1, SceneImg, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	
	vector<Point2f> obj;
	vector<Point2f> scene;
	for(unsigned int i = 0; i < m_InlierMatches.size(); i++)
	{
		obj.push_back(key1[m_InlierMatches[i].queryIdx].pt); //查询图像，即目标图像的特征描述
		scene.push_back(key2[m_InlierMatches[i].trainIdx].pt); //模版图像，即场景图像的特征描述
	}
	//求解变换矩阵
	//作用同getPerspectiveTransform函数，输入原始图像和变换之后图像中对应的4个点，然后建立起变换映射关系，即变换矩阵
	//findHomography直接使用透视平面来找变换公式
	Mat H = findHomography(obj, scene, CV_RANSAC);
	vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0);
	obj_corners[1] = cvPoint(SourceImg.cols, 0);
	obj_corners[2] = cvPoint(SourceImg.cols, SourceImg.rows);
	obj_corners[3] = cvPoint(0, SourceImg.rows);
	vector<Point2f> scene_corners(4);
	//透视变换，将图片投影到一个新的视平面
	//根据以求得的变换矩阵
	perspectiveTransform(obj_corners, scene_corners, H);

	line(imageMatches, scene_corners[0] + Point2f(SourceImg.cols, 0), scene_corners[1] + Point2f(SourceImg.cols, 0), Scalar(0, 0, 255), 4);
	line(imageMatches, scene_corners[1] + Point2f(SourceImg.cols, 0), scene_corners[2] + Point2f(SourceImg.cols, 0), Scalar(0, 0, 255), 4);
	line(imageMatches, scene_corners[2] + Point2f(SourceImg.cols, 0), scene_corners[3] + Point2f(SourceImg.cols, 0), Scalar(0, 0, 255), 4);
	line(imageMatches, scene_corners[3] + Point2f(SourceImg.cols, 0), scene_corners[0] + Point2f(SourceImg.cols, 0), Scalar(0, 0, 255), 4);
	imshow(string, imageMatches);

	imwrite("feature_detect.jpg", imageMatches);
}