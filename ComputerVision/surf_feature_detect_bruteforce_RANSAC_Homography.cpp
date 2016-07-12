#include "stdafx.h"
#include "surf_feature_detect_bruteforce_RANSAC_Homography.h"

/*
 * @function surf_feature_detect_RANSAC SURF������ȡ��ƥ�䣬RANSAC����������Լ�������
 * @return null
 * @method SURF feature detector
 * @method SURF descriptor
 * @method findFundamentalMat RANSAC����ȥ��
 * @method findHomography Ѱ��͸�ӱ任����
 */
void surf_feature_detect_bruteforce_RANSAC_Homography(Mat SourceImg, Mat SceneImg, Mat imageMatches, char* string)
{
	vector<KeyPoint> keyPoints1, keyPoints2;
	SurfFeatureDetector detector(400);
	detector.detect(SourceImg, keyPoints1); //��עԭͼ������
	detector.detect(SceneImg, keyPoints2); //��ע����ͼ������

	SurfDescriptorExtractor surfDesc;
	Mat SourceImgDescriptor, SceneImgDescriptor;
	surfDesc.compute(SourceImg, keyPoints1, SourceImgDescriptor); //����ԭͼsurf������
	surfDesc.compute(SceneImg, keyPoints2, SceneImgDescriptor); //��������ͼsurf������

	//��������ͼƬ��������ƥ����
	BruteForceMatcher<L2<float>>matcher;
	vector<DMatch> matches;
	matcher.match(SourceImgDescriptor, SceneImgDescriptor, matches);
	std::nth_element(matches.begin(), matches.begin() + 29 ,matches.end());
	matches.erase(matches.begin() + 30, matches.end());

	//FLANNƥ�����㷨
	//vector<DMatch> matches;
	//DescriptorMatcher *pMatcher = new FlannBasedMatcher;
	//pMatcher->match(SourceImgDescriptor, SceneImgDescriptor, matches);
	//delete pMatcher;

	//keyPoints1 ͼ1��ȡ�Ĺؼ���
	//keyPoints2 ͼ2��ȡ�Ĺؼ���
	//matches �ؼ����ƥ��
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

	// �����ڵ�
	vector<Point2f> m_LeftInlier;
	vector<Point2f> m_RightInlier;
	vector<DMatch> m_InlierMatches;

	// ���������������ڱ����ڵ��ƥ���ϵ
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

	// ���ڵ�ת��ΪdrawMatches����ʹ�õĸ�ʽ
	vector<KeyPoint> key1(InlinerCount);
	vector<KeyPoint> key2(InlinerCount);
	KeyPoint::convert(m_LeftInlier, key1);
	KeyPoint::convert(m_RightInlier, key2);

	//��ʾ����F������ڵ�ƥ��
	drawMatches(SourceImg, key1, SceneImg, key2, m_InlierMatches, imageMatches);
	//drawKeypoints(SourceImg, key1, SceneImg, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	
	vector<Point2f> obj;
	vector<Point2f> scene;
	for(unsigned int i = 0; i < m_InlierMatches.size(); i++)
	{
		obj.push_back(key1[m_InlierMatches[i].queryIdx].pt); //��ѯͼ�񣬼�Ŀ��ͼ�����������
		scene.push_back(key2[m_InlierMatches[i].trainIdx].pt); //ģ��ͼ�񣬼�����ͼ�����������
	}
	//���任����
	//����ͬgetPerspectiveTransform����������ԭʼͼ��ͱ任֮��ͼ���ж�Ӧ��4���㣬Ȼ������任ӳ���ϵ�����任����
	//findHomographyֱ��ʹ��͸��ƽ�����ұ任��ʽ
	Mat H = findHomography(obj, scene, CV_RANSAC);
	vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0);
	obj_corners[1] = cvPoint(SourceImg.cols, 0);
	obj_corners[2] = cvPoint(SourceImg.cols, SourceImg.rows);
	obj_corners[3] = cvPoint(0, SourceImg.rows);
	vector<Point2f> scene_corners(4);
	//͸�ӱ任����ͼƬͶӰ��һ���µ���ƽ��
	//��������õı任����
	perspectiveTransform(obj_corners, scene_corners, H);

	line(imageMatches, scene_corners[0] + Point2f(SourceImg.cols, 0), scene_corners[1] + Point2f(SourceImg.cols, 0), Scalar(0, 0, 255), 4);
	line(imageMatches, scene_corners[1] + Point2f(SourceImg.cols, 0), scene_corners[2] + Point2f(SourceImg.cols, 0), Scalar(0, 0, 255), 4);
	line(imageMatches, scene_corners[2] + Point2f(SourceImg.cols, 0), scene_corners[3] + Point2f(SourceImg.cols, 0), Scalar(0, 0, 255), 4);
	line(imageMatches, scene_corners[3] + Point2f(SourceImg.cols, 0), scene_corners[0] + Point2f(SourceImg.cols, 0), Scalar(0, 0, 255), 4);
	imshow(string, imageMatches);

	imwrite("feature_detect.jpg", imageMatches);
}