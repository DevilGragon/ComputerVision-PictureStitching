#include "stdafx.h"
#include "surf_feature_detect_bruteforce_RANSAC_Homography_stitching_underbuild.h"

/*
 * @function surf_feature_detect_RANSAC SURF������ȡ��ƥ�䣬RANSAC����������Լ�������
 * @return null
 * @method SURF feature detector
 * @method SURF descriptor
 * @method findFundamentalMat RANSAC����ȥ��
 * @method findHomography Ѱ��͸�ӱ任����
 */
void surf_feature_detect_bruteforce_RANSAC_Homography_stitching_underbuild(Mat SourceImg, Mat SceneImg, Mat stitching_img, Mat imageMatches, char* string)
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
	imshow(string, imageMatches);
	//waitKey(0);

	vector<Point2f> obj;
	vector<Point2f> scene;
	for(int i = 0; i < m_InlierMatches.size(); i++)
	{
		obj.push_back(key1[m_InlierMatches[i].queryIdx].pt);
		scene.push_back(key2[m_InlierMatches[i].trainIdx].pt);
	}
	Mat H = findHomography(obj, scene, CV_RANSAC); //HΪ�任����
	vector<Point2f> points, points_trans;
	for(int j = 0; j < SceneImg.rows; j++)
	{
		for(int i = 0; i < SceneImg.cols; i++)
		{
			points.push_back(Point2f(j ,i));
		}
	}
	perspectiveTransform(points, points_trans, H);
	Mat SceneImg_trans = Mat::zeros(2 * SceneImg.rows, 2 * SceneImg.cols, CV_8UC3);
	int count = 0;
	printf("Please wait..");
	for(int i = 0; i < SceneImg.rows; i++)
	{
		//printf("��%d��\n", i);
		uchar* p = SceneImg.ptr<uchar>(i);
		for(int j = 0; j < SceneImg.cols; j++)
		{
			//printf("��%d��", j);
			int y = abs(points_trans[count].y);
			int x = abs(points_trans[count].x);
			//printf("y = %d\n", y);
			uchar* t = SceneImg_trans.ptr<uchar>(y);
			t[x * 3] = p[j * 3];
			t[x * 3 + 1] = p[j * 3 + 1];
			t[x * 3    + 2] = p[j * 3 + 2];
			count++;
		}
		if((i % 100) == 0)
		{
			printf(".");
		}
	}
	imshow("SceneImg_trans", SceneImg_trans);
	waitKey(0);

	//key1Ϊ��ͼ������
	//key2Ϊ��ͼ������
	int Source_min = 0;
	int Scene_min = 0;
	int max = 0; //�����С���������
	float key_Source_min = key1[0].pt.x;
	float key_Scene_min = key2[0].pt.x;
	float key_max = key2[0].pt.x;
	for(int i = 0; i < 30; i++)
	{
		if(key1[i].pt.x < key_Source_min)
		{
			Source_min = i;
			key_Source_min = key1[i].pt.x;
		}
		if(key2[i].pt.x > key_max)
		{
			max = i;
			key_max = key2[i].pt.x;
		}
		if(key2[i].pt.x < key_Scene_min)
		{
			Scene_min = i;
			key_Scene_min = key2[i].pt.x;
		}
	}
	//SourceImg & SceneImg
	int SourceImg_min;
	int SceneImg_max;
	int SceneImg_min;
	SourceImg_min = key1[Source_min].pt.x;
	SceneImg_min = key2[Scene_min].pt.x;
	SceneImg_max = key2[max].pt.x;
	
	int type = SourceImg.type();
	stitching_img.create(SourceImg.rows, SourceImg_min + SceneImg.cols - SceneImg_min, type);

	for(int j1 = 0; j1 < SourceImg.rows; j1++)
	{
		for(int i1 = 0; i1 < SourceImg_min; i1++)
		{
			stitching_img.at<Vec3b>(j1, i1)[0] = SourceImg.at<Vec3b>(j1, i1)[0];
			stitching_img.at<Vec3b>(j1, i1)[1] = SourceImg.at<Vec3b>(j1, i1)[1];
			stitching_img.at<Vec3b>(j1, i1)[2] = SourceImg.at<Vec3b>(j1, i1)[2];
		}
	}
	for(int j2 = 0; j2 < SceneImg.rows; j2++)
	{
		for(int i2 = SceneImg_min; i2 < SceneImg_max; i2++)
		{
			stitching_img.at<Vec3b>(j2, SourceImg_min + i2 - SceneImg_min)[0] = SceneImg.at<Vec3b>(j2, i2)[0];
			stitching_img.at<Vec3b>(j2, SourceImg_min + i2 - SceneImg_min)[1] = SceneImg.at<Vec3b>(j2, i2)[1];
			stitching_img.at<Vec3b>(j2, SourceImg_min + i2 - SceneImg_min)[2] = SceneImg.at<Vec3b>(j2, i2)[2];
		}
	}
	for(int j3 = 0; j3 < SceneImg.rows; j3++)
	{
		for(int i3 = 0; i3 < SceneImg.cols - SceneImg_max; i3++)
		{
			stitching_img.at<Vec3b>(j3, SourceImg_min + SceneImg_max - SceneImg_min + i3)[0] = SceneImg.at<Vec3b>(j3, SceneImg_max + i3)[0];
			stitching_img.at<Vec3b>(j3, SourceImg_min + SceneImg_max - SceneImg_min + i3)[1] = SceneImg.at<Vec3b>(j3, SceneImg_max + i3)[1];
			stitching_img.at<Vec3b>(j3, SourceImg_min + SceneImg_max - SceneImg_min + i3)[2] = SceneImg.at<Vec3b>(j3, SceneImg_max + i3)[2];
		}
	}
	imshow("my stitching", stitching_img);
	imwrite("stitching.jpg", stitching_img);
	waitKey(0);
}