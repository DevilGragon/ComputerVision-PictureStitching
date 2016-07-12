#include "stdafx.h"
#include "image_stitching.h"

/*
 * @function img_stitching Í¼ÏñÆ´½Ó
 * @return null
 */
void img_stitching()
{
	string srcFile[3] = {"1.jpg", "2.jpg", "3.jpg"};
	string dstFile = "stitching.jpg";
	vector<Mat> imgs;
	for (int i = 0;i<3;++i)
	{
		Mat img = imread(srcFile[i]);
		if (img.empty())
		{
			cout << "Can't read image '" << srcFile[i] << "'\n";
			exit(0);
		}
		imgs.push_back(img);
	}
	cout << "Please wait..." << endl;
	Mat pano;
	Stitcher stitcher = Stitcher::createDefault(false);
	Stitcher::Status status = stitcher.stitch(imgs, pano);
	if (status != Stitcher::OK)
	{
		cout << "Can't stitch images, error code=" << int(status) << endl;
		exit(1);
	}

	imwrite(dstFile, pano);
	namedWindow("Result");
	imshow("Result", pano);

	waitKey(0);

	destroyWindow("Result");
	system("pause");
}