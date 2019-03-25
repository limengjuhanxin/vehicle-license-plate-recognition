#include<opencv2/features2d.hpp>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
vector<Rect> mserGetPlate(Mat srcImage)
{
	Mat gray,gray_neg;
	Mat hsi;
	cv::cvtColor(srcImage,hsi,CV_BGR2HSV);
	std::vector<Mat> channels;
	cv::split(hsi,channels);
	gray = channels[1];
	cvtColor(srcImage,gray,CV_BGR2GRAY);
	gray_neg = 255 -gray;

	cv::imshow("gray",gray);
	cv::imshow("gray_neg",gray_neg);

	std::vector<std::vector<cv::Point>> regContours;
	std::vector<std::vector<cv::Point>> charContours;

	cv::Ptr<cv::MSER> mesr1 = cv::MSER::create(2,10,5000,0.5,0.3);
	cv::Ptr<cv::MSER> mesr2 = cv::MSER::create(2,2,400,0.1,0.3);

	std::vector<cv::Rect>bboxes1;
	std::vector<cv::Rect>bboxes2;

	mesr1->detectRegions(gray,regContours,bboxes1);
	mesr2->detectRegions(gray_neg,charContours,bboxes2);

	cv::Mat mserMapMat = cv::Mat::zeros(srcImage.size(),CV_8UC1);
	cv::Mat mserNegMapMat = cv::Mat::zeros(srcImage.size(),CV_8UC1);
	
	for(int i=(int)regContours.size()-1;i>=0;i--)
	{
			const std::vector<cv::Point> &r = regContours[i];
			for(int j=0;j<(int)r.size();j++)
			{
					cv::Point pt = r[j];
					mserMapMat.at<unsigned char>(pt)=255;
			}
	}
	for(int i=(int)charContours.size()-1;i>=0;i--)
	{
		const std::vector<cv::Point> &r = charContours[i];
		for(int j=0;j<(int)r.size();j++)
		{
			cv::Point pt= r[j];
			mserNegMapMat.at<unsigned char>(pt)=255;
		}
	}

	cv::Mat mserResMat;
	mserResMat = mserMapMat & mserNegMapMat;
	cv::imshow("mserMapMat",mserMapMat);
	cv::imshow("mserNegMapMat",mserNegMapMat);
	cv::imshow("mserResMat",mserResMat);

	cv::Mat mserClosedMat;
 	cv::morphologyEx(mserResMat,mserClosedMat,cv::MORPH_CLOSE,cv::Mat::ones(1,20,CV_8UC1));
	cv::imshow("mserClosedMat",mserClosedMat);

	std::vector<std::vector<cv::Point>> plate_contours;
	cv::findContours(mserClosedMat,plate_contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));

	std::vector<cv::Rect> candidates;
	for(size_t i=0;i!=plate_contours.size();++i)
	{
		cv::Rect rect = cv::boundingRect(plate_contours[i]);
		double wh_ratio = rect.width /double(rect.height);
		if(rect.height>10 && wh_ratio> 2 && wh_ratio<5)
				candidates.push_back(rect);
	}
	return candidates;

}
int main(int argc,char **argv)
{
	if(argc!=2)
		return -1;

	Mat srcImg = imread(argv[1]);
	if(srcImg.empty())
	{
		return -1;
	}
	imshow("scr image",srcImg);
	
	std::vector<cv::Rect>candidates;
	candidates = mserGetPlate(srcImg);
	for(int i=0;i<candidates.size();i++)
	{
//			cv::imshow("rect",srcImg(candidates[i]));
//			cv::waitKey();
			cv::rectangle(srcImg,candidates[i],Scalar(0,255,0),2,2,0);
	}
	imshow("process image",srcImg);
	waitKey(0);
	return 0;
}
