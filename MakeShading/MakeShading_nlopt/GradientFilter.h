#ifndef GRADIENTFILTER_H
#define GRADIENTFILTER_H
#include "opencv2/opencv.hpp"
#include "nlopt.hpp"

class ObjFunc_data
{
public:
	cv::Mat inputImg;
	cv::Mat pixelWeight;
	double lambda;
	int count;
};

class GradientFilter
{
public:
	GradientFilter();
	~GradientFilter();
	bool init(cv::Mat img);
	cv::Mat optimize();
private:
	double static objFunc(const std::vector<double> &x, std::vector<double> &grad, void* objFunc_data);
	double static constraint(const std::vector<double> &x, std::vector<double> &grad, void* cons_data);
	cv::Mat static getGradient( cv::Mat src );
	cv::Mat static gradClipping( cv::Mat gradient );
	cv::Mat getPixelWeight( cv::Mat img );

	double lambda;
	double tao;
	cv::Mat m_inputImg;
	cv::Mat m_pixelWeights;
	nlopt::opt *opt;
	ObjFunc_data data;
};


#endif