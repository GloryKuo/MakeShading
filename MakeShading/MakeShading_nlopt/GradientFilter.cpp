#include "GradientFilter.h"
using namespace cv;

GradientFilter::GradientFilter()
{
	lambda = 0.1;
	tao = 0.5;
}
GradientFilter::~GradientFilter()
{
	delete opt;
}

bool GradientFilter::init(cv::Mat img)
{
	if(img.empty()) 
		return false;
	if(img.channels() != 1){
		cvtColor(img, img, CV_RGB2GRAY);
	}

	img.convertTo(m_inputImg, CV_64FC1, 1/255.0);	  // uint8 to double
	m_pixelWeights = getPixelWeight(m_inputImg);

	data.inputImg = m_inputImg;
	data.pixelWeight = m_pixelWeights;
	data.lambda = 0.1;
	data.count = 0;
		
	opt = new nlopt::opt(nlopt::LN_COBYLA, m_inputImg.rows*m_inputImg.cols);    /* algorithm and dimensionality */

	opt->set_min_objective(objFunc, &data);
	opt->set_stopval(1.0);
	//opt->set_maxtime(120);       /* set stopping criteria*/
	//std::vector<double> lb(m_inputImg.rows*m_inputImg.cols, 1e-8);
	//opt->set_lower_bounds(lb);

	return true;
}

double GradientFilter::objFunc(const std::vector<double> &x, std::vector<double> &grad, void* objFunc_data)
{
	ObjFunc_data *data =  reinterpret_cast<ObjFunc_data*>(objFunc_data);
	
	Mat inputImg = data->inputImg;
	Mat currentImg_1D(x);
	Mat currentImg = currentImg_1D.reshape(1, inputImg.rows);

	Mat grad_currImg = getGradient(currentImg);
	/* check iteration */
	/*Mat grad_currImg_8U;
	grad_currImg.convertTo(grad_currImg_8U, CV_8UC1, 255);
	char *s = new char[30];
	sprintf(s, "Debug/current/%d.jpg", data->count);
	imwrite(s, grad_currImg_8U);
	delete [] s;*/
	////////////////////////

	static Mat grad_inputImg, clipped_inputImg;
	if(!grad_inputImg.data){          // 只做第一次
		grad_inputImg = getGradient(inputImg);
		clipped_inputImg = gradClipping(grad_inputImg);
		//imshow("clipped image", clipped_inputImg);
	}

	if(!grad.empty()){
		Mat secondOrder_grad = getGradient(grad_currImg);
		Mat grad_forFunc, grad_forFunc1D, a, b, b2;
		a = 2*secondOrder_grad.mul(grad_currImg - clipped_inputImg);
		b2 = (data->lambda)*((data->pixelWeight).mul(currentImg - inputImg));
		b = 2*grad_currImg.mul(b2);
		grad_forFunc =  a + b; 
		grad_forFunc1D = grad_forFunc.reshape(1, inputImg.rows*inputImg.cols);
		grad_forFunc1D.copyTo(grad);
	}


	/* 計算cost */
	Mat a = grad_currImg - clipped_inputImg;
	pow(a, 2, a);
	Mat b = currentImg - inputImg;
	pow(b, 2, b);
	b = b.mul(data->pixelWeight);
	
	Mat result = a + (data->lambda)*b;
	double cost = cv::sum(result)[0];
	/////////////////////
	std::cout<<"iteration "<<++(data->count)<<": cost = "<<cost<<std::endl;
	return cost;
}

double GradientFilter::constraint(const std::vector<double> &x, std::vector<double> &grad, void* cons_data)
{
	return 0.0;
}

Mat GradientFilter::optimize()
{
	double minCost;    /* the minimum objective value, upon return */
	std::vector<double> img_v;
	//Mat img1D = Mat::zeros(Size(1, m_inputImg.rows*m_inputImg.cols), CV_64FC1);
	Mat img1D = m_inputImg.reshape(1, m_inputImg.rows*m_inputImg.cols);
	img_v.reserve(m_inputImg.rows*m_inputImg.cols);
	img1D.copyTo(img_v);          //2D Mat covert to 1D vector	
	
	nlopt::result result = opt->optimize(img_v, minCost);
	std::cout<<"result = "<<result<<std::endl;
	
	Mat img_opt1D(img_v);
	Mat img_opt = img_opt1D.reshape(1, m_inputImg.rows);
	std::cout<<"Count = "<< data.count <<std::endl;
	std::cout<<"Cost  = "<< minCost <<std::endl;
	
	Mat residual = abs(m_inputImg - img_opt);
	imshow("residual", residual);
	return img_opt.clone();
}

Mat GradientFilter::getGradient(Mat src )
{
    Mat src_gray, grad;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_64F;

    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

	if(src.channels() != 1)
		cvtColor( src, src_gray, CV_RGB2GRAY );
	else
		src_gray = src;
	
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    //convertScaleAbs( grad_x, abs_grad_x );
	abs_grad_x = abs(grad_x);
	
    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    //convertScaleAbs( grad_y, abs_grad_y );
	abs_grad_y = abs(grad_y);

    //addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
	
	grad = abs_grad_x + abs_grad_y;

	//imshow("gradient", grad);
	return grad;
}

Mat GradientFilter::gradClipping( Mat gradient )
{
	Mat grad_Clipped, grad_f, clipped_d;

	/* 取平均值作為threshold */
	Mat m1, m2;
	reduce(gradient, m1, 0, CV_REDUCE_AVG);
	reduce(m1, m2, 1, CV_REDUCE_AVG);
	double mean = m2.at<double>(0);
	std::cout<<"The mean of gradient = "<<mean<<std::endl;
	gradient.convertTo(grad_f, CV_32FC1);
	threshold(grad_f, grad_Clipped, mean, 1.0, THRESH_TOZERO_INV);   //threshold只能接受uint8 or float
	grad_Clipped.convertTo(clipped_d, CV_64FC1);

	//std::cout<<grad_f.row(0);
	//imshow("clipped image", grad_Clipped);
	return clipped_d;

}

Mat GradientFilter::getPixelWeight( Mat img )
{
	Mat img_G, Ones, specH;
	GaussianBlur(img, img_G, Size(101,101), 0, 0, BORDER_DEFAULT);
	Ones = Mat::ones(img.size(), img.type());
	Mat result = Ones - abs(img - img_G);
	specH = Ones.clone();      // zero when the pixel is classified as a specular highlight
	result = result.mul(specH);
	Mat result_double;
	result.convertTo(result_double, CV_64FC1);
	
	//imshow("weight", result_double);
	return result_double;
}