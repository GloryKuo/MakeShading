#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <iostream>
#include "nlopt.hpp"
#include "GradientFilter.h"

using namespace cv;

int main( int argc, char** argv ){
    Mat src = imread( "../input/input7.jpg", 0 );
    if( !src.data ){
		std::cout<<"Can not load image!"<<std::endl;
        return -1; 
    }
	imshow("source", src);
///////////////////////////////////////////////////////////////////////////////////
	/* Example of nlopt*/
	GradientFilter demo;
	demo.init(src);
	Mat shading = demo.optimize();
	Mat shading_8U;
	shading.convertTo(shading_8U, CV_8UC1, 255);
	resize(shading_8U, shading_8U, Size(src.cols,src.rows));
	imwrite("../output/shading.jpg", shading_8U);

/////////////////////////////////////////////////////////////////////////////////////

    waitKey();

	

    return 0;
}