#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <iostream>
#include "nlopt.hpp"
#include "GradientFilter.h"

using namespace cv;

int main( int argc, char** argv ){
    Mat src = imread( "../input/input2.png", 0 );
    if( !src.data ){ 
        return -1; 
    }
	resize(src, src, Size(0,0), 0.1, 0.1);
	imshow("source", src);
///////////////////////////////////////////////////////////////////////////////////
	/* Example of nlopt*/
	GradientFilter demo;
	demo.init(src);
	Mat shading = demo.optimize();
	imshow("shading", shading);
	Mat shading_8U;
	shading.convertTo(shading_8U, CV_8UC1, 255);
	imwrite("output/shading.jpg", shading_8U);

/////////////////////////////////////////////////////////////////////////////////////

    waitKey();

	

    return 0;
}