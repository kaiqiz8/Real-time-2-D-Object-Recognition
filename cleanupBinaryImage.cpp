#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "util/util.h"
#include "util/kmeans.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

int main(int argc, char *argv[]) {
    cv::Mat src;
    cv::Mat binaryImage;
    cv::Mat clean;
    char filename[256];

    //check if enough command line arguments
	if(argc < 2) {
		printf("usage: %s < image filename>\n", argv[0]);
		exit(-1);
	}
	strcpy( filename, argv[1]);

	//read the image 
	src = cv::imread( filename ); 
    blur5x5_2(src, src);
    threshold(src, binaryImage, 2);
    cleanup(binaryImage, clean);

    cv::imshow("Binary Image", binaryImage);
    cv::imshow("Cleaned Image", clean);
    cv::imshow("Original Image", src);
    cv::waitKey(0);
    cv::destroyWindow( filename );
	printf("Terminating\n");
    return 0;
}