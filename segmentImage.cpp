/*
    Kaiqi Zhang
	Spring 2024
	CS 5330 Project 3

    Segment the image and display the region map
*/
#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "util/util.h"
#include "util/kmeans.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <stack>
#include <utility> // For std::pair
#include "util/csv_util.h"

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

    cv::Mat region_map_color;
    cv::Mat region_map_reduced = cv::Mat::ones(src.rows, src.cols, CV_8U) * 0;;
    region_map_color = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
    segmentation(clean, region_map_color, region_map_reduced);

    cv::imshow("Binary Image", binaryImage);
    cv::imshow("Cleaned Image", clean);
    cv::imshow("Original Image", src);
    cv::imshow("Region map color", region_map_color);
    cv::waitKey(0);
    cv::destroyWindow( filename );
	printf("Terminating\n");
    return 0;
}