/*
    Kaiqi Zhang
	Spring 2024
	CS 5330 Project 3

    Connect to the camera, threshold and display the video stream
*/
#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "util/kmeans.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include "util/util.h"

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;
    capdev = new cv::VideoCapture(0); // open the default camera

    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }
    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                        (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    capdev->set(cv::CAP_PROP_CONVERT_RGB, true);
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame;

    for(;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream

        blur5x5_2(frame, frame);
        cv::Mat binaryImage;
        threshold(frame, frame, 2);

        cv::imshow("Video", frame);
        char key = cv::waitKey(10);
        if( key == 'q') {
            break;
        }
    }

    delete capdev;
    return(0);
}
