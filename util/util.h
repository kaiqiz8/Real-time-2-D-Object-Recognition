/*
    Kaiqi Zhang
	Spring 2024
	CS 5330 Project 3

    Header file for util functions
*/

#ifndef UTIL_H
#define UTIL_H

#include <opencv2/opencv.hpp> 
int blur5x5_2( cv::Mat &src, cv::Mat &dst);
int threshold(cv::Mat &src, cv::Mat &dst, int threshold);
int cleanup(cv::Mat &src, cv::Mat &dst);
int segmentation(cv::Mat &src, cv::Mat &top5RegionColorMap, cv::Mat &region_map_reduced);
cv::Mat computeFeaturesAndDisplay(const cv::Mat &regionMap, const cv::Mat &src, int regionID, std::vector<std::vector<float>> &features);
std::string kNN(std::vector<float> euclideanDistanceSums, std::vector<char *> labels, int k);

// Define a comparator for the priority queue
struct compare {
    bool operator()(const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return a.second < b.second; // Change to > for ascending order
    }
};
#endif
