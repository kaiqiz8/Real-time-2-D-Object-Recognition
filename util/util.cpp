/*
    Kaiqi Zhang
	Spring 2024
	CS 5330 Project 3

    Implementation of util functions
*/

#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp> 
#include "kmeans.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <stack>
#include <utility>
#include "util.h"

//implementation of 5x5 blur, src is the input image, dst is the output image
int blur5x5_2( cv::Mat &src, cv::Mat &dst ) {
    src.copyTo(dst);
    for (int i = 2; i < src.rows - 2; i++){
        cv::Vec3b *ptrup2 = src.ptr<cv::Vec3b>(i-2);
        cv::Vec3b *ptrup1 = src.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *ptrmd = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *ptrdn1 = src.ptr<cv::Vec3b>(i+1);
        cv::Vec3b *ptrdn2 = src.ptr<cv::Vec3b>(i+2);
        cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);

        for (int j = 2; j < src.rows - 2; j++) {
            for (int k = 0; k < src.channels(); k++) {
                int sum = ptrup2[j - 2][k] + 2 * ptrup2[j - 1][k] + 4 * ptrup2[j][k] + 2 * ptrup2[j + 1][k] + ptrup2[j + 2][k]
                        + 2 * ptrup1[j - 2][k] + 4 * ptrup1[j - 1][k] + 8 * ptrup1[j][k] + 4 * ptrup1[j + 1][k] + 2 * ptrup1[j + 2][k]
                        + 4 * ptrmd[j - 2][k] + 8 * ptrmd[j - 1][k] + 16 * ptrmd[j][k] + 8 * ptrmd[j + 1][k] + 4* ptrmd[j + 2][k]
                        + 2 * ptrdn1[j - 2][k] + 4 * ptrdn1[j - 1][k] + 8 * ptrdn1[j][k] + 4 * ptrdn1[j + 1][k] + 2* ptrdn1[j + 2][k]
                        + ptrdn2[j - 2][k] + 2 * ptrdn2[j - 1][k] + 4 * ptrdn2[j][k] + 2 * ptrdn2[j + 1][k] + ptrdn2[j + 2][k];
                sum /= 100;
                dptr[j][k] = sum;
                
            }
        }
    }
    return(0);
}

//implementation of thresholding, src is the input image, dst is the output image, threshold is the threshold value
int threshold(cv::Mat &src, cv::Mat &dst, int threshold) {
    cv::Mat hsvImage;
    cv::cvtColor(src, hsvImage, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsvChannels;
    split(hsvImage, hsvChannels);

    // hsvChannels[1] *= 0.5;
    hsvChannels[1] *= 1.5;
    merge(hsvChannels, hsvImage);
    // Mat desaturatedImage;
    cvtColor(hsvImage, src, cv::COLOR_HSV2BGR);

    int totalPixels = src.rows * src.cols;
    int sampleSize = totalPixels ;

    // Vector to hold sampled pixels
    std::vector<cv::Vec3b> sampledPixels;
    sampledPixels.reserve(sampleSize);

    std::vector<int> indices(totalPixels);

    // Sample pixels
    for (int i = 0; i < sampleSize; ++i) {
        int idx = indices[i];
        int row = idx / src.cols;
        int col = idx % src.cols;
        sampledPixels.push_back(src.at<cv::Vec3b>(row, col));
    }

    int tcolors = 2;
    std::vector<cv::Vec3b> means;
    int *labels = new int[sampledPixels.size()];
    if (kmeans(sampledPixels, means, labels, tcolors) < 0) {
        printf("kmeans failed\n");
        return -1;
    }

    // Calculate the threshold value
    double thresholdB;
    double thresholdG;
    double thresholdR;
    for (int i = 0; i < threshold; i++) {
        thresholdB += means[i][0];
        thresholdG += means[i][1];
        thresholdR += means[i][2];
    }
    thresholdB /= threshold;
    thresholdG /= threshold;
    thresholdR /= threshold;


    std::vector<cv::Mat> channels(3);
    cv::split(src, channels);

    cv::Mat binaryR, binaryG, binaryB;
    cv::threshold(channels[2], binaryR, thresholdR, 255, cv::THRESH_BINARY);
    cv::threshold(channels[1], binaryG, thresholdG, 255, cv::THRESH_BINARY);
    cv::threshold(channels[0], binaryB, thresholdB, 255, cv::THRESH_BINARY);

    cv::Mat mask;
    cv::bitwise_and(binaryR, binaryG, mask);
    cv::bitwise_and(mask, binaryB, dst);

    int countBlack = 0;
    int countWhite = 0;
    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            if (dst.at<uchar>(i, j) == 0) {
                countBlack++;
            } 
            if (dst.at<uchar>(i, j) == 255) {
                countWhite++;
            }
        }
    }
    //determine which color is the background
    if (countBlack < countWhite) {
        cv::bitwise_not(dst, dst);
    }

    return 0;
}

//implementation of cleanup, src is the input image, dst is the output image
int cleanup(cv::Mat &src, cv::Mat &dst) {
    cv::Mat grow;
    grow = src.clone();

    // grow in 4 first, shrink in 8 next
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            if (src.at<uchar>(i, j) == 0) {
                if (src.at<uchar>(i - 1, j) == 255 
                || src.at<uchar>(i + 1, j)== 255 
                || src.at<uchar>(i, j - 1) == 255 
                || src.at<uchar>(i, j + 1)== 255) {
                    grow.at<uchar>(i, j) = 255;
                }
            }
        }
    }

    dst = grow.clone();

    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            if (grow.at<uchar>(i, j) == 255) {
                if (grow.at<uchar>(i - 1, j) == 0 
                || grow.at<uchar>(i + 1, j) == 0 
                || grow.at<uchar>(i, j - 1)== 0 
                || grow.at<uchar>(i, j + 1) == 0 
                || grow.at<uchar>(i - 1, j - 1) == 0
                || grow.at<uchar>(i - 1, j + 1) == 0 
                || grow.at<uchar>(i + 1, j - 1) == 0
                || grow.at<uchar>(i + 1, j + 1)== 0){
                    dst.at<uchar>(i, j) = 0;
                }
            }
        }
    }
    return 0;
}

//implementation of connected component labeling with 4-connectivity, src is the input image, top5RegionColorMap is the output image with top 5 regions colored, region_map_reduced is the output image with top 5 regions labeled
int segmentation(cv::Mat &src, cv::Mat &top5RegionColorMap, cv::Mat &region_map_reduced) {
    std::stack<std::pair<int, int>> stack; // int, int: Region id, pixel count
    int region_counter = 0;
    cv::Mat region_map;
    region_map = cv::Mat::ones(src.rows, src.cols, CV_32SC1) * 0;

    int pixel_per_region = 0;
    // priority queue to store the region with the most pixels in region, pixel count pair
    // Declare a priority queue that uses the compare structure for sorting pairs
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, compare> pq;

    //implementation of connected component labeling with 4-connectivity
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (src.at<uchar>(i, j) == 255 and region_map.at<uchar>(i, j) == 0) {
                region_counter++;
                pixel_per_region = 1;
                region_map.at<uint8_t>(i, j) = region_counter;
                stack.push(std::make_pair(i, j));
                while (!stack.empty()) {
                    
                    std::pair<int, int> current = stack.top();
                    stack.pop();
                    // Directions: N, S, E, W
                    int dx[] = {-1, 1, 0, 0};
                    int dy[] = {0, 0, 1, -1};
                    
                    for (int k = 0; k < 4; k++) {
                        int row = current.first + dx[k];
                        int col = current.second + dy[k];
                        if (row >= 0 && col >= 0 && row < src.rows && col < src.cols) {
                            if (src.at<uchar>(row, col) == 255 && region_map.at<uint8_t>(row, col) == 0) {
                                region_map.at<uint8_t>(row, col) = region_counter;
                                pixel_per_region++; // Correctly counting each pixel
                                stack.push(std::make_pair(row, col));
                            }
                        }
                    }
                }
                pq.push(std::make_pair(region_counter, pixel_per_region));
            } 
        }
    }

    int colors[5][3] = {
                        {255, 0, 0},    // Red
                        {0, 255, 0},    // Green
                        {0, 0, 255},    // Blue
                        {0, 255, 255},  // Cyan
                        {255, 0, 255}  // Magenta
                        }; 

    std::vector<int> top5_region_id;
    if (!pq.empty()) {
        // Print the top 5 regions with the most pixels
        for (int i = 0; i < 5 && !pq.empty(); i++) {
            std::pair<int, int> region_info = pq.top();
            int region_id = region_info.first;
            int region_pixel_count = region_info.second;
            top5_region_id.push_back(region_id);
            pq.pop();
            printf("Region ID: %d, Pixel Count: %d\n", region_id, region_pixel_count);
        }
        // Color the top 5 regions
        for (int k = 0; k < top5_region_id.size(); k++) {
            for (int i = 0; i < region_map.rows; i++) {
                for (int j = 0; j < region_map.cols; j++) {
                    if (region_map.at<uint8_t>(i, j) == top5_region_id[k]) {
                        region_map_reduced.at<uint8_t>(i, j) = 1 + k;
                        cv::Vec3b color(colors[k][2], colors[k][1], colors[k][0]);
                        top5RegionColorMap.at<cv::Vec3b>(i, j) = cv::Vec3b(color);
                    }
                }
            }
        }
    }
    return 0;
}

//implementation of computeFeaturesAndDisplay, regionMap is the input image with regions labeled, src is the input image, regionID is the region ID, features is the output vector of features
cv::Mat computeFeaturesAndDisplay(const cv::Mat& regionMap, const cv::Mat& src, int regionID, std::vector<std::vector<float>> &features) {
    //calculate features for regions from 1 to regionID, regionID >=1
    if (regionID < 1) {
        std::cerr << "Invalid regionID: " << regionID << std::endl;
        return cv::Mat(); // Return an empty matrix to indicate failure
    }
    cv::Mat displayImage;
    if (src.channels() == 1) {
        cv::cvtColor(src, displayImage, cv::COLOR_GRAY2BGR);
    } else {
        displayImage = src.clone(); // Use directly if it's already a color image
    }
    
    for (int i = 1; i <= regionID; i++) {
        std::vector<float> regionFeatures;
        // printf("Region ID: %d\n", i);
        cv::Mat mask = (regionMap == i);
        // Find contours for the extracted region
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (contours.empty()) {
            std::cerr << "No contours found for regionID: " << regionID << std::endl;
            return cv::Mat(); // Return an empty matrix to indicate failure
        }
        double contourArea = cv::contourArea(contours[0]);
        // Calculate moments for the largest contour (assuming it's our region of interest)
        cv::Moments m = cv::moments(contours[0]);

        // Step 2: Compute the orientation (axis of least central moment)
        double theta = 0.5 * std::atan2(2 * m.mu11, m.mu20 - m.mu02);

        // Step 3: Compute the oriented bounding box and other features
        cv::RotatedRect rotRect = cv::minAreaRect(contours[0]);
        //calculate percentFilled;
        double percentFilled;
        double boundingBoxRatio;
        if (rotRect.size.area() > 0) {
            percentFilled = contourArea / rotRect.size.area();
        } else {
            percentFilled = 0;
        }

        //calculate bounding box heigh/width ratio
        boundingBoxRatio = rotRect.size.height / rotRect.size.width;
        regionFeatures.push_back(percentFilled);
        regionFeatures.push_back(boundingBoxRatio);

        //e1 and e2
        double e1_x = std::cos(theta);
        double e1_y = std::sin(theta); 
        double e2_x = -std::sin(theta);
        double e2_y = std::cos(theta);
        regionFeatures.push_back(e1_x);
        regionFeatures.push_back(e1_y);
        regionFeatures.push_back(e2_x);
        regionFeatures.push_back(e2_y);

        //calculate Hu moments
        std::vector<double> huMoments;
        cv::HuMoments(m, huMoments);
        for (int i = 0; i < huMoments.size(); i++) {
            regionFeatures.push_back(huMoments[i]);
            printf("Hu Moment %d: %f\n", i, huMoments[i]);
        }
        features.push_back(regionFeatures);


        // Draw the oriented bounding box
        cv::Point2f points[4];
        rotRect.points(points);
        for (int i = 0; i < 4; i++) {
            cv::line(displayImage, points[i], points[(i+1)%4], cv::Scalar(0, 255, 0), 2); // Green for box
        }

        double halfDiagonal = 0.5 * std::sqrt(rotRect.size.width * rotRect.size.width + rotRect.size.height * rotRect.size.height);
        // Draw the axis of least central moment
        cv::Point center(static_cast<int>(m.m10/m.m00), static_cast<int>(m.m01/m.m00)); // Centroid
        cv::Point endpoint(center.x + std::cos(theta) * halfDiagonal, center.y + std::sin(theta) * halfDiagonal); // Extend line from centroid
        cv::line(displayImage, center, endpoint, cv::Scalar(255, 0, 0), 2); // Blue for axis

        //add label to center of region
        std::string text = "Region " + std::to_string(i);
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double baseFontScale = 0.5;
        double fontScale = baseFontScale * (rotRect.size.height / 300.0);
        double thickness = 2;
        cv::Scalar textColor(50, 205, 50); //lime color
        // Calculate text size to adjust placement
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
        // Adjust text origin to be at the center of the region
        cv::Point textOrg(center.x - textSize.width / 2, center.y + textSize.height / 2);

        cv::putText(displayImage, text, textOrg, fontFace, fontScale, textColor, thickness);

    }
    return displayImage;
}

//implementation of kNN, euclideanDistanceSums is the vector of euclidean distance sums, labels is the vector of labels, k is the number of nearest neighbors
std::string kNN(std::vector<float> euclideanDistanceSums, std::vector<char *> labels, int k) {
    if (k > euclideanDistanceSums.size()) {
        printf("Invalid k value: %d\n", k);
        return NULL;
    }
    std::vector<std::pair<float, std::string>> euclideanDistanceSumLabelPairs; 
    for (int i = 0; i < euclideanDistanceSums.size(); i++) {
        euclideanDistanceSumLabelPairs.push_back(std::make_pair(euclideanDistanceSums[i], std::string(labels[i])));
    }
    std::sort(euclideanDistanceSumLabelPairs.begin(), euclideanDistanceSumLabelPairs.end(), [](std::pair<float, std::string> a, std::pair<float, std::string> b) {
        return a.first < b.first; //use < operator for ascending order
    });
    std::map<std::string, int> myMap;
    for (int i = 0; i < k; i++) {
        printf("euclideanDistanceSumLabelPairs [%d]:%f, %s\n", i, euclideanDistanceSumLabelPairs[i].first, euclideanDistanceSumLabelPairs[i].second.c_str());
        myMap[euclideanDistanceSumLabelPairs[i].second]++;
        
    }
    for (const auto& [key, value] : myMap) {
        std::cout << key << ": " << value << std::endl;
    }

    if (myMap.size() == k) {
        return euclideanDistanceSumLabelPairs[0].second;
    } else {
        
        int max = 0;
        std::string maxLabel;
        for (auto it = myMap.begin(); it != myMap.end(); it++) {
            printf("Label: %s, Count: %d\n", it->first.c_str(), it->second);
            if (it->second > max) {
                max = it->second;
                maxLabel = it->first;
            }
        }
        return maxLabel;
    }
}
