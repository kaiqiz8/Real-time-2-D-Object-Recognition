/*
    Kaiqi Zhang
	Spring 2024
	CS 5330 Project 3

    Classify an image using the nearest neighbor Euclidean distance metric
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
#include <boost/filesystem.hpp>
#include "util/csv_util.h"
#include <cmath>

using namespace boost::filesystem;
struct recursive_directory_range
{
    typedef recursive_directory_iterator iterator;
    recursive_directory_range(path p) : p_(p) {}

    iterator begin() { return recursive_directory_iterator(p_); }
    iterator end() { return recursive_directory_iterator(); }

    path p_;
};

int main(int argc, char *argv[]) {
    cv::Mat src;
    cv::Mat binaryImage;
    cv::Mat clean;
    char filename[256];

    if(argc < 2) {
		printf("usage: %s < image filename>\n", argv[0]);
		exit(-1);
	}
	strcpy( filename, argv[1]);
    src = cv::imread( filename ); 
    blur5x5_2(src, src);
    threshold(src, binaryImage, 2);
    cleanup(binaryImage, clean);

    cv::Mat region_map_color;
    cv::Mat region_map_reduced = cv::Mat::ones(src.rows, src.cols, CV_8U) * 0;;
    region_map_color = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
    segmentation(clean, region_map_color, region_map_reduced);
    cv::Mat markedImage;
    std::vector<std::vector<float>> features;
    markedImage = computeFeaturesAndDisplay(region_map_reduced, src, 3, features);
    cv::imshow("Marked Image", markedImage);
    cv::imshow("Region map color", region_map_color);
    cv::waitKey(1);

    int desiredRegion;
    bool validInput = false;

    // Prompt the user for a regionID until a valid input is provided
    do {
        std::cout << "Enter the regionID you want to select (positive integer): " << std::flush;
        std::cin >> desiredRegion;

        if (std::cin.fail() || desiredRegion < 1 || desiredRegion > 3) {
            std::cout << "Invalid input. Please enter a positive integer." << std::endl;
            std::cin.clear(); // Reset the error state of cin
            // Ignore the rest of the incorrect input
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            validInput = false;
        } else {
            validInput = true;
        }
    } while (!validInput);

    std::cout << "You selected regionID: " << desiredRegion << std::endl;
    printf("Extracting region %d\n", desiredRegion);
    for (int i = 0; i < features.size(); i++) {
        printf("Region %d: ", i);
        for (int j = 0; j < features[i].size(); j++) {
            printf("%f, ", features[i][j]);
        }
        printf("\n");
    }
    std::vector<float> desiredFeatures = features[desiredRegion - 1];

    std::vector<char *> labels;
    std::vector<std::vector<float>> featuresOfAllImages;
    std::vector<float> stdevs;
    std::vector<std::vector<float>> stdevsOfAllImages;
    read_image_data_csv("../database.csv", labels, featuresOfAllImages);
    std::vector<std::tuple<std::string, int, int>> label_range; // in each tuple, (label, start index, end index)
    int l = 0;
    while (l < labels.size()) {
        int start = l;
        while (l < labels.size() && strcmp(labels[l], labels[start]) == 0) {
            l++;
        }
        label_range.push_back(std::make_tuple(labels[start], start, l - 1));
    }


    for (int i = 0; i < label_range.size(); i++) {
        printf("Label: %s\n", std::get<0>(label_range[i]).c_str());
        printf("Start: %d\n", std::get<1>(label_range[i]));
        printf("End: %d\n", std::get<2>(label_range[i]));
    }

    for (int i = 0; i < label_range.size(); i++) {
        std::vector<float> stdevs;
        for (int j = 0; j < featuresOfAllImages[0].size(); j++) {
            float sum = 0.0;
            for (int k = std::get<1>(label_range[i]); k <= std::get<2>(label_range[i]); k++) {
                sum += featuresOfAllImages[k][j];
            }
            float mean = sum / (std::get<2>(label_range[i]) - std::get<1>(label_range[i]) + 1);
            float squareSum = 0.0;
            for (int k = std::get<1>(label_range[i]); k <= std::get<2>(label_range[i]); k++) {
                squareSum += (featuresOfAllImages[k][j] - mean) * (featuresOfAllImages[k][j] - mean);
            }
            float variance = squareSum / (std::get<2>(label_range[i]) - std::get<1>(label_range[i]) + 1);
            float stdev = sqrt(variance);
            if (stdev == 0) {
                stdev = 1.0;
                printf("Feature %d has stdev 0\n", j);
            }
            stdevs.push_back(stdev);
        }
        stdevsOfAllImages.push_back(stdevs);
    }
    
    std::vector<std::vector<float>> euclideanDistances;

    for (int i = 0; i < featuresOfAllImages.size(); i++) {
        std::vector<float> distances;
        for (int j = 0; j < featuresOfAllImages[i].size(); j++) {
            distances.push_back(0.0);
        }
        euclideanDistances.push_back(distances);
    }
    
    
    for (int i = 0; i < desiredFeatures.size(); i++) {
        for (int j = 0; j < label_range.size(); j++) {
            for (int k = std::get<1>(label_range[j]); k <= std::get<2>(label_range[j]); k++) {
                float distance = (desiredFeatures[i] - featuresOfAllImages[k][i]) / stdevsOfAllImages[j][i];
                euclideanDistances[k][i] = std::abs(distance);
            }
        }
    }

    std::vector<float> euclideanDistanceSums;
    for (int i = 0; i < euclideanDistances.size(); i++) {
        float sum = 0.0;
        for (int j = 0; j < euclideanDistances[i].size(); j++) {
            sum += euclideanDistances[i][j];
        }
        euclideanDistanceSums.push_back(sum);
    }
    int minIndex = 0;
    float minDistance = euclideanDistanceSums[0];
    for (int i = 0; i < euclideanDistanceSums.size(); i++) {
        if (euclideanDistanceSums[i] < minDistance) {
            minDistance = euclideanDistanceSums[i];
            minIndex = i;
        }
        printf("euclideanDistanceSums [%d]:%f, \n", i, euclideanDistanceSums[i]);
    }

    printf("The image is most similar to label: %s\n", labels[minIndex]);
    std::string knnResult = kNN(euclideanDistanceSums, labels, 3);
    printf("The image is most similar to label using KNN: %s\n", knnResult.c_str());


    cv::Mat classifyImage = src.clone();
    cv::Mat mask = (region_map_reduced == desiredRegion);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        std::cerr << "No contours found for regionID: " << desiredRegion << std::endl;
    }
    double contourArea = cv::contourArea(contours[0]);
    cv::Moments m = cv::moments(contours[0]);
    double theta = 0.5 * std::atan2(2 * m.mu11, m.mu20 - m.mu02);
    cv::RotatedRect rotRect = cv::minAreaRect(contours[0]);
    cv::Point2f points[4];
    rotRect.points(points);
    for (int i = 0; i < 4; i++) {
        cv::line(classifyImage, points[i], points[(i+1)%4], cv::Scalar(0, 255, 0), 2); // Green for box
    }
    cv::Point center(static_cast<int>(m.m10/m.m00), static_cast<int>(m.m01/m.m00)); // Centroid
    //add label to center of region
    std::string text(labels[minIndex]);
    // std::string text =  labels[minIndex];
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

    cv::putText(classifyImage, text, textOrg, fontFace, fontScale, textColor, thickness);

    cv::imshow("Classified Image", classifyImage);
    cv::waitKey(0);

    printf("Terminating\n");


    return 0;
}