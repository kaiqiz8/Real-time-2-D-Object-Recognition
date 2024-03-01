/*
    Kaiqi Zhang
	Spring 2024
	CS 5330 Project 3

    Collect training data from a directory of images to a csv file specified by user
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

using namespace boost::filesystem;

struct recursive_directory_range
{
    typedef recursive_directory_iterator iterator;
    recursive_directory_range(path p) : p_(p) {}

    iterator begin() { return recursive_directory_iterator(p_); }
    iterator end() { return recursive_directory_iterator(); }

    path p_;
};


//argv[1] = directory path
//argv[2] = csv file name
int main(int argc, char *argv[]) {
    std::string directory_path;
    char csv_file_name[256];
    
    if (argc < 3) {
        printf("usage: %s <directory path> <csv file name>\n", argv[0]);
        exit(-1);
    }
    directory_path = argv[1];
    strcpy( csv_file_name, argv[2]);

    //loop through all the files in the directory
    for (auto it: recursive_directory_range(directory_path)) {
        if (is_regular_file(it) && it.path().extension() == ".png") {
            std::vector<std::string> results;
            std::istringstream iss(directory_path);
            std::string token;
            while (std::getline(iss, token, '/')) {
                results.push_back(token);
            }
            char tokenChar[256];
            strcpy( tokenChar, token.c_str());
            // const char* tokenChar = token.c_str();
            
            char image_filename[256];
            strcpy( image_filename, it.path().string().c_str());
            cv::Mat src;
            src = cv::imread( image_filename );
            if (src.data == NULL) {
                printf("Unable to open image file: %s\n", image_filename);
                exit(-1);
            }
            blur5x5_2(src, src);
            cv::Mat binaryImage;
            threshold(src, binaryImage, 2);
            cv::Mat clean;
            cleanup(binaryImage, clean);
            cv::Mat region_map_color;
            cv::Mat region_map_reduced = cv::Mat::ones(src.rows, src.cols, CV_8U) * 0;;
            region_map_color = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
            segmentation(clean, region_map_color, region_map_reduced);
            cv::Mat markedImage;
            std::vector<std::vector<float>> features;
            // for (int i = 1; i < 4; i++) {
            markedImage = computeFeaturesAndDisplay(region_map_reduced, src, 3, features);
            cv::imshow("Region map color", region_map_color);
            cv::imshow("Marked Image", markedImage);
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
            printf("Desired Features: ");
            for (int i = 0; i < desiredFeatures.size(); i++) {
                printf("%f, ", desiredFeatures[i]);
            }
            append_image_data_csv(csv_file_name, tokenChar, desiredFeatures, 0);
        }
    }
    return 0;

}