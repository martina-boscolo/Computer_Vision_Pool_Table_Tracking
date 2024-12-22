//Author: Martina Boscolo Bacheto

#ifndef TABLE_ORIENTATION_H
#define TABLE_ORIENTATION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <opencv2/core/utils/filesystem.hpp>

float calculateDistance(cv::Point2f pt1, cv::Point2f pt2);
std::vector<cv::String> getPaths(const cv::String& folder, const cv::String& pattern);
void readVerticesFromFile(const cv::String& filePath, cv::Point2f srcPoints[4]);
cv::Mat processImageLine(const cv::Mat& inputImage, const cv::Point2f& start, const cv::Point2f& end);
int countComponents(const cv::Mat& image);
void orderVerticesShortFirst(cv::Point2f srcPoints[4], const std::vector<int>& connectedComponents);
void writeVerticesToFile(const cv::String& filePath, const cv::Point2f srcPoints[4]);
std::vector<cv::Point2f> processImage(const cv::String& imagePath, const cv::String& vertexPath, const cv::String& outputPath);
std::vector<std::vector<cv::Point2f>> tableOrientation();
#endif // TABLE_ORIENTATION_H