//CODE WRITTEN BY LOVO MANUEL 2122856

#ifndef UTILITIES_H
#define UTILITIES_H

#include "Utilities.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <opencv2/core/utils/filesystem.hpp>

cv::Mat combinedImages(std::vector<cv::Mat> &images);

std::vector<cv::Mat> vectorDilation(const std::vector<cv::Mat>& images, int structElem);

std::vector<cv::Mat> vectorErosion(const std::vector<cv::Mat>& images, int structElem);

std::vector<cv::Mat> vectorMedianBlur(const std::vector<cv::Mat>& images, int ksize);

std::vector<cv::Mat> vectorSmoothing(std::vector<cv::Mat>& images, int ksize);

std::vector<cv::Mat> vectorCanny(const std::vector<cv::Mat>& images, double lower_thresh, double upper_thresh);

void drawCircles(cv::Mat &img, std::vector<cv::Vec3f> circles,cv::Scalar color);

std::vector<cv::Mat> readImages(const std::vector<std::string>& path, bool flag);

void savePointsToFile(const std::vector<std::vector<cv::Point2f>>& data, const std::string& filename);

std::vector<std::vector<cv::Point2f>> readPointsFromFile(const std::string& filename);

std::vector<std::string> getImagePaths(const std::string & folder, const std::string & pattern);

#endif // UTILITIES_H

