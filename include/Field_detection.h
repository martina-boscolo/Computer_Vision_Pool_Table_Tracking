//CODE WRITTEN BY LOVO MANUEL 2122856

#ifndef FIELD_DETECTION_H
#define FIELD_DETECTION_H

#include "Field_detection.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream> 
#include <iomanip>
#include <sstream>


std::vector<cv::Mat> HSVsegmentation(const std::vector<cv::Mat>& images, int lowH, int lowS, int lowV, int highH, int highS, int highV);

cv::Point2f computeIntersect(const cv::Vec2f line1, const cv::Vec2f line2);

void orderPoint(std::vector<cv::Point2f>& points);

std::vector<std::vector<cv::Point2f>> vectorHoughLines(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& output, std::vector<cv::Mat>& masks, int divisory, int threshold, int number_lines);

#endif // FIELD_DETECTION_H