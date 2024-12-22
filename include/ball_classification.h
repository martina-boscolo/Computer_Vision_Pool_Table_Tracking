//CODE WRITTEN BY ALBERTO BRESSAN
#ifndef BALLS_CLASSIFICATION_H
#define BALLS_CLASSIFICATION_H


#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "Classes.h"


void applyMedianFilter(cv::Mat& image, const cv::Rect& bbox);

int countCannyEdges(const cv::Mat& img, const cv::Rect& bbox, int lowerThreshold, int upperThreshold);

void applyLogTransform(const cv::Mat& src, cv::Mat& dst);

std::pair<float, int> whiteRatio(const cv::Mat image, const cv::Rect& bbox, int usat, int lval);

std::pair<float, int> GrWhiteRatio(const cv::Mat image, const cv::Rect& bbox, int threshold);

float darkRatio(const cv::Mat image, const cv::Rect& bbox, int lval);

float smoothWhiteRatio(const cv::Mat& image, const cv::Rect& bbox, int kernelSize, int usat, int lval);

int classifyBall(const cv::Mat& image, const cv::Rect& bbox);

std::vector<std::vector<int>> classifiedVector(const std::vector<cv::Mat>& images, const std::vector<std::vector<BoundingBox>>& allBBoxes);


#endif // BALLS_CLASSIFICATION_H