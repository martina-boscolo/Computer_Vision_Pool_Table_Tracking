//CODE WRITTEN BY LOVO MANUEL 2122856

#ifndef BALLS_SEGMENTATION_H
#define BALLS_SEGMENTATION_H

#include "Balls_segmentation.h"
#include <opencv2/opencv.hpp>
#include <iostream>

cv::Vec3b computeMedianColor(const cv::Mat& image, const cv::Point& center, int neighborhood);

cv::Vec3b computeMedianColorOfCenteredROI(const cv::Mat& image, int roiSize);

std::vector<cv::Mat> vectorMaskRGB(const std::vector<cv::Mat>& images, const std::vector<cv::Vec3b>& colors, int threshold);

std::vector<cv::Vec3f> houghCircles(const cv::Mat &image, double dp, double minDist, double param1, double param2, int minRadius, int maxRadius);

double calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2);

void filterCircles_nearHoles(std::vector<cv::Vec3f>& circles, const std::vector<cv::Point2f>& holes_points, float threshold);

double calculateColorDifference(const cv::Vec3b& color1, const cv::Vec3b& color2);

double getColorDifferenceSum(const cv::Mat& image, const cv::Vec3f& circle, const cv::Vec3b& tableMedianColor);

std::vector<cv::Vec3f> filterCircles_replicas_1(const cv::Mat& image, const std::vector<cv::Vec3f>& set1, const std::vector<cv::Vec3f>& set2, const cv::Vec3b& tableMedianColor, double distanceThreshold);

void filterCirclesByColor(const cv::Mat& image, const cv::Vec3b& targetColor, double threshold, std::vector<cv::Vec3f>& circles);

std::vector<cv::Vec3f> refineCircles_replicas_2(const std::vector<cv::Vec3f>& set1, const std::vector<cv::Vec3f>& set2, double distanceThreshold);

cv::Point2f closestPointOnLineSegment(const cv::Point2f& point, const cv::Point2f& lineStart, const cv::Point2f& lineEnd);

std::vector<cv::Vec3f> pointsNearLines(const std::vector<cv::Vec3f>& circles, const std::vector<cv::Point2f>& rectanglePoints, float threshold);

std::vector<cv::Vec3f> refineCircles_ROI(const cv::Mat& image, const std::vector<cv::Vec3f>& circles, float scaleFactor, int param2, double scaleFactorMinRadius, double scaleFactorMaxRadius);

std::vector<cv::Mat> maskBlack(const std::vector<cv::Mat>& images, int threshold);


#endif // BALLS_SEGMENTATION_H

