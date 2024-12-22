//CODE WRITTEN BY LOVO MANUEL 2122856

#ifndef METRICS_H
#define METRICS_H

#include "Metrics.h"
#include "Classes.h"

#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <opencv2/opencv.hpp>

float calculateIoU(const Ball& ball1, const Ball& ball2);

bool checkDetection(float IoU, float threshold);

std::vector<std::vector<std::tuple<float, int, int>>> calculateIoUVectors(
    const std::vector<std::vector<Ball>>& detected_balls,
    const std::vector<std::vector<Ball>>& true_balls);

float calculateAP(const std::vector<std::tuple<int, int>>& tp_fp, int num_ground_truths);


std::vector<std::vector<float>> calculateAPs(
    const std::vector<std::vector<std::tuple<float, int, int>>>& IoU_vectors);

double computeIoUForClass(const cv::Mat& groundTruth, const cv::Mat& predicted, int classId);

double computeMeanIoU(const cv::Mat& groundTruth, const cv::Mat& predicted);

std::vector<double> vectormIoU(const std::vector<cv::Mat>& groundTruths, const std::vector<cv::Mat>& predicted);


#endif // METRICS_H