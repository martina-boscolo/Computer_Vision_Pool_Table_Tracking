//Author: Martina Boscolo Bacheto 

#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <opencv2/core/utils/filesystem.hpp>
#include "Classes.h"
#include "table_orientation.h"


void drawTransformedBalls(cv::Mat& image, const cv::Mat& perspectiveMatrix, const cv::Rect bbox, int id);
void drawTransformedPath(cv::Mat& image, const cv::Mat& perspectiveMatrix, const std::vector<cv::Point2f> trackPoints);
bool readBoundingBoxes(const std::string& bboxFile, std::vector<Ball>& balls);
bool initializeVideoCapture(const std::string& videoFile, cv::VideoCapture& cap);
cv::Mat computePerspectiveMatrix(std::vector<cv::Point2f>& realWorldPoints);
bool initializeOutputVideo(const std::string& outputVideoFile, cv::VideoWriter& outputVideo, const cv::VideoCapture& cap);
void initializeTrackers(const cv::Mat& frame, const std::vector<Ball>& balls, std::vector<cv::Ptr<cv::Tracker>>& trackers, std::vector<cv::Rect>& rois, std::vector<int>& categoryIDs, std::vector<std::vector<cv::Point>>& paths);
cv::Mat loadAndResizeTableMap(const std::string& imagePath, int desiredHeight, int& newWidth);
void processFrames(cv::VideoCapture& cap, cv::VideoWriter& outputVideo, const cv::Mat& perspectiveMatrix, std::vector<cv::Ptr<cv::Tracker>>& trackers, std::vector<cv::Rect>& rois, std::vector<int>& categoryIDs, const cv::String nameBB, const cv::String name2DMap);
void writeLastFrameBB(const std::vector<cv::Rect>& rois, const std::vector<int>& categoryIDs, const cv::String& nameBB, const cv::String& name2DMap, const cv::Mat& lastframe);
   
#endif // TRACKING_H