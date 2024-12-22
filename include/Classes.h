//CODE WRITTEN BY LOVO MANUEL 2122856

#ifndef CLASSES_H
#define CLASSES_H

#include "Classes.h"
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>

//Structure used for the circles: The choice of the structure instead of the class is due to the fact we don't have relevant methods to implement for circles.
struct Circle {
    float x; // x-coordinate of the center
    float y; // y-coordinate of the center
    float radius; // radius of the circle
    Circle(float x, float y, float radius); //Constructor
};

// Structure used for the bounding boxes. Same reasoning as above
struct BoundingBox {
    float x; // x-coordinate (top left corner)
    float y; // y-coordinate (top left corner)
    float width; // width of the bounding box
    float height; // height of the bounding box

    BoundingBox(float x, float y, float width, float height); //Constructor
};

// Ball class, the choice of a class is due to the fact we have some useful methods to implement for the balls
class Ball {
public:
    Ball(float x, float y, float width, float height, int ID);
    void display() const;
    std::string getInfo() const;
    float computeArea() const;

    float getX() const { return x; } //Get top left corner x
    float getY() const { return y; } //Get top left corner y
    float getWidth() const { return width; } //Get width
    float getHeight() const { return height; } //Get height
    int getID() const { return ID; } //Get the ID

private:
    float x;
    float y;
    float width;
    float height;
    int ID;
    void validateID(int &ID); //Private method for checking that the ID is in the correct range
};


std::vector<std::vector<BoundingBox>> createBoundingBoxes(const std::vector<std::vector<Circle>>& final_circles, float scaleFactor);

void showBoundingShapes(std::vector<cv::Mat>& images, const std::vector<std::vector<Ball>>& allBalls, bool drawBoundingBoxes);

std::vector<std::vector<Ball>> createBallsFromBoundingBoxes(const std::vector<std::vector<BoundingBox>>& boundingBoxes, const std::vector<std::vector<int>>& classifier);

std::vector<std::vector<Ball>> readBallsFromTextFiles(const std::vector<std::string>& filePaths);

void createFinalMasks(std::vector<cv::Mat>& images, const std::vector<std::vector<Ball>>& allBalls, const std::vector<std::vector<cv::Point2f>>& points);

std::vector<cv::Mat> transformMasksToColor(const std::vector<cv::Mat>& grayscaleImages);

std::vector<std::vector<Circle>> createCircles(const std::vector<std::vector<cv::Vec3f>>& circles);

#endif // COMBINED_HEADER_H