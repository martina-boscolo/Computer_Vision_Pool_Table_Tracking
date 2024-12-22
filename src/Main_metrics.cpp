// Code written by Lovo Manuel 2122856
#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <vector>
#include <string>

#include "Classes.h"
#include "Metrics.h"
#include "Utilities.h"


int main(int argc, char** argv)
{   

    std::vector<std::string> truth_boundingboxes_first_frames = getImagePaths("../First_frames_truth_bboxes", "*.txt");

    std::vector<std::string> detected_bboxes_first_frames = getImagePaths("../First_frames_detected_bboxes", "*.txt");

    std::vector<std::string> truth_boundingboxes_last_frames = getImagePaths("../Last_frames_truth_bboxes", "*.txt");

    std::vector<std::string> detected_bboxes_last_frames = getImagePaths("../Last_frames_detected_bboxes", "*.txt");

    //Take all paths for computing metrics from correct folders

    std::vector<std::vector<Ball>> true_balls_first_frames = readBallsFromTextFiles(truth_boundingboxes_first_frames);
    std::vector<std::vector<Ball>> detected_balls_first_frames = readBallsFromTextFiles(detected_bboxes_first_frames);
    std::vector<std::vector<Ball>> true_balls_last_frames = readBallsFromTextFiles(truth_boundingboxes_last_frames);
    std::vector<std::vector<Ball>> detected_balls_last_frames = readBallsFromTextFiles(detected_bboxes_last_frames);

    //From paths we create vectors of balls, one for each image

    std::vector<std::vector<std::tuple<float, int, int>>> IoU_vectors_firstframes = calculateIoUVectors(detected_balls_first_frames, true_balls_first_frames);
    std::vector<std::vector<std::tuple<float, int, int>>> IoU_vectors_lastframes = calculateIoUVectors(detected_balls_last_frames, true_balls_last_frames);

    //Here we compute vectors containing information like IoU, ID etc for all balls

    std::cout<<"FIRST FRAMES"<<std::endl;
    std::cout<<"--------------"<<std::endl;
    std::vector<std::vector<float>> AP_values_firstframes = calculateAPs(IoU_vectors_firstframes);

    std::cout<<"LAST FRAMES"<<std::endl;
    std::cout<<"--------------"<<std::endl;
    std::vector<std::vector<float>> AP_values_lastframes = calculateAPs(IoU_vectors_lastframes);

    //Here we compute the mAP 


    std::vector<std::string> maskpaths_first_frames = getImagePaths("../First_frames_truthmasks", "*.png");
    std::vector<std::string> maskpaths_last_frames = getImagePaths("../Last_frames_truthmasks", "*.png");

    //Take all paths for computing metrics from correct folders

    std::vector<cv::Mat> groundMasks_first_frames = readImages(maskpaths_first_frames,false);
    std::vector<cv::Mat> groundMasks_last_frames = readImages(maskpaths_last_frames,false);

    //From paths we read truth masks, in gray scale (bool false)

    std::vector<cv::Mat> detectedMasks_first_frames;
    std::vector<cv::Mat> detectedMasks_last_frames;
    for(int i=0; i<10; i++)
    {
        cv::Mat grayOutput_1(groundMasks_first_frames[i].size(), CV_8UC1);
        cv::Mat grayOutput_2(groundMasks_last_frames[i].size(), CV_8UC1);
        detectedMasks_first_frames.push_back(grayOutput_1);
        detectedMasks_last_frames.push_back(grayOutput_2);
    }
    std::vector<std::vector<cv::Point2f>> pointsList = readPointsFromFile("../Points.txt");
    createFinalMasks(detectedMasks_first_frames, detected_balls_first_frames, pointsList);
    createFinalMasks(detectedMasks_last_frames, detected_balls_last_frames, pointsList);

    //From paths we create our masks

    std::cout<<"FIRST FRAMES"<<std::endl;
    std::cout<<"--------------"<<std::endl;
    std::vector<double> mIoU_vector_firstframes = vectormIoU(groundMasks_first_frames, detectedMasks_first_frames);

    std::cout<<"LAST FRAMES"<<std::endl;
    std::cout<<"--------------"<<std::endl;
    std::vector<double> mIoU_vector_lastframes = vectormIoU(groundMasks_last_frames, detectedMasks_last_frames);

    //We compute the mIoU on segmentation masks

    return 0;
}
