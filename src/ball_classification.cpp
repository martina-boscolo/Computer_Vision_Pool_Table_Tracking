//CODE WRITTEN BY ALBERTO BRESSAN
#include "ball_classification.h"


void applyMedianFilter(cv::Mat& image, const cv::Rect& bbox) {
   
        cv::Mat roi = image(bbox);
        cv::Mat filteredROI;
        cv::medianBlur(roi, filteredROI, 3);
        filteredROI.copyTo(image(bbox));
}


/*This method is used to compute the number of edges in the circle inscrbed in each bounding box. The idea behind 
this is that a striped ball will probably show more edges than a solid ball. The number of edges is computed using 
a Canny image of the roi in the grayscale image.
Parameters: image, bounding box, lower threshold, upper threshold
Returns: number of Canny edges in the inscribed circle
*/

int countCannyEdges(const cv::Mat& img, const cv::Rect& bbox, int lowerThreshold = 110, int upperThreshold = 250) {
    //Compute the roi and the inscrbed circle
    cv::Mat gray = img.clone();
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

    cv::Mat roi = gray(bbox);

    int radius = std::min(bbox.width, bbox.height) / 2;
    cv::Point center(bbox.width / 2, bbox.height / 2);

    cv::Mat mask = cv::Mat::zeros(roi.size(), CV_8UC1);
    cv::circle(mask, center, radius, cv::Scalar(255), -1);

    cv::Mat maskedROI;
    roi.copyTo(maskedROI, mask);
    //Apply Canny edge detection
    cv::Mat edges;
    cv::Canny(maskedROI, edges, lowerThreshold, upperThreshold);

    //Count the number of non-zero (edge) points in the Canny image
    int edgePointCount = cv::countNonZero(edges);

    return edgePointCount;
}


/*
The following methods are used to compute the ratio of white and dark pixels in the inscribed circle for each bounding box.
 In each method the inscribed circle is computed and the computation done inside this roi.
*/


/*
This method computes the ratio of white pixels using thresholds on the HSV image.
Parameters: image, bounding box, upper saturation threshold, lower value threshold
(the hue parameter is set to a value, since it is not useful for the computations)
Returns: ratio of white pixels in the inscribed circle
*/
std::pair<float, int> whiteRatio(const cv::Mat image, const cv::Rect& bbox, int usat, int lval) {

    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
   
   // Extract the region of interest (ROI) using the bounding box and the inscribed circle
    cv::Mat roi = hsv(bbox);

    int radius = std::min(bbox.width, bbox.height) / 2;
    cv::Point center(bbox.width / 2, bbox.height / 2);

    cv::Mat mask = cv::Mat::zeros(roi.size(), CV_8UC1);
    cv::circle(mask, center, radius, cv::Scalar(255), -1);

    // Thresholds for white color in HSV space
    int lower_hue = 0, upper_hue = 180;    
    int lower_saturation = 0, upper_saturation = usat;  
    int lower_value = lval, upper_value = 255;

    // Analyze each pixel within the inscribed circle counting the white pixels
    int whitePixelCount = 0;
    int darkPixelCount = 0;
    int totalPixelCount = 0;

    for (int y = 0; y < roi.rows; y++) {
        for (int x = 0; x < roi.cols; x++) {
            if (mask.at<uchar>(y, x) > 0) { 
                cv::Vec3b pixel = roi.at<cv::Vec3b>(y, x);
                int hue = pixel[0];
                int saturation = pixel[1];
                int value = pixel[2];

                if (hue >= lower_hue && hue <= upper_hue &&
                    saturation >= lower_saturation && saturation <= upper_saturation &&
                    value >= lower_value && value <= upper_value) {
                    whitePixelCount++;
                }
                totalPixelCount++;
            }
        }
    }
    // Calculate the ratio of white pixels
    float whitePixelRatio = static_cast<float>(whitePixelCount) / totalPixelCount;
    return std::make_pair(whitePixelRatio, totalPixelCount);
}

/*
This method computes the ratio of dark pixels using a threshold on the HSV image.
Parameters: image, bounding box, threshold
Returns: ratio of dark pixels in the inscribed circle
*/
float darkRatio(const cv::Mat image, const cv::Rect& bbox, int lval) {
    // Extract the region of interest (ROI) using the bounding box and the inscribed circle
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
   
    cv::Mat roi = hsv(bbox);
    int radius = std::min(bbox.width, bbox.height) / 2;
    cv::Point center(bbox.width / 2, bbox.height / 2);

    cv::Mat mask = cv::Mat::zeros(roi.size(), CV_8UC1);
    cv::circle(mask, center, radius, cv::Scalar(255), -1);

    // Thresholds for dark color in HSV space
    int upper_value = lval; 

    int darkPixelCount = 0;
    int totalPixelCount = 0;

    for (int y = 0; y < roi.rows; y++) {
        for (int x = 0; x < roi.cols; x++) {
            if (mask.at<uchar>(y, x) > 0) { 
                cv::Vec3b pixel = roi.at<cv::Vec3b>(y, x);
                int value = pixel[2];

                if (value <= upper_value) {
                    darkPixelCount++;
                }
                totalPixelCount++;
            }
        }
    }
    // Calculate the ratio of dark pixels
    float darkPixelRatio = static_cast<float>(darkPixelCount) / totalPixelCount;
    return darkPixelRatio;
}

/*
This method is used to simplify the code in the classifyBall method. It computes the ratio of white pixels on the HSV image,
after applying Gaussian smoothing and a median filter to remove noise and inferences.
Parameters: image, bounding box, kernelSize, usat, lval
Returns: ratio of white pixels
*/

float smoothWhiteRatio(const cv::Mat& image, const cv::Rect& bbox, int kernelSize = 3, int usat = 90, int lval = 170) {
    cv::Mat image2 = image.clone();
    cv::GaussianBlur(image2, image2, cv::Size(kernelSize, kernelSize), 0);
    applyMedianFilter(image2, bbox);
    return whiteRatio(image2, bbox, usat, lval).first;
}

/*
    This method is the core of the classification task. It iteravely applies the various methods modifying the parameters
    and the preprocessing of the image.
    The threshold become increasingly specific to classify balls that have particular properties given by their position and
    the charateristics of the image.
    Parameters: image, bounding box
    Returns: ball type
*/

int classifyBall(const cv::Mat& image, const cv::Rect& bbox) {
    float ratio = whiteRatio(image, bbox, 90, 170).first;
    float ratioDark = darkRatio(image, bbox, 100);
    int totalPixelCount = whiteRatio(image, bbox, 90, 170).second;

    if (ratio < 0.008 || ratioDark > 0.7) {
        return 3;
    }

    if (ratio > 0.19) {
        return 4;
    } else {
        ratio = smoothWhiteRatio(image, bbox);

        if (ratio < 0.02) {
            return 3;
        }
        if (ratio > 0.24) {
            return 4;
        } else {
            int cannyCount = countCannyEdges(image, bbox);
            if (cannyCount > 160) {
                return 4;
            } else {
               if (ratioDark > 0.5 || cannyCount < 50 || ratio < 0.02) {
                    return 3;
                } else {
                    cv::Mat immg = image.clone();
                    cv::GaussianBlur(immg, immg, cv::Size(3, 3), 0);
                    cannyCount = countCannyEdges(immg, bbox);
                    float cannyRatio = (float)cannyCount / totalPixelCount;

                    if (cannyCount > 130) {
                        return 4 ;
                    } else {
                        if (cannyRatio < 0.26) {
                            return 3;
                        }
                        else{
                            ratio = whiteRatio(image, bbox, 110, 170).first;
                            if(ratio < 0.11 || ratio > 0.25){
                                return 3;
                            }else{
                                return 4;
                            }
                            
                        }
                    }
                }
            }
        }
    }
}

std::vector<std::vector<int>> classifiedVector(const std::vector<cv::Mat>& images, const std::vector<std::vector<BoundingBox>>& allBBoxes) {
    std::vector<std::vector<int>> result;

    for (size_t imgIndex = 0; imgIndex < images.size(); ++imgIndex) {
        const cv::Mat& image = images[imgIndex];
        const std::vector<BoundingBox>& bboxes = allBBoxes[imgIndex];

        std::vector<std::tuple<cv::Rect, int, float, float>> labeledBBoxesWithRatios;

        for (const auto& bbox : bboxes) {
            cv::Rect rect(bbox.x, bbox.y, bbox.width, bbox.height);
            int label = classifyBall(image, rect);
            float whiteBallRatio = whiteRatio(image, rect, 100, 170).first;
            float blackBallRatio = darkRatio(image, rect, 70);
            labeledBBoxesWithRatios.push_back(std::make_tuple(rect, label, whiteBallRatio, blackBallRatio));
        }
        
        auto maxWhiteRatioIt = std::max_element(labeledBBoxesWithRatios.begin(), labeledBBoxesWithRatios.end(),
            [](const std::tuple<cv::Rect, int, float, float>& a, const std::tuple<cv::Rect, int, float, float>& b) {
                return std::get<2>(a) < std::get<2>(b);
            });

        auto maxBlackRatioIt = std::max_element(labeledBBoxesWithRatios.begin(), labeledBBoxesWithRatios.end(),
            [](const std::tuple<cv::Rect, int, float, float>& a, const std::tuple<cv::Rect, int, float, float>& b) {
                return std::get<3>(a) < std::get<3>(b);
            });

        if (maxWhiteRatioIt != labeledBBoxesWithRatios.end()) {
            std::get<1>(*maxWhiteRatioIt) = 1; 
        }
        if (maxBlackRatioIt != labeledBBoxesWithRatios.end()) {
            std::get<1>(*maxBlackRatioIt) = 2; 
        }
       
        std::vector<int> labeledBBoxes;
        for (const auto& item : labeledBBoxesWithRatios) {
            labeledBBoxes.push_back(std::get<1>(item));
        }

        for (const auto& item : labeledBBoxesWithRatios) {
            const cv::Rect& rect = std::get<0>(item);
            int label = std::get<1>(item);
        }
        
        result.push_back(labeledBBoxes);
    }

    return result;
}