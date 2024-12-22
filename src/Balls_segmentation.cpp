//CODE WRITTEN BY LOVO MANUEL 2122856

#include "Balls_segmentation.h"

/*
This function has been used for computing the median color of a neighborhood, centered in a point, of an image.
More precisely, the function requires as input the image, the center of the neighborhood, and the size of the neighborhood and then check all the BGR 
color of each pixel inside a circle of radius equal to the half of the neighborhood size.

The function returns the median color of the neighborhood. We choose the neighborhood since this function needs for computing a mask based
on the median color of a portion of table. In our idea, in the portion of table there could be also a small piece of ball, so we have decided to take the emdian to avoid this problem.

Parameters:
- image: The input image.
- center: The center of the neighborhood.
- neighborhood: The size of the neighborhood.

Returns:
- The median color of the neighborhood.
*/

cv::Vec3b computeMedianColor(const cv::Mat& image, const cv::Point& center, int neighborhood) {
    std::vector<int> blues, greens, reds;
    int radius = neighborhood / 2;
    
    for (int y = center.y - radius; y <= center.y + radius; ++y) { // For each pixel in the columns of the circle
        for (int x = center.x - radius; x <= center.x + radius; ++x) { // For each pixel in the rows of the circle
            int dx = x - center.x; // Compute the distance between the center of the circle and the current pixel
            int dy = y - center.y;
            if (x >= 0 && x < image.cols && y >= 0 && y < image.rows && (dx * dx + dy * dy <= radius * radius)) { // Check if the current pixel is inside the circle
                cv::Vec3b color = image.at<cv::Vec3b>(cv::Point(x, y));
                blues.push_back(color[0]);
                greens.push_back(color[1]);
                reds.push_back(color[2]);
            } //Increment the vector with the correspective BGR color
        }
    }
    
    std::nth_element(blues.begin(), blues.begin() + blues.size() / 2, blues.end()); //Function of the STL library used to find the median
    std::nth_element(greens.begin(), greens.begin() + greens.size() / 2, greens.end());
    std::nth_element(reds.begin(), reds.begin() + reds.size() / 2, reds.end());
    
    cv::Vec3b medianColor;
    medianColor[0] = blues[blues.size() / 2]; //Take the median since it's already in the correct position
    medianColor[1] = greens[greens.size() / 2];
    medianColor[2] = reds[reds.size() / 2];
    
    return medianColor;
}

/*
This function has been used for computing the median color of a centered ROI (region of interest) of an image.
The most important part is done by the previously defined function computeMedianColor. Simply this function provides the image center 
parameter of computeMedianColor and the roiSize, determined by the user.

Parameters:
- image: The input image.
- roiSize: The size of the ROI.

Returns:
- The median color of the ROI centered in the image.
*/
cv::Vec3b computeMedianColorOfCenteredROI(const cv::Mat& image, int roiSize) {
    int centerX = image.cols / 2;
    int centerY = image.rows / 2;
    cv::Point center(centerX, centerY);
    return computeMedianColor(image, center, roiSize);
}

/*
This function is the core of our computation for segmenting balls.
The idea near this function is to segment the table from the balls. There are three case in which a pixel will be masked:

- If the pixel color is within a distance threshold respect to the color of the table, meaning that it's a table pixel
- If the pixel color is within a distance threshold respect to the black, meaning that it's a shadow pixel or an hole pixel
- If the pixel color is within a distance threshold respect to the white, meaning that it's a hole pixel etc.
The two second cases were difficult to handle since there are some colors that are close to black and also part of the ball etc. 
So, after various test, the threshold was set equal for all calls of this function.

Parameters:
- images: The input images.
- color: A color, that, in our case, will be the color of the table.
- threshold: The color distance threshold for the mask.

Returns:
- The vector of masks.
*/
std::vector<cv::Mat> vectorMaskRGB(const std::vector<cv::Mat>& images, const std::vector<cv::Vec3b>& colors, int threshold) {
    std::vector<cv::Mat> masks;
    int k = 0;
    for(const auto& img : images) {
        cv::Mat mask = cv::Mat::ones(img.rows, img.cols, CV_8UC1)*255; //Create a mask all white
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
                
                // Calculate the absolute difference between the current pixel and the specified color, that, in our case will be the color of the table
                int diff_b = std::abs(pixel[0] - colors[k][0]);
                int diff_g = std::abs(pixel[1] - colors[k][1]);
                int diff_r = std::abs(pixel[2] - colors[k][2]);

                //If the pixel color is within a distance threshold respect to the color of the table
                bool isColorMatch = (diff_b < threshold && diff_g < threshold && diff_r < threshold);

                // Check if the pixel is within the threshold for black
                bool isBlackMatch = (pixel[0] < 100 && pixel[1] < 100 && pixel[2] < 100);

                // Check if the pixel is within the threshold for white
                bool isWhiteMatch = (pixel[0] > 255 - 30 && pixel[1] > 255 - 30 && pixel[2] > 255 - 30);

                // If any of the conditions are met, set the mask pixel to 0
                if (isColorMatch || isBlackMatch || isWhiteMatch) {
                    mask.at<uchar>(i, j) = 0;
                }
            }
        }
        masks.push_back(mask);
        k++;
    
    }
    return masks;  
}

/*
Function that has been used to apply Hough Circle to an image.

Parameters:
- image: The input image.
- dp: The inverse ratio of resolution.
- minDist: The minimum distance between the centers of the detected circles.
- param1: The first method-specific parameter.
- param2: The second method-specific parameter.
- minRadius: The minimum radius of the circle.
- maxRadius: The maximum radius of the circle.

Returns:
- The vector of circles.
*/
std::vector<cv::Vec3f> houghCircles(const cv::Mat &image, double dp, double minDist, double param1, double param2, int minRadius, int maxRadius) {
    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);  // Convert to grayscale if it's not already
    } else {
        grayImage = image.clone();
    }
    std::vector<cv::Vec3f> circles;  // Vector to store the circles

    // Apply Hough Circle Transform, storing circles in 'circles'
    cv::HoughCircles(grayImage, circles, cv::HOUGH_GRADIENT, dp, minDist, param1, param2, minRadius, maxRadius);
    return circles;
}

/*
Function that calculates the distance between two points.

Parameters:
- p1: The first point.
- p2: The second point.

Returns:
- The distance between the two points.
*/
double calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    // Calculate the squared difference for the x and y coordinates
    double diffX = std::pow(p1.x - p2.x, 2);
    double diffY = std::pow(p1.y - p2.y, 2);

    // Sum the squared differences and take the square root to get the Euclidean distance
    return std::sqrt(diffX + diffY);
}

/*
Function that has been used to remove circles that are too close to the holes.
That is why, due to the circular shape of the holes, in some cases HoughCircle find circles in the near or corresponding to the holes.
So this function will remove those circles too close to the holes, computing the distance between the circle and the hole point and by using a threshold.

Parameters:
- circles: The vector of circles, passed by reference.
- holes_points: The vector of hole points.
- threshold: The distance threshold.

Returns:
- The filtered vector of circles, modified in place.
*/
void filterCircles_nearHoles(std::vector<cv::Vec3f>& circles, const std::vector<cv::Point2f>& holes_points, float threshold) {
    std::vector<cv::Vec3f> filteredCircles;

    for (const cv::Vec3f& circle : circles) { // Loop through each circle
        cv::Point2f circleCenter(circle[0], circle[1]);
        bool tooClose = false;

        for (const cv::Point2f& point : holes_points) { // Loop through each hole point
            if (calculateDistance(circleCenter, point) < threshold) { // If the distance between the circle and the hole point is less than the threshold
                tooClose = true;
                break;
            }
        }

        if (!tooClose) { // If the circle is not too close to any hole point is maintained
            filteredCircles.push_back(circle);
        }
    }
    circles = filteredCircles;
}

/*
This function has been used to calculate the Euclidean distance between two colors.
It needs for further processing to refine two set of circles.

Parameters:
- color1: The first color.
- color2: The second color. 

Returns:
- The Euclidean distance between the two colors.
*/
double calculateColorDifference(const cv::Vec3b& color1, const cv::Vec3b& color2) {
    
    double diffBlue = std::pow(color1[0] - color2[0], 2);
    double diffGreen = std::pow(color1[1] - color2[1], 2); //Squared difference between two channels of different colors
    double diffRed = std::pow(color1[2] - color2[2], 2);

    double sumOfSquares = diffBlue + diffGreen + diffRed; //Sum of squared differences

    return std::sqrt(sumOfSquares); 
}

/*
This function has been used to calculate the sum of the differences between the colors inside the circle and the median color of the table, provided as parameter.
To do so, it's created a mask in correspondence of circles and for each pixel inside the circle, it's calculated the difference between the color of the pixel and the median color of the table using the function defined above.
It's used for further processing to refine two set of circles.

Parameters:
- image: The image in which the circles are located.
- circle: The circles.
- tableMedianColor: The median color of the table.

Returns:
- The sum of the differences between the colors inside the circle and the median color of the table.
*/
double getColorDifferenceSum(const cv::Mat& image, const cv::Vec3f& circle, const cv::Vec3b& tableMedianColor) {
    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(0));
    cv::circle(mask, cv::Point(circle[0], circle[1]), circle[2], cv::Scalar(255), -1); // Create a mask for circles

    double totalDifference = 0.0;
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            if (mask.at<uchar>(y, x) == 255) {
                cv::Vec3b color = image.at<cv::Vec3b>(y, x);
                totalDifference += calculateColorDifference(color, tableMedianColor); // Sum the differences
            }
        }
    }
    return totalDifference;
}

/*
This function has been used to refine two set of circles.
Since our program finds all circles from different processes, there are some cases where a circle is found by more than one process.
This function aims to maintain only one circle between two near circles (if two circles are within a certain distance, meaning that with high probability are referred to the same ball, we will only keep one). 
The circle is maintained is the one that is "better": the goodness of a circle is measured by the sum of the differences between the colors inside the circle and the median color of the table.
The idea is that, if a circle has a portion in the table, the sum of the differences between the colors inside the circle and the median color of the table is lower, and so we maintain the other.

Parameters:
- image: The image in which the circles are located.
- set1: The first set of circles.
- set2: The second set of circles.
- tableMedianColor: The median color of the table.
- distanceThreshold: The distance threshold.

Returns:
- The filtered set of circles.
*/
std::vector<cv::Vec3f> filterCircles_replicas_1(const cv::Mat& image, const std::vector<cv::Vec3f>& set1, const std::vector<cv::Vec3f>& set2, const cv::Vec3b& tableMedianColor, double distanceThreshold) {

    std::vector<cv::Vec3f> filteredCircles;

    for (const auto& circle1 : set1) { // Loop through the first set of circles
        bool keep = true;
        for (const auto& circle2 : set2) { // Loop through the second set of circles
            double distance = calculateDistance(cv::Point2f(circle1[0], circle1[1]), cv::Point2f(circle2[0], circle2[1]));
            if (distance < distanceThreshold) { // If the distance is less than the threshold
                double colorDiffSum1 = getColorDifferenceSum(image, circle1, tableMedianColor); 
                double colorDiffSum2 = getColorDifferenceSum(image, circle2, tableMedianColor); 
                //Calculate the sum of the differences between the colors inside the circle and the median color of the table

                if (colorDiffSum1 < colorDiffSum2) { // If circle1 has a lower color difference sum
                    // Replace circle1 with circle2 if circle2 has a greater color difference sum
                    keep = false;
                }
                break;
            }
        }
        if (keep) {
            filteredCircles.push_back(circle1);
        }
    }

    //Here we simply add the set of circles that are not fallen under the threshold
    for (const auto& circle2 : set2) {
        bool replaced = false;
        for (const auto& circle1 : filteredCircles) { // Loop through the filtered circles
            double distance = calculateDistance(cv::Point2f(circle2[0], circle2[1]), cv::Point2f(circle1[0], circle1[1]));
            if (distance < distanceThreshold) {
                replaced = true;
                break;
            }
        }
        if (!replaced) { // If the circle is not replaced
            filteredCircles.push_back(circle2); // Add the circle to the filtered circles
        }
    }

    return filteredCircles;
}

/*
This function has been used to refine a set of circles.
More precisely, there are same false positives that comes from the boundary of the image, for example in the table bound.
However, these circles have the same color of the median color of the table, so, this function, is used to filter out the circles that have the same color as the median color of the table, within a threshold.

Parameters:
- image: The image in which the circles are located.
- circles: The set of circles.
- tableMedianColor: The median color of the table.
- threshold: The threshold.

Returns:
- The filtered set of circles, not given as output but stored in the parameter circles, passed by reference.
*/
void filterCirclesByColor(const cv::Mat& image, const cv::Vec3b& targetColor, double threshold, std::vector<cv::Vec3f>& circles) {
    std::vector<cv::Vec3f> filteredCircles;

    for (const auto& circle : circles) { // Loop through the circles
        cv::Point center(cvRound(circle[0]), cvRound(circle[1])); // Get the center of the circle
        int radius = cvRound(circle[2]); // Get the radius of the circle

        cv::Vec3b medianColor = computeMedianColor(image, center, 2 * radius);
        double distance = calculateColorDifference(medianColor, targetColor);

        if (distance > threshold) { // If the distance is greater than the threshold
            filteredCircles.push_back(circle);
        }
    }

    circles = filteredCircles;
}

/*
This function has been used to refine two sets of circles. It's very similar as the refineCircles_replicas_1 function.
The idea behind is the same: since our program finds all circles from different processes, we need to filter out the same circle found by two different processes.
In this case, since we have worked with different function etc, there are some cases in which circles are smaller than the correct balls.
So, this function aims to filter out those circles that, being compared to another circles for the same ball, is smaller.

Parameters:
- set1: The first set of circles.
- set2: The second set of circles.
- distanceThreshold: The distance threshold.

Returns:
- The filtered set of circles.

*/
std::vector<cv::Vec3f> refineCircles_replicas_2(const std::vector<cv::Vec3f>& set1, const std::vector<cv::Vec3f>& set2, double distanceThreshold) {
    std::vector<cv::Vec3f> filteredCircles;

    for (const auto& circle1 : set1) { // Loop through the set1 circles
        bool keep = true;
        for (const auto& circle2 : set2) { // Loop through the set2 circles
            double distance = calculateDistance(cv::Point2f(circle1[0], circle1[1]), cv::Point2f(circle2[0], circle2[1]));
            if (distance < distanceThreshold) { // If the distance is within the threshold
                if (circle1[2] < circle2[2]) { // If the circle1 is smaller than the circle2
                    keep = false; // Then the circle1 is not kept
                }
                break;
            }
        }
        if (keep) {
            filteredCircles.push_back(circle1);
        }
    }

    // Add circles from set2 that are not replaced by set1 circles
    for (const auto& circle2 : set2) {
        bool replaced = false;
        for (const auto& filteredCircle : filteredCircles) {
            double distance = calculateDistance(cv::Point2f(circle2[0], circle2[1]), cv::Point2f(filteredCircle[0], filteredCircle[1]));
            if (distance < distanceThreshold) {
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            filteredCircles.push_back(circle2);
        }
    }

    return filteredCircles;
}

/*
This function has been used as support to a function defined below for computing the minimum distance point  between a point and the bound of the table.
This function works by following rules of geometry projection as explained below
*/
cv::Point2f closestPointOnLineSegment(const cv::Point2f& point, const cv::Point2f& lineStart, const cv::Point2f& lineEnd) {
    cv::Point2f vectorLineStartToPoint = point - lineStart; // Vector from lineStart to point
    cv::Point2f vectorLineStartToEnd = lineEnd - lineStart; // Vector from lineStart to lineEnd
    
    float lineLengthSquared = vectorLineStartToEnd.x * vectorLineStartToEnd.x + vectorLineStartToEnd.y * vectorLineStartToEnd.y; 
    //It's not been used the square root of the lineLengthSquared because it's not needed
    float projectionFactor = (vectorLineStartToPoint.x * vectorLineStartToEnd.x + vectorLineStartToPoint.y * vectorLineStartToEnd.y) / lineLengthSquared;
    //The projection factor tells us how far along the line segment (from lineStart to lineEnd) the projection of the point lies
    
    if (projectionFactor < 0.0f) projectionFactor = 0.0f; 
    //. If the projection factor is less than 0, the closest point on the segment to the given point is the start of the segment (lineStart). If the projection factor is greater than 1, the closest point is the end of the segment (lineEnd).
    else if (projectionFactor > 1.0f) projectionFactor = 1.0f;
    
    return cv::Point2f(lineStart.x + vectorLineStartToEnd.x * projectionFactor, lineStart.y + vectorLineStartToEnd.y * projectionFactor);
}

/*
This function has been used for refining the circles that are close to the table.
This is because there are some false positive lying on the boundary of the table.
This methods works well if the perspective of the image is correct, so we know about limitation on this method.
The idea is to check if the distance between the center of the circle and the closest point on the table is lower than a threshold and, if it is remove that point.

Parameters:
- circles: The circles to be refined.
- rectanglePoints: The points defining the table.
- threshold: The threshold for the distance between the center of the circle and the closest point on the table.

Returns:
- The refined circles.
*/
std::vector<cv::Vec3f> pointsNearLines(const std::vector<cv::Vec3f>& circles, const std::vector<cv::Point2f>& rectanglePoints, float threshold) {
    if (rectanglePoints.size() != 4) {
        throw std::invalid_argument("The input vector must contain exactly 4 points.");
    }

    std::vector<cv::Vec3f> newCircles;

    for (const cv::Vec3f& circle : circles) {
        cv::Point2f center(circle[0], circle[1]);
        float minDistance = std::numeric_limits<float>::max();
        
        for (size_t i = 0; i < 4; ++i) {
            const cv::Point2f& startPoint = rectanglePoints[i]; //Work on points adjacent 
            const cv::Point2f& endPoint = rectanglePoints[(i + 1) % 4]; 
            cv::Point2f closestPoint = closestPointOnLineSegment(center, startPoint, endPoint);
            float distance = static_cast<float>(calculateDistance(center, closestPoint));

            if (distance < minDistance) { //Check which is the minimum distance
                minDistance = distance;
            }
        }

        if (minDistance >= threshold) {
            newCircles.push_back(circle);
        }
    }

    return newCircles;
}

/*
Function that has been used for eliminating false positives in hand, boundary of the table etc.
The idea is to take a roi around the circle and check if there is a circle in that roi, using HoughCircle with the parameter2 given as input of the function.
In this way, we also refine the circle position.
At the end, due to the fact that we have taken a bigger roi, we adjust the position back to the image.

Parameters:
- image: The image used for computing masks.
- circles: The circles to be refined.
- scaleFactor: The scale factor for the roi.
- param2: The parameter 2 for the HoughCircle function.
- scaleFactorMinRadius: The minimum radius scale factor for the roi.
- scaleFactorMaxRadius: The maximum radius scale factor for the roi.

Returns:
- The refined circles.
*/
std::vector<cv::Vec3f> refineCircles_ROI(const cv::Mat& image, const std::vector<cv::Vec3f>& circles, float scaleFactor, int param2, double scaleFactorMinRadius, double scaleFactorMaxRadius) {
    std::vector<cv::Vec3f> refinedCircles;
    for (size_t i = 0; i < circles.size(); i++) { //For each circle
        float x = circles[i][0]; //Get its coordinates
        float y = circles[i][1];
        float r = circles[i][2];
        
        int radius = static_cast<int>(r * scaleFactor); //We take a roi a little bit bigger than the circle since there are circles that are not precise enough
        int startX = std::max(0, static_cast<int>(x - radius)); 
        int startY = std::max(0, static_cast<int>(y - radius));
        int endX = std::min(image.cols, static_cast<int>(x + radius));
        int endY = std::min(image.rows, static_cast<int>(y + radius));
        
        cv::Rect roi(startX, startY, endX - startX, endY - startY);
        
        // Extract ROI
        cv::Mat roiImage = image(roi);
        cv::Mat gray;
        cv::cvtColor(roiImage, gray, cv::COLOR_BGR2GRAY); 
        // Apply HoughCircles again on the ROI
        std::vector<cv::Vec3f> refinedCircle;
        cv::HoughCircles(gray, refinedCircle, cv::HOUGH_GRADIENT, 1, gray.rows / 8, 130, param2, radius*scaleFactorMinRadius, radius*scaleFactorMaxRadius);

        
        // If a circle is detected, adjust its position relative to the original image
        if (!refinedCircle.empty()) {
            cv::Vec3f detectedCircle = refinedCircle[0];
            detectedCircle[0] += startX;
            detectedCircle[1] += startY;
            refinedCircles.push_back(detectedCircle);
        }
    }
    return refinedCircles;
}


/*
This function has been used due to the fact that in some regions of the image, illumination changes make difficult to find the circles corresponding to black balls.
So, this method is specified for finding black balls in regions very dark.
It works very similar to the mask function defined above, but here we check if the pixel is within the threshold for black.

Parameters:
- images: The images used for computing masks.
- threshold: The threshold for black.

Returns:
- The masks.
*/
std::vector<cv::Mat> maskBlack(const std::vector<cv::Mat>& images, int threshold) {
    std::vector<cv::Mat> masks;

    for (const auto& img : images) {
        cv::Mat mask = cv::Mat::ones(img.rows, img.cols, CV_8UC1) * 255;

        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);

                // Check if the pixel is within the threshold for black
                bool isBlackMatch = (pixel[0] < threshold && pixel[1] < threshold && pixel[2] < threshold);

                // If the pixel is very similar to black, set the mask pixel to 0
                if (isBlackMatch) {
                    mask.at<uchar>(i, j) = 0;
                }
            }
        }
        masks.push_back(mask);
    }
    return masks;
}
