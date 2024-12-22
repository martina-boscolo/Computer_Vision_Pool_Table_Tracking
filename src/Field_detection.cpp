//CODE WRITTEN BY LOVO MANUEL 2122856

#include "Field_detection.h"

/*
This function has been used for computing masks in HSV space based on a range of colors of a vector of images.
The function returns a vector of masks of images where the colors within the range are white and the rest is black.

Parameters:
- images: The input images.
- lowH: The lower bound of the hue range.
- lowS: The lower bound of the saturation range.
- lowV: The lower bound of the value range.
- highH: The upper bound of the hue range.
- highS: The upper bound of the saturation range.
- highV: The upper bound of the value range.

Returns:
- Masks of the images where the colors within the range are white and the rest is black.
*/
std::vector<cv::Mat> HSVsegmentation(const std::vector<cv::Mat>& images, int lowH, int lowS, int lowV, int highH, int highS, int highV) {
    std::vector<cv::Mat> masks;
    // Check if the HSV bounds are within valid ranges
    if (lowH < 0 || lowH > 179 || highH < 0 || highH > 179 ||
        lowS < 0 || lowS > 255 || highS < 0 || highS > 255 ||
        lowV < 0 || lowV > 255 || highV < 0 || highV > 255) {
        std::cerr << "Error: HSV bounds are out of range." << std::endl;
        return masks; 
    }
    for (const auto& image : images) { //Accesses by reference s.t no copy is made
        if (image.empty()) {
                std::cerr << "Error: One of the input images is empty." << std::endl;
                continue; // Skip the empty image
            }
        cv::Mat hsvImage;
        cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV); // Convert the image to HSV space
        cv::Scalar lower(lowH, lowS, lowV); // Define the lower and upper bounds of the mask
        cv::Scalar upper(highH, highS, highV);
        cv::Mat mask;
        cv::inRange(hsvImage, lower, upper, mask); // Create the mask
        masks.push_back(mask);  
    }  
    return masks;
}

/*
For different reasons, we need to compute the intersection between two lines. These points are then used for 
-the detection of the field in a precise way
-the filtering of circles, output of HoughCircles, that are too close to the billiard holes; in fact, sometimes, the holes where confused for balls (false positive) and these points were used for this purpose.
-the mapping in the 2D table
More precisely, it's computed by passing in the rho, theta space of the lines and computing the intersection by using the same formula seen in class for HoughLines.
Moreover, there is the condition that the lines are not parallel, controlled by a function.

Parameters:
- line1: The first line, represented by its rho and theta, using the cv::Vec2f type.
- line2: The second line, represented by its rho and theta, using the cv::Vec2f type.

Returns:
- The intersection point between two lines, given by the cv::Point2f type.
*/
cv::Point2f computeIntersect(const cv::Vec2f line1, const cv::Vec2f line2) {
    float rho1 = line1[0], theta1 = line1[1]; 
    float rho2 = line2[0], theta2 = line2[1];
    float sinT1 = sin(theta1), cosT1 = cos(theta1);
    float sinT2 = sin(theta2), cosT2 = cos(theta2);
    
    float denominator = (cosT1 * sinT2 - cosT2 * sinT1);

    // Check if the denominator is very close to zero (indicating parallel lines)
    if (fabs(denominator) < std::numeric_limits<float>::epsilon()) {
        
        return cv::Point2f(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    }

    float x = (sinT2 * rho1 - sinT1 * rho2) / denominator;
    float y = (cosT1 * rho2 - cosT2 * rho1) / denominator;
    return cv::Point2f(x, y);
}

/*
This function has been used for ordering the points of a vector of points in the order of an angle. 
Points are arranged in a counterclockwise direction around the centroid of the set of points, using a lambda function to sort the points based on their angles with respect to the centroid

Parameters:
- points: The input points.

Returns:
- The vector of points, sorted in counterclockwise order.
*/

void orderPoint(std::vector<cv::Point2f>& points) {
    if (points.size() < 2) {
        std::cerr << "Warning: Less than 2 points provided. No sorting will be performed." << std::endl;
        return;
    }
    cv::Point2f centroid = cv::Point2f(0, 0); 
    for(int i=0; i<points.size(); i++) { //Computation of the centroid, average point of points provided as input
        centroid.x += points[i].x;
        centroid.y += points[i].y;
    }
    centroid.x /= points.size();
    centroid.y /= points.size();
    std::sort(points.begin(), points.end(), [&centroid](const cv::Point2f& a, const cv::Point2f& b) { //Lambda function used to sort points in counterclockwise order
        double angleA = atan2(a.y - centroid.y, a.x - centroid.x); //Comparison of angles for ordering
        double angleB = atan2(b.y - centroid.y, b.x - centroid.x); 
        return angleA < angleB;
    });
}

/*
This function has been used for detecting the lines for images in a vector of images.
More precisely, it uses the cv::HoughLines function for finding lines, tuned with parameters given as input from the function. 
Then we collect only the first number_lines lines, the strongest ones. Then we compute intersection between points of these lines and we maintain only points that are in the bound of the image.
Points are then sorted using the function previously defined and added to a vector of vectors of points, one for each image, used then for further computations.
Then, as last, we compute the mask inside the field modifying the image passed as reference in input and the output image, with lines drawn on it.

Parameters:
- images: The input images, that in our case will be canny images.
- output: The output images, that in our case will be the original images where we want to draw the lines.
- masks: The output masks, that in our case will be the ROI inside the field just detected.
- divisory: The divisor for the rho parameter of the HoughLines function.
- threshold: The minimum number of intersections to "detect" a line
- number_lines: The number of lines we want to detect.

Returns:
- The vector of vectors of points, one for each image, used then for further computations.
Moreover, images passed as reference are modified.
*/

std::vector<std::vector<cv::Point2f>> vectorHoughLines(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& output, std::vector<cv::Mat>& masks, int divisory, int threshold, int number_lines) {
    std::vector<std::vector<cv::Point2f>> pointsList;
    
    for (int i = 0; i < images.size(); i++) {
        std::vector<cv::Vec2f> lines;
        cv::HoughLines(images[i], lines, 1, CV_PI / divisory, threshold);

        for (size_t j = 0; j < std::min(number_lines, (int)lines.size()); j++) {  
            float rho = lines[j][0];
            float theta = lines[j][1];
            cv::Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a * rho, y0 = b * rho;
            pt1.x = cvRound(x0 + 1000 * (-b));
            pt1.y = cvRound(y0 + 1000 * (a));
            pt2.x = cvRound(x0 - 1000 * (-b));
            pt2.y = cvRound(y0 - 1000 * (a));
        }

        std::vector<cv::Point2f> intersection_points;  //Compute the intersection points between couples of lines
        size_t linesCount = std::min(number_lines, (int)lines.size());
        for (size_t j = 0; j < linesCount; j++) {
            for (size_t k = j + 1; k < linesCount; k++) {
                cv::Point2f intersection = computeIntersect(lines[j], lines[k]);
                if (intersection.x >= 0 && intersection.x < images[i].cols && //Check if the intersection point is inside the image
                    intersection.y >= 0 && intersection.y < images[i].rows) {
                    intersection_points.push_back(intersection);  //Include the point only if it is inside the image
                }
            }
        }
        orderPoint(intersection_points); //Application of the function defined above
        if (intersection_points.size() == 4) {
            cv::line(output[i], intersection_points[0], intersection_points[1], cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            cv::line(output[i], intersection_points[1], intersection_points[2], cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            cv::line(output[i], intersection_points[2], intersection_points[3], cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            cv::line(output[i], intersection_points[3], intersection_points[0], cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        } else {
            std::cerr << "Not enough points to form a rectangle." << std::endl;
        }
        pointsList.push_back(intersection_points); //Store the list of points that will be used in further process
        if (intersection_points.size() >= 3) { // Ensure there are enough points to form a polygon
            cv::Mat mask = cv::Mat::zeros(images[i].size(), CV_8UC1);
            std::vector<std::vector<cv::Point>> fillPoints(1); //Vector of cv::Points that will be used to fill the polygon
            for (const auto& point : intersection_points) {
                fillPoints[0].push_back(point); //Implicit conversion from cv::Point to cv::Point2f
            }

            cv::fillPoly(mask, fillPoints, cv::Scalar(255), cv::LINE_8); //Fill the polygon of white

            cv::Mat outputImage = cv::Mat::zeros(images[i].size(), images[i].type()); 
            masks[i].copyTo(outputImage, mask); //Apply the mask to the original image

            masks[i] = outputImage.clone(); // Ensure a deep copy is made to prevent unintentional modifications
        }
        else {
            std::cerr << "Not enough points to form a polygon." << std::endl;
        }

        cv::String t = (i+1 < 10) ? "0"+std::to_string(i+1) : std::to_string(i+1);

        std::ofstream outFile("../Table_vertices/vertices_table_"+t+".txt"); //print vertices in a file to be used in next steps
        if (outFile.is_open()) {
        for (const auto& point : intersection_points) {
            outFile  << point.x <<" "<< point.y << "\n";
        }
        outFile.close(); 
        } else {
            std::cerr << "Unable to open file for writing." << std::endl;
        }
    }
    return pointsList;
}




