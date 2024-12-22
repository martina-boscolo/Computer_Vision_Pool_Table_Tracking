//Author: Martina Boscolo Bacheto
#include "table_orientation.h"

using namespace cv;
using namespace std;

/**
 * @brief Calculates the euclidean distance between two points.
 * @param pt1 The first point.
 * @param pt2 The second point.
 * @return The distance between the two points.
 */
float calculateDistance(cv::Point2f pt1, cv::Point2f pt2) {
    return std::sqrt(std::pow(pt2.x - pt1.x, 2) + std::pow(pt2.y - pt1.y, 2));
}

/**
 * @brief Retrieves file paths matching a pattern in a specified folder.
 * @param folder The folder to search in.
 * @param pattern The file pattern to match
 * @return A vector of file paths matching the pattern.
 */
std::vector<cv::String> getPaths(const cv::String& folder, const cv::String& pattern) {
    std::vector<cv::String> allPaths;
    cv::utils::fs::glob(folder, pattern, allPaths);
    return allPaths;
}

/**
 * @brief Reads vertex coordinates from a file.
 * @param filePath The path to the file containing vertex coordinates.
 * @param srcPoints An array to store the read coordinates.
 */
void readVerticesFromFile(const cv::String& filePath, cv::Point2f srcPoints[4]) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr<<"Could not open or find the image"<< std::endl;
    }

    for (int i = 0; i < 4; ++i) {
        if (!(file >> srcPoints[i].x >> srcPoints[i].y)) {
            std::cerr<<"Could not open or find the image"<< std::endl;
        }
    }
    file.close();
}

/**
 * @brief Processes a line segment of the table edge in the image.
 * Computes a shorter line for avoid holes in the angles, then dilates the line and applies a
 * medianBlur for smoothing the image. Finally, applies a threshold to better detect the holes.
 * @param inputImage The original grayscale image.
 * @param start The start point of the line segment.
 * @param end The end point of the line segment.
 * @return A processed image of the line segment region.
 */
cv::Mat processImageLine(const cv::Mat& inputImage, const cv::Point2f& start, const cv::Point2f& end) {
    cv::Mat mask = cv::Mat::zeros(inputImage.size(), inputImage.type());
    cv::Point2f midpoint = (start + end) * 0.5;
    float factor = 0.7;
    cv::Point2f newStart = midpoint + (start - midpoint) * factor;
    cv::Point2f newEnd = midpoint + (end - midpoint) * factor;

    cv::line(mask, newStart, newEnd, cv::Scalar(255, 255, 255), 2);

    cv::Mat dilatedMask; //for inspecting an area around the line
    int dilationSize = 8;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * dilationSize + 1, 2 * dilationSize + 1),
        cv::Point(dilationSize, dilationSize));
    cv::dilate(mask, dilatedMask, element);

    cv::Mat resultImage;
    inputImage.copyTo(resultImage, dilatedMask);
    cv::medianBlur(resultImage, resultImage, 5); //noise removal
    cv::threshold(resultImage, resultImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU); //treshold to select the hole

    return resultImage;
}

/**
 * @brief Counts the number of separated components in the image.
 * @param image The image to analyze.
 * @return The number of components found in the image.
 */
int countComponents(const cv::Mat& image) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(image, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return static_cast<int>(contours.size());
}

/**
 * @brief Reorders the vertices based on the detected table orientation.
 * Order should be such the first point and the second one form a short side,
 * the others are ordered consequently clockwise.
 * @param srcPoints The array of vertex points to be reordered.
 * @param countComponents A vector containing the number of components for each edge.
 */
void orderVerticesShortFirst(cv::Point2f srcPoints[4], const std::vector<int>& connectedComponents) {
    cv::Point2f srcPointsOrdered[4];
    if ((connectedComponents[0] < 2 && connectedComponents[2] < 2) || (connectedComponents[1] > 1 && connectedComponents[3] > 1)) {
        // vertical orientation
        for (int i = 0; i < 4; ++i) {
            srcPointsOrdered[i] = srcPoints[(i + 1) % 4]; // rotate the points by one
        }
    }
    else {
        // horizontal orientation
        std::copy(srcPoints, srcPoints + 4, srcPointsOrdered); // left as they are
    }//other cases may be implemented to make this more robust
    std::copy(srcPointsOrdered, srcPointsOrdered + 4, srcPoints);
}

/**
 * @brief Writes the ordered vertices to a file.
 * @param filePath The path of the file to write to.
 * @param srcPoints The array of ordered vertex points.
 */
void writeVerticesToFile(const cv::String& filePath, const cv::Point2f srcPoints[4]) {
    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {
        std::cerr<<"Could not open or find the image"<< std::endl;
    }
    for (int i = 0; i < 4; ++i) {
        outFile << srcPoints[i].x << " " << srcPoints[i].y << "\n";
    }
    outFile.close();
}

/**
 * @brief Processes a single image, detecting table lateral holes and ordering vertices.
 * @param imagePath The path to the input image.
 * @param vertexPath The path to the file containing initial vertex coordinates.
 * @param outputPath The path to write the ordered vertices.
 */
vector<Point2f> processImage(const cv::String& imagePath, const cv::String& vertexPath, const cv::String& outputPath) {
    cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    vector<Point2f> vertices;
    if (inputImage.empty()) {
        std::cerr<<"Could not open or find the image"<< std::endl;
    }

    cv::Point2f srcPoints[4];
    readVerticesFromFile(vertexPath, srcPoints);

    std::vector<int> connectedComponents;
    for (int i = 0; i < 4; ++i) {
        cv::Mat processedLine = processImageLine(inputImage, srcPoints[i], srcPoints[(i + 1) % 4]);
        connectedComponents.push_back(countComponents(processedLine));
    }

    orderVerticesShortFirst(srcPoints, connectedComponents);
    writeVerticesToFile(outputPath, srcPoints);
    for (int i = 0; i < 4; ++i) {
        vertices.push_back(srcPoints[i]);
    }
    return vertices;
}

/**
 * @brief call this to execute all steps above and find the correct otrientation of the vertices of the table
 * @return allVertices A vector of vectors containing the vertices of the table for each video. 
 */
vector<vector<Point2f>> tableOrientation() {
    cv::String inputFolder = "../Last_frames_images";
    cv::String vertexFolder = "../Table_vertices";
    cv::String outputFolder = "../Table_vertices_for_tracking";
    cv::String pattern = "*.png";
    std::vector<cv::String> imagePaths = getPaths(inputFolder, pattern);
    std::vector<cv::String> vertexPaths = getPaths(vertexFolder, "*.txt");
    vector<vector<Point2f>> allVertices;
    for (size_t i = 0; i < imagePaths.size(); ++i) {
        
        String t = (i+1 < 10) ? "0"+std::to_string(i+1) : std::to_string(i+1);
        cv::String outputPath = outputFolder + "/vertices_table_" + t + ".txt";//writes them in a file (debug and testing)
        vector<Point2f> vertices = processImage(imagePaths[i], vertexPaths[i], outputPath); //saves the vertices
        allVertices.push_back(vertices);
    }
    return allVertices;
 
}