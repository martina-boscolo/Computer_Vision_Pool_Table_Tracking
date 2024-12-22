//CODE WRITTEN BY LOVO MANUEL 2122856

#include "Utilities.h"

/*
This function has been used for combining a vector of images into one large image, subdivided as a grid.
This is why we need to look together all the images in the grid in all processes of the program.

Parameters:
- images: The input images.
    
Returns:
- The combined image.
*/
cv::Mat combinedImages(std::vector<cv::Mat> &images) { 
    int numImages = images.size();
    int gridRows = static_cast<int>(std::ceil(std::sqrt(numImages))); //Number of rows equal to the square root of the number of images
    int gridCols = static_cast<int>(std::ceil(static_cast<double>(numImages) / gridRows)); //Number of columns equal to the number of images divided by the number of rows

    //Find the maximum width and height of the images s.t they fit in the grid
    int maxWidth = 0;
    int maxHeight = 0;
    for (size_t i = 0; i < images.size(); ++i) {
        if (images[i].cols > maxWidth) {
            maxWidth = images[i].cols;
        }
        if (images[i].rows > maxHeight) {
            maxHeight = images[i].rows;
        }
    }

    //Creation of a large image
    int combinedWidth = gridCols * maxWidth;
    int combinedHeight = gridRows * maxHeight;
    cv::Mat combinedImage(combinedHeight, combinedWidth, images[0].type(), cv::Scalar::all(0));

    //Insert each image in the grid
    for (int i = 0; i < numImages; ++i) {
        int row = i / gridCols; //For finding dimension of image
        int col = i % gridCols;
        cv::Rect roi(col * maxWidth, row * maxHeight, images[i].cols, images[i].rows);
        images[i].copyTo(combinedImage(roi)); //Copy the image in the correct point
    }

    return combinedImage;
}

/*
This function has been used for applying dilation to a vector of images, instead than on a single image.

Parameters:
- images: The input images.
- structElem: The size of the structuring element.

Returns:
- A vector images after the application of dilation.
*/
std::vector<cv::Mat> vectorDilation(const std::vector<cv::Mat>& images, int structElem) {

    // Ensure the structuring element size is odd
    if (structElem % 2 == 0) {
        std::cerr << "Warning: Structuring element size is even. Incrementing by one to make it odd." << std::endl;
        structElem += 1;
    }
    std::vector<cv::Mat> dilated_images;
    for (const auto& image : images) {
        // Check if the image is in grayscale
        if (image.channels() != 1) {
            std::cerr << "Error: Input images must be in grayscale." << std::endl;
            return {};
        }
        cv::Mat dilated_image;
        cv::dilate(image, dilated_image, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(structElem, structElem)));
        dilated_images.push_back(dilated_image);
    }
    return dilated_images;
}

/*
This function has been used for applying erosion to a vector of images, instead than on a single image.

Parameters:
- images: The input images.
- structElem: The size of the structuring element.

Returns:
- A vector images after the application of erosion.
*/
std::vector<cv::Mat> vectorErosion(const std::vector<cv::Mat>& images, int structElem) {
    if (structElem % 2 == 0) {
        std::cerr << "Warning: Structuring element size is even. Incrementing by one to make it odd." << std::endl;
        structElem += 1;
    }
    std::vector<cv::Mat> eroded_images;
    for (const auto& image : images) {
        // Check if the image is in grayscale
        if (image.channels() != 1) {
            std::cerr << "Error: Input images must be in grayscale." << std::endl;
            return {};
        }

        cv::Mat eroded_image;
        cv::erode(image, eroded_image, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(structElem, structElem)));
        eroded_images.push_back(eroded_image);
    }
    return eroded_images;
}

/*
This function has been used for applying median blur to a vector of images, instead than on a single image.

Parameters:
- images: The input images.
- ksize: The size of the kernel.

Returns:
- A vector images after the application of median blur.
*/
std::vector<cv::Mat> vectorMedianBlur(const std::vector<cv::Mat>& images, int ksize) {
    if(ksize % 2 == 0) {
        std::cerr << "Warning: Kernel size is even. Incrementing by one to make it odd." << std::endl;
        ksize += 1;
    }
    std::vector<cv::Mat> median_blurred_images;
    for(const auto& image : images) {
        cv::Mat median_blurred_image;
        cv::medianBlur(image, median_blurred_image, ksize);
        median_blurred_images.push_back(median_blurred_image);
    }
    return median_blurred_images;
}

/*
This function has been used for applying smoothing to a vector of images, instead than on a single image.

Parameters:
- images: The input images.
- ksize: The size of the kernel.

Returns:
- A vector images after the application of smoothing.

Notes:
- If the kernel size is even, it will be incremented by one to make it odd.
*/
std::vector<cv::Mat> vectorSmoothing(std::vector<cv::Mat>& images, int ksize) {
    if(ksize % 2 == 0) {
        std::cerr << "Warning: Kernel size is even. Incrementing by one to make it odd." << std::endl;
        ksize += 1;
    }
    std::vector<cv::Mat> smoothed_images;
    for(const auto& image : images) {
        cv::Mat smoothed_image;
        cv::GaussianBlur(image, smoothed_image, cv::Size(ksize, ksize), 0);
        smoothed_images.push_back(smoothed_image);
    }
    return smoothed_images;
}

/*
This function has been used for applying canny edge detection to a vector of images, instead than on a single image.

Parameters:
- images: The input images.
- lower_thresh: The lower threshold.
- upper_thresh: The upper threshold.

Returns:
- A vector images after the application of canny edge detection.

Notes:
-If the image is not in grayscale, it will be converted to grayscale, since Canny edge detection works on grayscale images.
*/
std::vector<cv::Mat> vectorCanny(const std::vector<cv::Mat>& images, double lower_thresh, double upper_thresh) {
    std::vector<cv::Mat> canny_images;
    for (const auto& image : images) {
        cv::Mat gray;
        // Check if the image is in grayscale
        if (image.channels() != 1) {
            std::cerr << "Warning: Input images must be in grayscale." << std::endl;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        }
        else {
            gray = image;
        }
        cv::Mat canny_image;
        cv::Canny(gray, canny_image, lower_thresh, upper_thresh);
        canny_images.push_back(canny_image);
    }
    return canny_images;
}

/*
This function has been used for drawing circles on an image passed by reference.
This is very important since circles derives from different processes and it's useful to look in the same image for debugging purposes.

Parameters:
- img: The image to draw the circles on.
- circles: The circles to draw.
- color: The color of the circles.

Returns:
- The image with the drawn circles, passed by reference.
*/
void drawCircles(cv::Mat &img, std::vector<cv::Vec3f> circles,cv::Scalar color) {
    for (size_t i = 0; i < circles.size(); ++i) {
        cv::Vec3i c = circles[i];
        cv::circle(img, cv::Point(c[0], c[1]), c[2],color, 2, cv::LINE_AA);
    }
}

/*
This function has been used for reading a vector of images from a vector of paths.
Moreover, it's possible to read images in grayscale or in color by tuning a boolean flag.
*/
std::vector<cv::Mat> readImages(const std::vector<std::string>& path, bool flag) {
    std::vector<cv::Mat> images;
    if(flag) {
        for (size_t i = 0; i < path.size(); ++i) {
        const std::string& imagePath = path[i]; 
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << imagePath << std::endl;
            continue;
        }
        images.push_back(img);
        }   
    }
    else {
        for (size_t i = 0; i < path.size(); ++i) {
        const std::string& imagePath = path[i]; 
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << imagePath << std::endl;
            continue;
        }
        images.push_back(img);
        }
    }
    return images;
}

/*
Function used for saving a vector of points to a file. The file has been generated by the main and then used in the metrics.
It's saved in a way that it can be read by the readPointsFromFile function.

Parameters:
- data: The vector of points to save.
- filename: The name of the file to save.

Returns:
- void
*/
void savePointsToFile(const std::vector<std::vector<cv::Point2f>>& data, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    for (const auto& row : data) {
        for (const auto& point : row) {
            file << point.x << " " << point.y << " ";
        }
        file << "\n"; 
    }

    file.close();
}

/*
Function used for reading a vector of points from a file. the file has been generated by the main and then used in the metrics.

Parameters:
- filename: The name of the file to read.

Returns:
- A vector of vector of points.
*/
std::vector<std::vector<cv::Point2f>> readPointsFromFile(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<cv::Point2f>> data;

    if (!file.is_open()) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return data;
    } //If there is no file, error and empty vector

    std::string line; // Line read from the file
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<cv::Point2f> row; // Points read from the line
        float x, y; 
        while (iss >> x >> y) { // Extract x and y coordinates
            row.emplace_back(cv::Point2f(x, y));
        }
        data.push_back(row);
    }

    file.close();
    return data;
}

/*
Function used in order to avoid absolute paths, as suggested in laboratory.
The function is taken from the laboratory part of the course.

Parameter
- folder: The folder containing the images.
- pattern: The pattern of the FILES.

Returns
- A vector of paths.
*/
std::vector<std::string> getImagePaths(const std::string & folder, const std::string & pattern) {
    std::vector<cv::String> allPathsCV;
    cv::utils::fs::glob(folder, pattern, allPathsCV);

    std::vector<std::string> allPaths;
    for (const auto& path : allPathsCV) {
        allPaths.push_back(path);
    }

    return allPaths;
}