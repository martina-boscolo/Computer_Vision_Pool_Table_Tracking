//CODE WRITTEN BY LOVO MANUEL 2122856

#include "Classes.h"

// Circle constructor
Circle::Circle(float x, float y, float radius) : x(x), y(y), radius(radius) {}

/*
This function has been used for creating Circles object from vector of vector of Vec3f.
This structure is used then for for creating the object BoundingBox.

Parameter
-circles: a vector of vector of Vec3f (ideally, for each image we will have a vector of Vec3f)

Return
-a vector of vector of Circles
*/
std::vector<std::vector<Circle>> createCircles(const std::vector<std::vector<cv::Vec3f>>& circles) {

    std::vector<std::vector<Circle>> final_circles;
    for (const auto& image : circles) {
        std::vector<Circle> circles;
        for (const auto& circle : image) {
            circles.push_back(Circle(circle[0], circle[1], circle[2]));
        }
        final_circles.push_back(circles);
    }
    return final_circles;  
}


// BoundingBox constructor
BoundingBox::BoundingBox(float x, float y, float width, float height)
    : x(x), y(y), width(width), height(height) {}



/*This function has been used for creating bounding boxes from Circles object. 
This structure is used then for computing the final ouptut and also as input for the classifier and for creating the object Balls.

Parameter
-final_circles: a vector of vector of Circles (ideally, for each image we will have a vector of Circles)
-scaleFactor: a float value which is used to scale the radius of the circles

Return
-a vector of vector of BoundingBoxes
*/

std::vector<std::vector<BoundingBox>> createBoundingBoxes(const std::vector<std::vector<Circle>>& final_circles, float scaleFactor) {
    std::vector<std::vector<BoundingBox>> allBoundingBoxes;

    for (const auto& circles : final_circles) {
        std::vector<BoundingBox> boundingBoxes;
        for (const auto& circle : circles) {
            float scaledRadius = circle.radius * scaleFactor;
            float diameter = 2 * scaledRadius;
            int bboxX = static_cast<int>(std::round(circle.x - scaledRadius));
            int bboxY = static_cast<int>(std::round(circle.y - scaledRadius));
            int bboxWidth = static_cast<int>(std::round(diameter));
            int bboxHeight = static_cast<int>(std::round(diameter));
            BoundingBox bbox(bboxX, bboxY, bboxWidth, bboxHeight);
            boundingBoxes.push_back(bbox);
        }
        allBoundingBoxes.push_back(boundingBoxes);
    }

    return allBoundingBoxes;
}

/*METHODS DEFINED INSIDE THE CLASS BALL
*/
// Ball class constructor
Ball::Ball(float x, float y, float width, float height, int ID)
    : x(x), y(y), width(width), height(height) {
    validateID(ID);
    this->ID = ID;
}

// Private method for checking that the ID is in the correct range
void Ball::validateID(int &ID) {
    if (ID < 0 || ID > 5) {
        std::cerr << "Error: ID must be between 0 and 5. Setting ID to 0." << std::endl;
        ID = 0;
    }
}

// Method to display ball information
void Ball::display() const {
    std::cout << "Ball ID: " << ID << "\n";
    std::cout << "Position: (" << x << ", " << y << ")\n";
    std::cout << "Size: " << width << " x " << height << "\n";
}

// Method to get ball information as a string
std::string Ball::getInfo() const {
    return "Ball ID: " + std::to_string(ID) + ", Position: (" + std::to_string(x) + ", " + std::to_string(y) + ")";
}

// Method to compute the area of the ball
float Ball::computeArea() const {
    return width * height;
}


/*
This function has been used for merging the work of classification and bounding box detection, creating the object Ball from the vector of BoundingBoxes and the vector of classification.
More specifically, for each image there will be a vector of Balls, where dimensions are taken from the vector of BBoxes and ID from the vector of classifier.

Parameter
-boundingBoxes: a vector of vector of BoundingBoxes
-classifier: a vector of vector of int, corresponding to ID for balls

Return
-a vector of vector of Balls
*/
std::vector<std::vector<Ball>> createBallsFromBoundingBoxes(const std::vector<std::vector<BoundingBox>>& boundingBoxes, const std::vector<std::vector<int>>& classifier) {
    std::vector<std::vector<Ball>> allBalls; //Container of all balls

    // Loop through each image's bounding boxes and classifier
    for (size_t imgIndex = 0; imgIndex < boundingBoxes.size(); ++imgIndex) {
        const auto& boxes = boundingBoxes[imgIndex]; //BoundingBoxes of first image
        const auto& classifiers = classifier[imgIndex]; //IDs of first image

        std::vector<Ball> balls; //Vector of balls for first image
        for (size_t boxIndex = 0; boxIndex < boxes.size(); ++boxIndex) { //For each bbox
            const auto& box = boxes[boxIndex]; //Corresponding bbox
            int id = classifiers[boxIndex]; // Corresponding classifier ID
            Ball ball(box.x, box.y, box.width, box.height, id); //Create ball
            balls.push_back(ball);
        }
        allBalls.push_back(balls);
    }

    return allBalls;
}

/*
Since we need to compare our bounding boxes with the ground truth, we need to read the text files with the ground truth.
This function has been used for reading the text files and from this, to create a Ball.

Parameter
-filePaths: a vector of strings, corresponding to the paths of the text files

Return
-a vector of vector of Ball, one for each image
*/
std::vector<std::vector<Ball>> readBallsFromTextFiles(const std::vector<std::string>& filePaths) {
    std::vector<std::vector<Ball>> allBalls; //Container of all balls

    for (const auto& filePath : filePaths) {
        std::ifstream file(filePath); //Open file
        if (!file) {
            std::cerr << "Error: Unable to open file " << filePath << std::endl; //Error
            continue;
        }

        std::vector<Ball> balls;
        std::string line;
        while (std::getline(file, line)) { //Read line
            std::istringstream iss(line); //Split line
            float x, y, width, height; 
            int ID;
            if (iss >> x >> y >> width >> height >> ID) { //Extract values
                balls.push_back(Ball(x, y, width, height, ID)); //Create ball
            } else {
                std::cerr << "Error: Invalid format in file " << filePath << " at line: " << line << std::endl;
            }
        }

        allBalls.push_back(balls);
        file.close(); //Close file
    }

    return allBalls;
}

/*
This function has been used for displaying the bounding boxes or circles, with different colors for each ID.
We take the constructed vector of vector of Balls, one for each image, and we give as input the vector of images that we want to modify.
Moreover, we give as input also a boolean to decide whether we want to draw the bounding boxes or circles.

Parameter
-images: a vector of images
-allBalls: a vector of vector of Ball
-drawBoundingBoxes: a boolean, if true designs the bounding boxes, otherwise draws the circles

Return
-Nothing, however images have been modified in place by drawing the bounding boxes
*/
void showBoundingShapes(std::vector<cv::Mat>& images, const std::vector<std::vector<Ball>>& allBalls, bool drawBoundingBoxes) {
    for (size_t i = 0; i < images.size(); ++i) {
        auto& image = images[i];
        const auto& balls = allBalls[i];
        for (const auto& ball : balls) {
            // Determine color based on ID
            cv::Scalar color;
            switch (ball.getID()) {
                case 1:
                    color = cv::Scalar(255, 255, 255); // White
                    break;
                case 2:
                    color = cv::Scalar(0, 0, 0); // Black
                    break;
                case 3:
                    color = cv::Scalar(255, 0, 0); // Blue
                    break;
                case 4:
                    color = cv::Scalar(0, 0, 255); // Red
                    break;
                default:
                    color = cv::Scalar(0, 255, 255); // Default to Yellow
                    break;
            }

            if (drawBoundingBoxes) {
                // Draw rectangle
                cv::rectangle(image, 
                              cv::Point(ball.getX(), ball.getY()), 
                              cv::Point(ball.getX() + ball.getWidth(), ball.getY() + ball.getHeight()), 
                              color, 2);
            } else {
                // Draw circle
                int centerX = ball.getX() + ball.getWidth() / 2;
                int centerY = ball.getY() + ball.getHeight() / 2;
                int radius = std::min(ball.getWidth(), ball.getHeight()) / 2;
                cv::circle(image, 
                           cv::Point(centerX, centerY), 
                           radius, 
                           color, 2);
            }
        }
    }
}

/*
This function has been used for creating final masks as required by the project.
More specifically, we take as input images of the size of the original images, vector of vector of Balls (one for each image), and vector of vector of points (one for each image), corresponding to the table boundary.
Then for each image and for each vector of balls, we create a mask, assigning to each type of pixel the correct gray value in order to compute mIoU.

Parameter
-images: a vector of images
-allBalls: a vector of vector of Ball
-points: a vector of vector of points

Return
-Nothing, however images have been modified in place by drawing the masks
*/
void createFinalMasks(std::vector<cv::Mat>& images, const std::vector<std::vector<Ball>>& allBalls, const std::vector<std::vector<cv::Point2f>>& points) {
    for (size_t i = 0; i < images.size(); ++i) {
        auto& image = images[i];
        const auto& balls = allBalls[i];
        const auto& currentPoints = points[i];

        //General mask
        cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);

        if (currentPoints.size() >= 4) { //Only if it is possible to construct the rectangle
            std::vector<cv::Point> polygon;
            for (const auto& pt : currentPoints) {
                cv::Point point(static_cast<int>(pt.x), static_cast<int>(pt.y)); //Casting for using cv::fillConvexPoly
                polygon.push_back(point);
            }
            cv::fillConvexPoly(mask, polygon, 5); // Color the area inside the table with 5, as required
        }

        for (const auto& ball : balls) { //For each ball we determine the mask value based on its ID
            uchar maskValue;
            switch (ball.getID()) {
                case 1:
                    maskValue = 1; // White ball, as required by the project
                    break;
                case 2:
                    maskValue = 2; // Black ball, as required by the project
                    break;
                case 3:
                    maskValue = 3; // Solid ball, as required by the project
                    break;
                case 4:
                    maskValue = 4; // Striped ball
                    break;
            }

            //BBoxes and inscribed circle
            int x = ball.getX();
            int y = ball.getY();
            int width = ball.getWidth();
            int height = ball.getHeight();
            int radius = std::min(width, height) / 2;
            cv::Point center(x + width / 2, y + height / 2);

            cv::circle(mask, center, radius, maskValue, -1); // Correct color
        }

        // Apply the mask to the image
        image = mask;
    }
}

/*
This function has been used for showing in a visibile way the mask. Since the mask created is in only 5 values of gray, with this function we create a color version of the mask.
The function takes as input a vector of grayscale images, and returns a vector of color images, as required by the project.

Parameter
-grayscaleImages: a vector of grayscale images, in our case our output masks

Return
-colorImages: a vector of color images, in out case our output masks
*/
std::vector<cv::Mat> transformMasksToColor(const std::vector<cv::Mat>& grayscaleImages) {
    std::vector<cv::Mat> colorImages;

    for (const auto& grayscaleImage : grayscaleImages) { // For each grayscale image
        cv::Mat colorImage(grayscaleImage.size(), CV_8UC3, cv::Scalar(0, 0, 0)); // Create a color image

        // Iterate through each pixel in the grayscale image
        for (int i = 0; i < grayscaleImage.rows; ++i) {
            for (int j = 0; j < grayscaleImage.cols; ++j) {
                uchar pixelValue = grayscaleImage.at<uchar>(i, j);
                cv::Vec3b& colorPixel = colorImage.at<cv::Vec3b>(i, j);

                switch (pixelValue) {
                    case 0:
                        colorPixel = cv::Vec3b(0, 0, 0); // Background remains black
                        break;
                    case 1:
                        colorPixel = cv::Vec3b(255, 255, 255); // White ball
                        break;
                    case 2:
                        colorPixel = cv::Vec3b(0, 0, 0); // Black ball
                        break;
                    case 3:
                        colorPixel = cv::Vec3b(255, 0, 0); // Blue ball
                        break;
                    case 4:
                        colorPixel = cv::Vec3b(0, 0, 255); // Red ball
                        break;
                    case 5:
                        colorPixel = cv::Vec3b(9, 131, 0); // Area inside the table
                        break;
                    default:
                        colorPixel = cv::Vec3b(0, 0, 0); // Default to black
                        break;
                }
            }
        }
        // Add the color image to the output vector
        colorImages.push_back(colorImage);
    }

    return colorImages;
}


