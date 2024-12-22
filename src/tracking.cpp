//Author: Martina Boscolo Bacheto

#include "tracking.h"

using namespace std;
using namespace cv;


/**
 * @brief Draws a ball in the 2D map after computation of the poprospective transformation of the original ball
 * @param image The image on which to draw the balls.
 * @param perspectiveMatrix The perspective transformation matrix.
 * @param bbox The bounding box of the ball.
 * @param id The ID of the ball, used to determine its color.
 */
void drawTransformedBalls(Mat& image, const Mat& perspectiveMatrix, const Rect bbox, int id)
{
    map<int, Scalar> colors = {
        {1, Scalar(255, 255, 255)}, //white ball
        {2, Scalar(0, 0, 0)}, //black ball
        {3, Scalar(255, 0, 0)}, //striped ball
        {4, Scalar(0, 0, 255)} //solid ball
        }; //colors for the balls in the 2D map

    Scalar contourColor = Scalar(0, 0, 0);
    Point2f center(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);
    vector<Point2f> srcPoints = { center };
    vector<Point2f> dstPoints(1);
    perspectiveTransform(srcPoints, dstPoints, perspectiveMatrix);
    Scalar fillColor = colors[id];
    int radius = 5;
    circle(image, dstPoints[0], radius, contourColor, -1); //contour of the ball
    circle(image, dstPoints[0], radius - 1, fillColor, -1); //ball
}

/**
 * @brief Draws the trajectory of the ball in the 2D map starting from the center of the ball in the 
 * original image and computing the prospective transformation. 
 * A point on the 2D representation is left in correspondece of the center of the ball for each frame.
 * @param image The image on which to draw the path.
 * @param perspectiveMatrix The perspective transformation matrix.
 * @param trackPoints Vector of points representing the path.
 */
void drawTransformedPath(Mat& image, const Mat& perspectiveMatrix, const vector<Point2f> trackPoints)
{
    vector<Point2f> dstPoints(trackPoints.size());
    perspectiveTransform(trackPoints, dstPoints, perspectiveMatrix);
    for (size_t i = 0; i < dstPoints.size(); i++)
    {
        circle(image, dstPoints[i], 1, Scalar(0, 0, 0), -1); //point on the map representing the urrent position
    }
}

/**
 * @brief Reads bounding box information from a file.
 * @param bboxFile Path to the file containing bounding box information.
 * @param balls Vector to store the Ball objects created from the file data.
 * @return true if reading was successful, false otherwise.
 */
bool readBoundingBoxes(const string& bboxFile, vector<Ball>& balls)
{
    ifstream inFile(bboxFile);
    if (!inFile)
    {
        cerr << "Error opening bounding box file." << endl;
        return false;
    }

    string line;
    while (getline(inFile, line))
    {
        istringstream iss(line);
        float x, y, width, height;
        int id;
        if (!(iss >> x >> y >> width >> height >> id))
        {
            cerr << "Error reading bounding box parameters." << endl;
            return false;
        }
        balls.emplace_back(x, y, width, height, id);
    }

    inFile.close();
    return true;
}


/**
 * @brief Initializes video capture from a file.
 * @param videoFile Path to the video file.
 * @param cap VideoCapture object to be initialized.
 * @return true if initialization was successful, false otherwise.
 */
bool initializeVideoCapture(const string& videoFile, VideoCapture& cap)
{
    cap.open(videoFile);
    if (!cap.isOpened())
    {
        cerr << "Error opening video file." << endl;
        return false;
    }
    return true;
}

/**
 * @brief Computes a perspective transformation matrix.
 * @param realWorldPoints Vector of points in the real world coordinate system.
 * @return Mat The computed perspective transformation matrix.
 */
Mat computePerspectiveMatrix(vector<Point2f> &realWorldPoints)
{
    std::vector<Point2f> destinationPoints = {
        Point2f(21, 387),
        Point2f(342, 387),
        Point2f(342, 555),
        Point2f(21, 555)
    }; //points hardcoded specifically for the table 2D map selected

    return getPerspectiveTransform(realWorldPoints, destinationPoints);
}

/**
 * @brief Initializes a VideoWriter for output.
 * @param outputVideoFile Path to the output video file.
 * @param outputVideo VideoWriter object to be initialized.
 * @param cap VideoCapture object used to get video properties.
 * @return true if initialization was successful, false otherwise.
 */
bool initializeOutputVideo(const string& outputVideoFile, VideoWriter& outputVideo, const VideoCapture& cap)
{
    int codec = VideoWriter::fourcc('m', 'p', '4', 'v'); //writes an mp4 video
    double fps = cap.get(CAP_PROP_FPS);
    Size frameSize(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));
    outputVideo.open(outputVideoFile, codec, fps, frameSize);
    if (!outputVideo.isOpened())
    {
        cerr << "Error opening output video file for writing." << endl;
        return false;
    }
    return true;
}

/**
 * @brief Initializes trackers for each ball in the first frame.
 * @param frame The first frame of the video.
 * @param balls Vector of Ball objects.
 * @param trackers Vector to store the created Tracker objects.
 * @param rois Vector to store the regions of interest.
 * @param categoryIDs Vector to store the category IDs of the balls.
 * @param paths Vector to store the paths of the balls.
 */
void initializeTrackers(const Mat& frame, const vector<Ball>& balls, vector<Ptr<Tracker>>& trackers, vector<Rect>& rois, vector<int>& categoryIDs, vector<vector<Point>>& paths)
{
    for (const auto& ball : balls)
    {
        rois.emplace_back(ball.getX(), ball.getY(), ball.getWidth(), ball.getHeight());
        trackers.push_back(TrackerCSRT::create()); 
        categoryIDs.push_back(ball.getID());
        paths.emplace_back();
    }//for each ball a TrackerCSRT get initialized
    for (size_t i = 0; i < rois.size(); ++i)
    {
        trackers[i]->init(frame, rois[i]);
    }
}

/**
 * @brief Loads and resizes the table map image.
 * @param imagePath Path to the source image.
 * @param desiredHeight Desired height of the resized image.
 * @param newWidth Reference to store the new width of the resized image.
 * @return Mat The resized image.
 */
Mat loadAndResizeTableMap(const string& imagePath, int desiredHeight, int& newWidth) {
    Mat srcImage = imread(imagePath);
    if (srcImage.empty()) {
        cerr << "Could not load source image" << endl;
        return Mat();
    }

    double aspectRatio = static_cast<double>(srcImage.cols) / srcImage.rows; 
    newWidth = static_cast<int>(desiredHeight * aspectRatio); //resizes mantaining the original ratio
    Mat resizedSrcImage;
    resize(srcImage, resizedSrcImage, Size(newWidth, desiredHeight));
    return resizedSrcImage;
}

/**
 * @brief Processes a single frame of the video.
 * @param frame The current frame to process.
 * @param resizedSrcImage The resized source image to overlay.
 * @param roi Region of interest for the overlay.
 * @param trackers Vector of Tracker objects.
 * @param rois Vector of regions of interest.
 * @param categoryIDs Vector of category IDs.
 * @param perspectiveMatrix The perspective transformation matrix.
 * @param trackPoints Vector to store tracking points.
 * @param pathFrame Frame for drawing paths.
 */
void processSingleFrame(Mat& frame, const Mat& resizedSrcImage, const Rect& roi, vector<Ptr<Tracker>>& trackers, 
     vector<Rect>& rois, const vector<int>& categoryIDs, const Mat& perspectiveMatrix, vector<Point2f>& trackPoints, Mat& pathFrame) {

    resizedSrcImage.copyTo(frame(roi));

    for (size_t i = 0; i < trackers.size(); ++i) {
        Point center(rois[i].x + rois[i].width / 2, rois[i].y + rois[i].height / 2);
        trackPoints.push_back(center); 
        drawTransformedPath(frame, perspectiveMatrix, trackPoints);
    }
    //separated for for avoiding path dots to be drawn over the balls
    for (size_t i = 0; i < trackers.size(); ++i) {
        drawTransformedBalls(frame, perspectiveMatrix, rois[i], categoryIDs[i]); //draw balls in the 2D map
    }
    add(frame, pathFrame, frame);
}


/**
 * @brief Writes the bounding box information and the final 2D map of the last frame.
 * @param rois Vector of regions of interest.
 * @param categoryIDs Vector of category IDs.
 * @param nameBB Name of the file to write bounding box information.
 * @param name2DMap Name of the file to write the final 2D map.
 * @param lastframe The last frame of the video.
 * @brief Writes the bounding box information and the final 2D map of the last frame.
 * 
 */
void writeLastFrameBB(const vector<Rect>& rois, const vector<int>& categoryIDs, const String& nameBB, const String& name2DMap, const Mat& lastframe) {
    ofstream outFile(nameBB);
    for (size_t i = 0; i < rois.size(); ++i) {
        outFile << rois[i].x << " " << rois[i].y << " " << rois[i].width << " " << rois[i].height << " " << categoryIDs[i] << endl;
    }
    outFile.close();
    imwrite(name2DMap, lastframe);
}

/**
 * @brief Processes all frames in the video.
 * @param cap VideoCapture object for reading frames.
 * @param outputVideo VideoWriter object for writing output.
 * @param perspectiveMatrix The perspective transformation matrix.
 * @param trackers Vector of Tracker objects.
 * @param rois Vector of regions of interest.
 * @param categoryIDs Vector of category IDs.
 * @param nameBB Name of the file to write bounding box information.
 * @param name2DMap Name of the file to write the final 2D map.
 */
void processFrames(VideoCapture& cap, VideoWriter& outputVideo, const Mat& perspectiveMatrix, vector<Ptr<Tracker>>& trackers, vector<Rect>& rois, vector<int>& categoryIDs, const String nameBB, const String name2DMap) {
    Mat frame, pathFrame = Mat::zeros(cap.get(CAP_PROP_FRAME_HEIGHT), cap.get(CAP_PROP_FRAME_WIDTH), CV_8UC3);
    vector<Point2f> trackPoints;
    int frameCounter = 0;
    const int interval = 1;

    int newWidth;
    Mat resizedSrcImage = loadAndResizeTableMap("../table.png", 200, newWidth); //reads the 2d map
    if (resizedSrcImage.empty()) return;

    //image positioned in the frame such that it is 5px from the bottom and the left side
    int x = 5;
    int y = cap.get(CAP_PROP_FRAME_HEIGHT) - 200 - 5;
    Rect roi(Point(x, y), resizedSrcImage.size());

    Mat lastframe;

    while (cap.isOpened()) {
        cap >> frame;
        if (frame.empty()) break;

        if (frameCounter % interval == 0) {
            for (size_t i = 0; i < trackers.size(); ++i) {
                trackers[i]->update(frame, rois[i]);
            }
        }
        processSingleFrame(frame, resizedSrcImage, roi, trackers, rois, categoryIDs, perspectiveMatrix, trackPoints, pathFrame);
        outputVideo << frame;
        imshow("Tracking system", frame);
        if (waitKey(1) == 27) break;
        lastframe = frame;
        frameCounter++;
    }
    writeLastFrameBB(rois, categoryIDs, nameBB, name2DMap, lastframe);
}

