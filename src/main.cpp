//Author: Martina Boscolo Bacheto

/**THIS FILE CONTAINS THE MAINS OF ALL THE THREE TASKS ORGANIIZED IN FUNCTIONS: SEGMENTATION AND
 * CLASSIFICATION, TRACKING SYSTEM, METRICS COMPUTATION.
 * THIS IS USEFUL FOR EXECUTING ALL THE TASKS AT ONCE.
*/
#include "main.h"

using namespace cv;
using namespace std;

int main_tracker(){
    String originaVideoFolder = "../Input_videos";
    String videoPattern = "*.mp4";
    vector<String> allVideoPaths = getPaths(originaVideoFolder, videoPattern);

    String firstBboxesFolder = "../First_frames_detected_bboxes";
    String bboxesFirstPattern = "*.txt";
    vector<String> allBoxesPaths = getPaths(firstBboxesFolder, bboxesFirstPattern);

    String verticesFolder = "../Table_vertices_for_tracking";
    String verticesPattern = "*.txt";
    vector<String> allVerticesPaths = getPaths(verticesFolder, verticesPattern);

    vector<vector<Point2f>> allOrderedTablevertices = tableOrientation();

    for (size_t i = 0; i < allVideoPaths.size(); i++) {
        VideoCapture cap;
        if (!initializeVideoCapture(allVideoPaths[i], cap))
            return -1;

        VideoWriter outputVideo;
        cv::String t = (i + 1 < 10) ? "0" + std::to_string(i + 1) : std::to_string(i + 1);
       
        if (!initializeOutputVideo("../Output_videos/output_video_" + t+ ".mp4", outputVideo, cap))
            return -1;

         vector<Ball> boundingBoxes;
        if (!readBoundingBoxes(allBoxesPaths[i], boundingBoxes))
            return -1;

        
		cv::Point2f srcPoints[4];

		std::ifstream file(allVerticesPaths[i]);
        vector<Point2f> realWorldPoints = allOrderedTablevertices[i];

        Mat perspectiveMatrix = computePerspectiveMatrix(realWorldPoints);

        Mat initialFrame;
        cap >> initialFrame;
        if (initialFrame.empty())
        {
            cerr << "Error reading frame from video." << endl;
            return -1;
        }

        vector<Ptr<Tracker>> trackers;
        vector<Rect> rois;
        vector<int> categoryIDs;
        vector<vector<Point>> paths;
        initializeTrackers(initialFrame, boundingBoxes, trackers, rois, categoryIDs, paths);

        String lastBbFileName = ("../Last_frames_detected_bboxes/last_frame_detected_bboxes_"+ t+".txt");
        String last2DMap = ("../Last_frame_2D_maps/last_frame_2D_map_"+ to_string(i+1)+".png");
        processFrames(cap, outputVideo, perspectiveMatrix, trackers, rois, categoryIDs, lastBbFileName, last2DMap);

        cap.release();
        outputVideo.release();
        destroyAllWindows();
    }
    return 0;
}

 void main_field_balls(){
    //UPLOAD IMAGES IN A VECTOR

    std::vector<cv::Mat> images; //Creation of a vector of images
    std::vector<cv::Mat> masks;
    std::vector<cv::Mat> final_masks;
    std::vector<cv::Mat> output;
    std::vector<std::string> imagePaths = getImagePaths("../First_frames_images", "*.png");

    //std::vector<std::string> imagePaths = getImagePaths("Last_frames_images", "*.png");

    //BY CHOOSING THE CORRECT FOLDER, YOU CAN CHOOSE BETWEEN THE FIRST AND LAST FRAMES

    images = readImages(imagePaths, true); //We read the images and save in a vector, boolean true for RGB and false for gray
    cv::Mat initial_images = combinedImages(images); 
    cv::imshow("Original images", initial_images);

    int lowH = 71;
    int lowS = 66;
    int lowV = 66;
    int highH = 126;
    int highS = 255;
    int highV = 255;
    std::vector<cv::Mat> hsvMask = HSVsegmentation(images, lowH, lowS, lowV, highH, highS, highV); //We apply the HSV segmentation to the images
    for (const auto& image : images) {
        output.push_back(image.clone());
        masks.push_back(image.clone());
        cv::Mat grayOutput(image.size(), CV_8UC1);
        final_masks.push_back(grayOutput);
    }

    int dimension_kernel_opening = 5;
    int dimension_kernel_smoothing = 9;
    int lowThreshold = 207;
    int highThreshold = 358;
    std::vector<cv::Mat> noiseRemovedImages = vectorErosion(hsvMask, dimension_kernel_opening); //Post processing
    std::vector<cv::Mat> noiseRemovedImages2 = vectorDilation(noiseRemovedImages, dimension_kernel_opening);
    std::vector<cv::Mat> smoothed_images = vectorSmoothing(noiseRemovedImages2, dimension_kernel_smoothing);
    std::vector<cv::Mat> canny_image = vectorCanny(smoothed_images, lowThreshold , highThreshold);


    int divisory = 102;
    int threshold_houghlines = 10;
    int number_lines = 4;
    std::vector<std::vector<cv::Point2f>> pointsList = vectorHoughLines(canny_image, output, masks, divisory, threshold_houghlines, number_lines); //Hough lines

    savePointsToFile(pointsList, "points.txt"); //For saving points in a file for later use in the metrics

    std::vector<cv::Mat> masks_1;
    std::vector<cv::Mat> masks_2;
    std::vector<cv::Mat> masks_3;
    std::vector<cv::Mat> masks_4;
    std::vector<cv::Mat> masks_5;
    std::vector<cv::Mat> black_masks;

    std::vector<cv::Vec3b> color_tables;
    int RoiSize = 10;
    for(int i=0; i<images.size(); i++) {
        cv::Vec3b color =  computeMedianColorOfCenteredROI(masks[i], RoiSize); 
        color_tables.push_back(color);
    }

    int threshold_RGB1 = 49;
    int threshold_RGB2 = 129;
    int threshold_RGB3 = 36;
    int threshold_RGB4 = 90;
    int threshold_RGB5 = 25;
    int threshold_black = 120;
    masks_1 = vectorMaskRGB(masks, color_tables , threshold_RGB1); //Different RGB masks for finding all balls
    masks_2 = vectorMaskRGB(masks, color_tables , threshold_RGB2);
    masks_3 = vectorMaskRGB(masks, color_tables , threshold_RGB3);
    masks_4 = vectorMaskRGB(masks, color_tables , threshold_RGB4);
    masks_5 = vectorMaskRGB(masks, color_tables , threshold_RGB5);
    black_masks = maskBlack(masks, threshold_black);

    int dimension_structElem1 = 5;
    int dimension_structElem2 = 11;
    int dimension_structElem3 = 3;
    int dimension_structElem4 = 11;
    int dimension_structElem5 = 7;
    masks_1 = vectorDilation(masks_1, dimension_structElem1); //Post processing
    masks_2 = vectorDilation(masks_2, dimension_structElem2);
    masks_3 = vectorDilation(masks_3, dimension_structElem3);
    masks_4 = vectorDilation(masks_4, dimension_structElem4);
    masks_5 = vectorDilation(masks_5, dimension_structElem5);
    masks_5 = vectorErosion(masks_5, dimension_structElem5);

    int kernelSize_blackmask = 5;
    int dimension_structElem_blackmask = 7;
    black_masks = vectorMedianBlur(black_masks, kernelSize_blackmask);
    black_masks = vectorDilation(black_masks, dimension_structElem_blackmask);
    
    int smoothing_mask1 = 9;
    int smoothing_mask2 = 11;
    int smoothing_mask3 = 9;
    int smoothing_mask4 = 7;
    int smoothing_mask5 = 9;
    int smoothing_black_masks = 5;

    masks_1 = vectorSmoothing(masks_1, smoothing_mask1); //Post processing
    masks_2 = vectorSmoothing(masks_2, smoothing_mask2);
    masks_3 = vectorSmoothing(masks_3, smoothing_mask3);
    masks_4 = vectorSmoothing(masks_4, smoothing_mask4);
    masks_5 = vectorSmoothing(masks_5, smoothing_mask5);
    black_masks = vectorSmoothing(black_masks, smoothing_black_masks);

    int canny_threshold1_1 = 195;
    int canny_threshold2_1 = 428;
    int canny_threshold1_2 = 10;
    int canny_threshold2_2 = 11;
    int canny_threshold1_4 = 195;
    int canny_threshold2_4 = 428;

    std::vector<cv::Mat> canny_masks_1 = vectorCanny(masks_1, canny_threshold1_1, canny_threshold2_1);
    std::vector<cv::Mat> canny_masks_2 = vectorCanny(masks_2, canny_threshold1_2, canny_threshold2_2);
    std::vector<cv::Mat> canny_masks_3 = vectorCanny(masks_3, canny_threshold1_2, canny_threshold2_2);
    std::vector<cv::Mat> canny_masks_4 = vectorCanny(masks_4, canny_threshold1_4, canny_threshold2_4);
    std::vector<cv::Mat> canny_masks_5 = vectorCanny(masks_5, canny_threshold1_2, canny_threshold2_2);

    int hough_dp1 = 1;
    int hough_minDist1 = 20;
    int hough_cannyParam1 = 80;
    int hough_threshold1 = 14;
    int hough_minRadius1 = 8;
    int hough_maxRadius1 = 17;

    int hough_dp2 = 1;
    int hough_minDist2 = 19;
    int hough_cannyParam2 = 80;
    int hough_threshold2 = 9;
    int hough_minRadius2 = 10;
    int hough_maxRadius2 = 16;

    int hough_dp3 = 1;
    int hough_minDist3 = 23;
    int hough_cannyParam3 = 80;
    int hough_threshold3 = 14;
    int hough_minRadius3 = 9;
    int hough_maxRadius3 = 17;

    int hough_dp4 = 1;
    int hough_minDist4 = 20;
    int hough_cannyParam4 = 80;
    int hough_threshold4 = 13;
    int hough_minRadius4 = 7;
    int hough_maxRadius4 = 14;

    int hough_dp5 = 1;
    int hough_minDist5 = 20;
    int hough_cannyParam5 = 80;
    int hough_threshold5 = 10;
    int hough_minRadius5 = 6;
    int hough_maxRadius5 = 9;

    int hough_dp6 = 1;
    int hough_minDist6 = 20;
    int hough_cannyParam6 = 50;
    int hough_threshold6 = 14;
    int hough_minRadius6 = 5;
    int hough_maxRadius6 = 10;

    int filter_nearHoles1 = 55;
    int filter_nearHoles2 = 31;
    int filter_nearHoles3 = 31;
    int filter_nearHoles4 = 31;
    int filter_nearHoles5 = 55;
    int filter_nearHoles6 = 30;

    int distanceThreshold = 11;
    int filter_colorThreshold2 = 16;
    int filter_colorThreshold3 = 20;
    int filter_colorThreshold4 = 30;
    int filter_colorThreshold5 = 100;

    int pointsNearLinesThreshold1 = 15;
    int pointsNearLinesThreshold2 = 7;
    int pointsNearLinesThreshold3 = 24;
    int pointsNearLinesThreshold4 = 30;

    double refine_scaleFactor1 = 1.3;
    int refine_houghParam = 8;
    double refine_scaleFactorMinRadius = 0.5;
    double refine_scaleFactorMaxRadius = 0.75;

    int replicas_distanceThreshold1 = 11;
    int replicas_distanceThreshold2 = 11;
    int replicas_distanceThreshold3 = 15;
    int replicas_distanceThreshold4 = 13;

    int size;

    if(canny_masks_1.size() == canny_masks_2.size() && canny_masks_1.size() == canny_masks_3.size() && canny_masks_1.size() == canny_masks_4.size()) {
        size = canny_masks_1.size();
    }

    std::vector<std::vector<cv::Vec3f>> final_circles;

    for(int i=0; i<size; i++) {

        cv::Mat immagine = masks[i].clone();
        std::vector<cv::Vec3f> circles_1 = houghCircles(canny_masks_1[i], hough_dp1, hough_minDist1, hough_cannyParam1, hough_threshold1, hough_minRadius1, hough_maxRadius1);
        filterCircles_nearHoles(circles_1, pointsList[i], filter_nearHoles1);
        //Finding circles and then removing those near Holes

        std::vector<cv::Vec3f> circles_2 = houghCircles(canny_masks_2[i], hough_dp2, hough_minDist2, hough_cannyParam2, hough_threshold2, hough_minRadius2, hough_maxRadius2);
        filterCircles_nearHoles(circles_2, pointsList[i], filter_nearHoles2);

        std::vector<cv::Vec3f> circles_3 = houghCircles(canny_masks_3[i], hough_dp3, hough_minDist3, hough_cannyParam3, hough_threshold3, hough_minRadius3, hough_maxRadius3);
        filterCircles_nearHoles(circles_3, pointsList[i], filter_nearHoles3);

        std::vector<cv::Vec3f> circles_4 = houghCircles(canny_masks_4[i], hough_dp4, hough_minDist4, hough_cannyParam4, hough_threshold4, hough_minRadius4, hough_maxRadius4);
        filterCircles_nearHoles(circles_4, pointsList[i], filter_nearHoles4);

        std::vector<cv::Vec3f> circles_5 = houghCircles(canny_masks_5[i], hough_dp5, hough_minDist5, hough_cannyParam5, hough_threshold5, hough_minRadius5, hough_maxRadius5);
        filterCircles_nearHoles(circles_5, pointsList[i], filter_nearHoles5);

        std::vector<cv::Vec3f> circles_6 = houghCircles(black_masks[i], hough_dp6, hough_minDist6, hough_cannyParam6, hough_threshold6, hough_minRadius6, hough_maxRadius6);
        filterCircles_nearHoles(circles_6, pointsList[i], filter_nearHoles6);

        std::vector<cv::Vec3f> refined_circles_1 = filterCircles_replicas_1(masks[i], circles_1, circles_2, color_tables[i], distanceThreshold); //Merging circles 1 and 2
        filterCirclesByColor(masks[i], color_tables[i], filter_colorThreshold2, refined_circles_1);
        filterCirclesByColor(masks[i], color_tables[i], filter_colorThreshold3, circles_3);
        filterCirclesByColor(masks[i], color_tables[i], filter_colorThreshold4, circles_5); //Removing circles with color criterion
        filterCirclesByColor(masks[i], color_tables[i], filter_colorThreshold5, circles_4);

        std::vector<cv::Vec3f> refined_circles_2 = pointsNearLines(circles_3, pointsList[i], pointsNearLinesThreshold1); //Removing circles near lines
        std::vector<cv::Vec3f> refined_circles_3 = pointsNearLines(refined_circles_1, pointsList[i], pointsNearLinesThreshold2);
        std::vector<cv::Vec3f> circles_4_1 = pointsNearLines(circles_4, pointsList[i], pointsNearLinesThreshold3);
        std::vector<cv::Vec3f> circles_5_1 = pointsNearLines(circles_5, pointsList[i], pointsNearLinesThreshold4);
        std::vector<cv::Vec3f> refined_circles_4 = refineCircles_ROI(masks[i], refined_circles_3, refine_scaleFactor1, refine_houghParam, refine_scaleFactorMinRadius, refine_scaleFactorMaxRadius); //Removing circles using Hough on a ROI centered in each ball detected
        std::vector<cv::Vec3f> refined_circles_5 = filterCircles_replicas_1(masks[i], refined_circles_2, refined_circles_4, color_tables[i], replicas_distanceThreshold1); //Merging different circles
        std::vector<cv::Vec3f> refined_circles_6 = refineCircles_replicas_2(circles_4, refined_circles_5, replicas_distanceThreshold2); //Merging different circles
        std::vector<cv::Vec3f> refined_circles_7 = filterCircles_replicas_1(masks[i], refined_circles_6, circles_5_1, color_tables[i], replicas_distanceThreshold3);

        std::vector<cv::Vec3f> final_circle = refineCircles_replicas_2(circles_6, refined_circles_7, replicas_distanceThreshold4);

        final_circles.push_back(final_circle);
    }

    std::vector<std::vector<Circle>> final_circles_ = createCircles(final_circles);               // Creation of Circle objects
    std::vector<std::vector<BoundingBox>> boundingBoxes = createBoundingBoxes(final_circles_, 1); // Creation of BoundingBox objects
    std::vector<std::vector<int>> classified = classifiedVector(images, boundingBoxes);           // Classification of BoundingBox objects

    std::vector<std::vector<Ball>> balls = createBallsFromBoundingBoxes(boundingBoxes, classified); // Creation of Ball objects

    for (int j = 0; j < balls.size(); j++)
    {
        cv::String t = (j + 1 < 10) ? "0" + std::to_string(j + 1) : std::to_string(j + 1);

        std::ofstream outFile("../First_frames_detected_bboxes/first_frame_detected_bboxes_" + t + ".txt"); // print vertices in a file to be used in next steps
        if (outFile.is_open())
        {
            for (const auto &ball : balls[j])
            {
                outFile << ball.getX() << " " << ball.getY() << " " << ball.getWidth() << " " << ball.getHeight() << " " << ball.getID() << "\n";
            }
            outFile.close(); // Close the file after writing all balls for the current j
        }
        else
        {
            std::cerr << "Unable to open file for writing." << std::endl;
        }
    }

    showBoundingShapes(output, balls, true); //Display of Bounding Boxes/Circles, depending on the boolean
    cv::Mat final_images = combinedImages(output);
    cv::imshow("Output Images", final_images);

    createFinalMasks(final_masks, balls, pointsList); //Creation of final masks
    std::vector<cv::Mat > final_output = transformMasksToColor(final_masks); //Transformation of masks to color

    cv::Mat final_image = combinedImages(final_output);
    cv::imshow("Output masks", final_image);

    cv::waitKey(0);
    destroyAllWindows();

}
void main_metrics(){
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
    std::vector<std::vector<cv::Point2f>> points_list = readPointsFromFile("../Points.txt");
    createFinalMasks(detectedMasks_first_frames, detected_balls_first_frames, points_list);
    createFinalMasks(detectedMasks_last_frames, detected_balls_last_frames, points_list);

    //From paths we create our masks

    std::cout<<"FIRST FRAMES"<<std::endl;
    std::cout<<"--------------"<<std::endl;
    std::vector<double> mIoU_vector_firstframes = vectormIoU(groundMasks_first_frames, detectedMasks_first_frames);

    std::cout<<"LAST FRAMES"<<std::endl;
    std::cout<<"--------------"<<std::endl;
    std::vector<double> mIoU_vector_lastframes = vectormIoU(groundMasks_last_frames, detectedMasks_last_frames);

}

int main(int argc, char** argv)
{
    main_field_balls();
    main_tracker();
    main_metrics();

    return 0;

}