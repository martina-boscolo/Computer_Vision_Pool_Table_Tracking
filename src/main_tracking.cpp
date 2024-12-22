//Author: Martina Boscolo Bacheto
#include "tracking.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
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