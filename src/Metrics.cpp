//CODE WRITTEN BY LOVO MANUEL 2122856
#include "Metrics.h"


/*
This function has been used to calculate IoU between two balls.
Remembering what is IoU, it's the area of the intersection divided by the area of the union bewteen two bounding boxes.
Firstly we compute the intersection area, using also the min and max function since if there is not overlap the area will be 0.
Then we compute the union area that is the denominator of the IoU.

Parameter
-ball1: First ball
-ball2: Second ball

Return
-IoU, used then for determining a true or a false positive 
*/
float calculateIoU(const Ball& ball1, const Ball& ball2) {
    float x1 = std::max(ball1.getX(), ball2.getX()); // Get top left corner x for the intersection
    float y1 = std::max(ball1.getY(), ball2.getY()); // Get top left corner y for the intersection
    float x2 = std::min(ball1.getX() + ball1.getWidth(), ball2.getX() + ball2.getWidth()); // Get bottom right corner x for the intersection
    float y2 = std::min(ball1.getY() + ball1.getHeight(), ball2.getY() + ball2.getHeight()); // Get bottom right corner y for the intersection

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1); // Area of the intersection

    float union_area = ball1.computeArea() + ball2.computeArea() - intersection;

    return intersection / union_area;
}

/*
This function has been used once we have the value of IoU for each pair of balls.
We check if the IoU is greater than the threshold, if so we return true, otherwise we return false.

Parameter
-IoU: Value of IoU
-threshold: Threshold for determining a TP

Return
-True if IoU is greater than threshold
-False if IoU is not greater than threshold

*/
bool checkDetection(float IoU, float threshold) {
    return IoU > threshold;
}

/*
This function has been used for creating a vector that is then used to calculate mAP.
The idea is simple: we give as input a two vector of vector of balls (one for each image), that are the detected and true balls.
For each detected ball, we compute its IoU with all the true balls and we store the IoU, the index of the ball and the index of the detected ball in the vector.
Moreover, we give as input also two vector of int for recording the number of true and detected balls for each image for seeing if there are false negatives.

Parameter
-detected_balls: a vector of vector of balls
-true_balls: a vector of vector of balls
-num_ground_truths: a vector of int
-num_balls_detected: a vector of int
-total_true_balls: an int
-total_detected_balls: an int
*/
std::vector<std::vector<std::tuple<float, int, int>>> calculateIoUVectors(
    const std::vector<std::vector<Ball>>& detected_balls,
    const std::vector<std::vector<Ball>>& true_balls) {

    std::vector<std::vector<std::tuple<float, int, int>>> IoU_vectors;

    for (size_t i = 0; i < true_balls.size(); i++) { // For each image
        std::vector<std::tuple<float, int, int>> IoU_vector; // A vector for each image

        for (size_t j = 0; j < detected_balls[i].size(); j++) { // For each detected ball
            float max_IoU = -1.0f; 
            int correct_k = -1; //Used for remembering the correct index of the correspective ball
            int ID = -1;
            int ID_true = -1;


            for (size_t k = 0; k < true_balls[i].size(); k++) { // For each true ball
                float IoU = calculateIoU(detected_balls[i][j], true_balls[i][k]);
                if (IoU > max_IoU) { //Update the correct IoU
                    max_IoU = IoU;
                    correct_k = k;
                }
            }

            if (correct_k != -1) {
                ID = detected_balls[i][j].getID();
                ID_true = true_balls[i][correct_k].getID();
            }

            IoU_vector.push_back(std::make_tuple(max_IoU, ID, ID_true)); //Store the IoU, the index of the detected ball and the index of the true ball
        }
        IoU_vectors.push_back(IoU_vector); //Store the vector for each image
    }

    return IoU_vectors;
}

/*
This function has been used for computing AP for a given class.
The idea is to have a vector of tuple, where tuple are (TP, FP), where where there is a 1 means that there is a TP/FP and 0 the opposite.
Since for computing AP we need to iteratively compute the precision and recall, once a new element has been determined as TP or FP, we store this vector of tuple as this.
So, supposing that a classes has 10 elements, we will have 10 tuples in the vector.
Now that we have understood how it is given the input, we explain how it is computed.
For each tupla in the vector we add a the correspondence value to the total number of TP and FP.
Then we compute precision and recall with this values and we store them in a vector only if they are higher than the previous ones (since the interpolation 11 Pascal is computed as this). 
At the end, we finally compute the AP using the vector of precision.
*/
float calculateAP(const std::vector<std::tuple<int, int>>& tp_fp, int num_ground_truths) {
    int TP = 0, FP = 0;
    std::vector<float> precision(11, 0.0); //Vector of 11 elements to store 11 precision values
    std::vector<float> recall_levels = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    
    for (const auto& entry : tp_fp) {
        TP += std::get<0>(entry); //Take the first element of the tuple, if it's 1 this means that there is a TP
        FP += std::get<1>(entry); //Take the second element of the tuple, if it's 1 this means that there is a FP
        float precision_value = static_cast<float>(TP) / (TP + FP); //Compute precision
        float recall_value = static_cast<float>(TP) / num_ground_truths; //Compute recall

        for (int i = 0; i < 11; ++i) {
            if (recall_value >= recall_levels[i]) { //If the recall value is higher than the recall "past"
                precision[i] = std::max(precision[i], precision_value); //Update the precision
            }
        }
    }
    float ap = 0.0f;
    for (int i = 0; i < 11; ++i) {
        ap += precision[i];
    }
    ap /= 11.0f;
    return ap;
}

/*
This function has been used for computing AP for all classes.
The idea is to compute, for each image, a vector of AP, that is a vector containing the AP of each class. In our case will be a vector of 10 vector of 4 elements (one for each class).
We take as input the vector of vector of tuple, where each tuple is (IoU, ID, ID_true), remembering that represent, for each ball, the Iou with the correspective ground truth ball, its ID and the correct ID.
We sort the vector of tuple in order to have the highest IoU first, sinc eit's more probably that the classification is correct.
So, for each image we create a vector of 4 elements, where each element is the AP of the corresponding class.
Then we iterate for each ball and we check if it is TP or a FP based on the IoU and we add a tuple (1,0) (0,1) to the correspective vector, for that class.
Moreover, we have a vector of integers inside the for which contains, for each class, the total number of ground truth balls.
Then we compute the AP for each class using the function calculateAP, passing by input the vector of vector of tuple and the correct ground truth integer.
*/


std::vector<std::vector<float>> calculateAPs(
    const std::vector<std::vector<std::tuple<float, int, int>>>& IoU_vectors) {
    
    std::vector<std::vector<float>> AP_values(IoU_vectors.size(), std::vector<float>(4, 0.0f)); // Vector of vector for storing all AP for each image
    
    for (size_t i = 0; i < IoU_vectors.size(); ++i) { // For each image
        std::vector<std::vector<std::tuple<int, int>>> TP_FP(4); // Vector of vector of TP and FP tuples
        std::cout << " " << std::endl;
        std::cout << "Image " << i+1 << std::endl; 
        std::vector<int> TP_FP_classes(4, 0); // For recording TP and FP for each class

        // Sort IoU_vectors[i] based on the IoU value in descending order
        std::vector<std::tuple<float, int, int>> sorted_IoU_vector = IoU_vectors[i];
        std::sort(sorted_IoU_vector.begin(), sorted_IoU_vector.end(), 
            [](const std::tuple<float, int, int>& a, const std::tuple<float, int, int>& b) {
                return std::get<0>(a) > std::get<0>(b); // Sort by IoU value
        });

        for (size_t j = 0; j < sorted_IoU_vector.size(); ++j) { // For each ball
            for (int k = 1; k <= 4; ++k) {
                if (std::get<2>(sorted_IoU_vector[j]) == k) { // If the ID of the ground truth is the same as the ID of the ball
                    TP_FP_classes[k-1] = TP_FP_classes[k-1] + 1; // Increment
                }
                if (std::get<1>(sorted_IoU_vector[j]) == k) { // If the ID of the ball is the same as the ID
                    if (std::get<0>(sorted_IoU_vector[j]) > 0.5 && std::get<1>(sorted_IoU_vector[j]) == std::get<2>(sorted_IoU_vector[j])) { // If it's a TP
                        TP_FP[k-1].emplace_back(1, 0); // Add the tuple
                    } else { // It's an FP
                        TP_FP[k-1].emplace_back(0, 1);
                    }
                }
            }
        }

        float mAP = 0.0f;
        for (int class_id = 0; class_id < 4; ++class_id) { // For each class  
            float ap = calculateAP(TP_FP[class_id], TP_FP_classes[class_id]); // Compute AP
            AP_values[i][class_id] = ap;
            std::cout << "AP for class " << class_id << ": " << ap << std::endl;
            mAP += ap;
        }
        mAP /= 4.0f;
        std::cout << " " << std::endl;
        std::cout << "Mean AP: " << mAP << std::endl;
    }

    return AP_values;
}


/*
This function has been used for computing the IoU for a single class in a single pair of segmentation masks.
More specifically, it counts the number of pixels in common between the ground truth and the predicted masks, and the number of pixels in the union of the masks.
Then, by using these values, we compute the IoU.

Parameter
-groundTruth: an image, in our case the segmentation masks truth
-predicted: an image, in our case the segmentation masks predicted
-classId: the ID of the class we are interested in

Return
-the IoU for the class
*/
double computeIoUForClass(const cv::Mat& groundTruth, const cv::Mat& predicted, int classId) {
    if (groundTruth.size() != predicted.size() || groundTruth.type() != CV_8UC1 || predicted.type() != CV_8UC1) {
        std::cerr << "Error: Images must be of the same size and type (CV_8UC1)." << std::endl;
        return -1.0;
    }
    int intersection = 0;
    int union_ = 0;

    for (int i = 0; i < groundTruth.rows; ++i) {
        for (int j = 0; j < groundTruth.cols; ++j) {
            bool gt = groundTruth.at<uchar>(i, j) == classId; //Check if the pixel belongs to the class
            bool pr = predicted.at<uchar>(i, j) == classId;

            if (gt || pr) { // If at least one pixel belongs to the class
                union_++;
            }
            if (gt && pr) { // If pixels are in the same class
                intersection++;
            }
        }
    }

    if (union_ == 0) {
        return -1.0; // This class does not exist in the union of the masks
    }

    return static_cast<double>(intersection) / union_;
}

/*
This function has been used for computing the mean IoU for all classes in a single pair of segmentation masks.
It simply apply the computeIoUForClass function for each class and take the mean.

Parameter
-groundTruth: an image, in our case the segmentation masks truth
-predicted: an image, in our case the segmentation masks predicted

Return
-the mean IoU
*/
double computeMeanIoU(const cv::Mat& groundTruth, const cv::Mat& predicted) {
    if (groundTruth.size() != predicted.size() || groundTruth.type() != CV_8UC1 || predicted.type() != CV_8UC1) {
        std::cerr << "Error: Images must be of the same size and type (CV_8UC1)." << std::endl;
        return -1.0;
    }

    std::vector<int> classes = {0,1,2,3,4,5}; // Classes we are interested in

    // Compute IoU for each class
    double totalIoU = 0.0;
    for (int classId : classes) {
        double iou = computeIoUForClass(groundTruth, predicted, classId);
        std::cout << "IoU for class " << classId << ": " << iou << std::endl;
        if (iou >= 0.0) {
            totalIoU += iou;
        }
    }

    return totalIoU / 6.0;
}

/*
This function has been used for computing the mean IoU for all classes in a set of segmentation masks.
It simply apply the computeMeanIoU function for each pair of masks in the set. 
The result is printed in a vector and it's also shared as output.

Parameter
-groundTruths: a vector of images, in our case the segmentation masks truth
-predicted: a vector of images, in our case the segmentation masks predicted
*/
std::vector<double> vectormIoU(const std::vector<cv::Mat>& groundTruths, const std::vector<cv::Mat>& predicted) {
    std::vector<double> mIoU(groundTruths.size());
    for (int i = 0; i < groundTruths.size(); ++i) {
        std::cout << " " << std::endl;
        std::cout<<"Image " << i+1 << std::endl;
        mIoU[i] = computeMeanIoU(groundTruths[i], predicted[i]);
        std::cout << " " << std::endl;
        std::cout << " mIoU: " << mIoU[i] << std::endl;
    }
    return mIoU;
}


