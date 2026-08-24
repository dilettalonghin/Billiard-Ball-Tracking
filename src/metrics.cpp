#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <iomanip>

using namespace cv;
using namespace std;

// Structure to hold bounding box information
struct BoundingBox {
    int x, y, width, height, categoryID;
    bool isTruePositive = false;

    // Constructor
    BoundingBox(int x, int y, int width, int height, int categoryID)
        : x(x), y(y), width(width), height(height), categoryID(categoryID), isTruePositive(false) {}
};

// Function to read bounding boxes from a file
vector<BoundingBox> readBoundingBoxes(const string& filename) {
    vector<BoundingBox> boxes;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Could not open file: " << filename << endl;
        return boxes;
    }

    int x, y, width, height, categoryID;
    while (file >> x >> y >> width >> height >> categoryID) {
        boxes.push_back(BoundingBox(x, y, width, height, categoryID));
    }

    return boxes;
}

// Function to calculate IoU between two bounding boxes
// It calculates the Intersection over Union (IoU) between two bounding boxes.
double calculateIoU(const BoundingBox& box1, const BoundingBox& box2) {

    int x1 = max(box1.x, box2.x);
    int y1 = max(box1.y, box2.y);
    int x2 = min(box1.x + box1.width, box2.x + box2.width);
    int y2 = min(box1.y + box1.height, box2.y + box2.height);

    int intersectionArea = max(0, x2 - x1) * max(0, y2 - y1);
    int box1Area = box1.width * box1.height;
    int box2Area = box2.width * box2.height;

    double iou = static_cast<double>(intersectionArea) / (box1Area + box2Area - intersectionArea);
    return iou;
}

// Function to calculate Precision and Recall based on ground truth and predictions.
//It uses a IoU threshold set to 0.5
vector<pair<double, double>> calculatePrecisionRecall(vector<BoundingBox>& groundTruth, vector<BoundingBox>& predictions, double iouThreshold = 0.5) {
    vector<pair<double, double>> prValues;
    int cumulativeTP = 0, cumulativeFP = 0;
    int totalGroundTruths = groundTruth.size();

    // Tabulate cumulative TP and FP
    for (auto& pred : predictions) {
        double maxIoU = 0.0;
        BoundingBox* bestMatch = nullptr;
        for (auto& gt : groundTruth) {
            double iou = calculateIoU(pred, gt);
            if (iou >= iouThreshold && iou > maxIoU && !gt.isTruePositive) {
                maxIoU = iou;
                bestMatch = &gt;
            }
        }
        if (bestMatch != nullptr) {
            bestMatch->isTruePositive = true; // Mark ground truth as matched
            pred.isTruePositive = true; // Mark prediction as true positive
            cumulativeTP++;
        } else {
            cumulativeFP++;
        }

        // Calculate row-wise Precision and Recall
        double precision = static_cast<double>(cumulativeTP) / (cumulativeTP + cumulativeFP);
        double recall = static_cast<double>(cumulativeTP) / totalGroundTruths;
        prValues.push_back({precision, recall});
    }
    return prValues;
}

// Function to interpolate precision over recall
double interpolatePrecision(const vector<pair<double, double>>& prValues, double recallPoint) {
    double maxPrecision = 0.0;
    for (const auto& pr : prValues) {
        if (pr.second >= recallPoint) {
            maxPrecision = max(maxPrecision, pr.first);
        }
    }
    return maxPrecision;
}

// Function to calculate Average Precision (AP) using 11-point interpolation
double calculateAveragePrecision(const vector<pair<double, double>>& prValues) {
    double ap = 0.0;
    for (double recallPoint = 0.0; recallPoint <= 1.0; recallPoint += 0.1) {
        double interpolatedPrecision = interpolatePrecision(prValues, recallPoint);
        ap += interpolatedPrecision;
    }
    return ap / 11.0;
}

// Function to calculate mAP
// This function calculates the Mean Average Precision (mAP) for all classes.
// It computes the AP for each class separately and then averages them.
double calculateMAP(const unordered_map<int, vector<BoundingBox>>& groundTruths, const unordered_map<int, vector<BoundingBox>>& predictions) {
    double map = 0.0;
    int numClasses = groundTruths.size();
    double ap;
    for (const auto& gtPair : groundTruths) {
        int classID = gtPair.first;
        const auto& gtBoxes = gtPair.second;

        if (predictions.find(classID) == predictions.end()) {
             ap=0;
        }
        else{
            const auto& predBoxes = predictions.at(classID);

            auto prValues = calculatePrecisionRecall(const_cast<vector<BoundingBox>&>(gtBoxes), const_cast<vector<BoundingBox>&>(predBoxes));
            ap = calculateAveragePrecision(prValues);
            map += ap;
        }
    }

    return map / numClasses;
}

// Function to read segmentation mask
Mat readSegmentationMask(const string& filename) {
    Mat mask = imread(filename, IMREAD_GRAYSCALE);
    if (mask.empty()) {
        cerr << "Could not open or find the image: " << filename << endl;
    }
    return mask;
}

// Function to calculate IoU for segmentation masks
float calculateIoUMask(const Mat& groundTruthMask, const Mat& predictedMask, int classLabel) {
    int intersection = 0, unionArea = 0;
    bool gtPixel;
    bool predPixel;

    for (int i = 0; i < groundTruthMask.rows; i++) {
        for (int j = 0; j < groundTruthMask.cols; j++) {
            gtPixel = (groundTruthMask.at<uchar>(i, j) == classLabel);
            predPixel = (predictedMask.at<uchar>(i, j) == classLabel);

            if (gtPixel && predPixel) {
                intersection++;
            }
            if (gtPixel || predPixel) {
                unionArea++;
            }
        }
    }

    float iou=static_cast<float>(intersection) / (unionArea);
    return (unionArea > 0) ? iou : 0.0;
}

// Function to calculate mIoU for all classes
float calculateMIoU(const vector<string>& groundTruthMaskFiles, const vector<string>& predictedMaskFiles) {
    unordered_map<int, float> totalIoU;
    unordered_map<int, int> countIoU;
    int numImages = groundTruthMaskFiles.size();

    vector<int> classLabels;
    for (int i = 0; i < numImages; i++) {
        Mat groundTruthMask = readSegmentationMask(groundTruthMaskFiles[i]);
        Mat predictedMask = readSegmentationMask(predictedMaskFiles[i]);

        if (groundTruthMask.empty() || predictedMask.empty()) {
            cerr << "Error loading masks for image " << i << endl;
            continue;
        }

        int label;
        for (int y = 0; y < groundTruthMask.rows; y++) {
            for (int x = 0; x < groundTruthMask.cols; x++) {
                uchar gtPixel = groundTruthMask.at<uchar>(y, x);
                label=static_cast<int>(gtPixel);
                //In calculating metrics we realized that in the last frame of game4_clip1 there was a misclassified pixel:
                //it was marked with ID 3 but in the associated ground truth bounding box file, there was no ball with ID 3.
                //So in calculating the mAP we handled this exception.
                if(label ==3 && i==18){
                    label=0;
                }

                if (std::find(classLabels.begin(), classLabels.end(), gtPixel) == classLabels.end()) {
                    classLabels.push_back(label);
                }
            }
        }

        for (int classLabel : classLabels) {
            totalIoU[classLabel] = 0.0;
            countIoU[classLabel] = 0;
        }

        for (int classLabel : classLabels) {
            float iou = calculateIoUMask(groundTruthMask, predictedMask, classLabel);
            totalIoU[classLabel] += iou;
            countIoU[classLabel]++;
        }
    }

    float meanIoU = 0.0;
    int validClasses = 0;

    for (int classLabel : classLabels) {
        if (countIoU[classLabel] > 0) {
            float averageIoU = totalIoU[classLabel] / countIoU[classLabel];
            meanIoU += averageIoU;
            validClasses++;
        }
    }

    return (validClasses > 0) ? meanIoU / validClasses : 0.0;
}

int main(int argc, char** argv) {

    if (argc != 2) {
        return -1;
    }

    // Input images path
    std::string base_path = argv[1];
    // File paths for ground truth and predictions
    vector<string> groundTruthFiles = {
        base_path+"/bounding_boxes/frame_last_bbox.txt"
    };
    vector<string> predictionFiles = {
        base_path+"/output/frame_last_output_bounding_boxes.txt"
    };
    // File paths for segmentation masks
    vector<string> groundTruthMaskFiles = {
        base_path+"/masks/frame_last.png"
    };
    vector<string> predictedMaskFiles = {
        base_path+"/output/frame_last_output_colored_mask.png",
    };

    // Calculate mAP
    unordered_map<int, vector<BoundingBox>> groundTruthsByClass;
    unordered_map<int, vector<BoundingBox>> predictionsByClass;

    //Read bounding boxes from files
    for (const auto& file : groundTruthFiles) {
        vector<BoundingBox> groundTruthBoxes = readBoundingBoxes(file);
        for (const auto& box : groundTruthBoxes) {
            groundTruthsByClass[box.categoryID].push_back(box);
        }
    }

    for (const auto& file : predictionFiles) {
        vector<BoundingBox> predictionBoxes = readBoundingBoxes(file);
        for (const auto& box : predictionBoxes) {
            predictionsByClass[box.categoryID].push_back(box);
        }
    }

    //Calculate MAP
    double mAP = calculateMAP(groundTruthsByClass, predictionsByClass);
    cout << "Mean Average Precision (mAP) at IoU 0.5: " << mAP << endl;

    // Calculate mIoU
    double mIoU = calculateMIoU(groundTruthMaskFiles, predictedMaskFiles);
    cout << "Mean Intersection over Union (mIoU): " << mIoU << endl;

    return 0;
}
