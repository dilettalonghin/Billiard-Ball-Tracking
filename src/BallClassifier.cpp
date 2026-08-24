#include "BallClassifier.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include "FieldDetection.h"

namespace BallClassification{

    // Function to apply Gaussian blur to the image for denoising
    cv::Mat denoise(const cv::Mat& image) {
        cv::Mat denoised;
        cv::GaussianBlur(image, denoised, cv::Size(5, 5), 0);
        return denoised;
    }

    // Function to correct lighting in the image using CLAHE
    cv::Mat correctLighting(const cv::Mat& image) {
        cv::Mat lab_image, corrected_image;
        cv::cvtColor(image, lab_image, cv::COLOR_BGR2Lab);

        std::vector<cv::Mat> lab_planes(3);
        cv::split(lab_image, lab_planes);

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(4);
        clahe->apply(lab_planes[0], lab_planes[0]);

        cv::merge(lab_planes, lab_image);
        cv::cvtColor(lab_image, corrected_image, cv::COLOR_Lab2BGR);

        return corrected_image;
    }

    // Function to calculate the percentage of white pixels in a bounding box
    double calculateWhitePercentage(const cv::Mat& image, const BoundingBox& bbox) {
        cv::Rect rect(bbox.x, bbox.y, bbox.width, bbox.height);
        cv::Mat roi = image(rect);

        cv::Mat gray, mask;
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, mask, 200, 255, cv::THRESH_BINARY);

        int whitePixels = cv::countNonZero(mask);
        int totalPixels = roi.rows * roi.cols;

        return (static_cast<double>(whitePixels) / totalPixels) * 100;
    }

    // Function to calculate the percentage of black pixels in a bounding box
    double calculateBlackPercentage(const cv::Mat& image, const BoundingBox& bbox) {
        cv::Rect rect(bbox.x, bbox.y, bbox.width, bbox.height);
        cv::Mat roi = image(rect);

        cv::Mat gray, mask;
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, mask, 50, 255, cv::THRESH_BINARY_INV);

        int blackPixels = cv::countNonZero(mask);
        int totalPixels = roi.rows * roi.cols;

        return (static_cast<double>(blackPixels) / totalPixels) * 100;
    }

    // Function to classify balls based on white and black pixel percentages
    void classifyBalls(std::vector<BoundingBox>& boundingBoxes, const cv::Mat& image) {
        int cueBallCount = 0;
        int eightBallCount = 0;

        for (auto& bbox : boundingBoxes) {
            double whitePercentage = calculateWhitePercentage(image, bbox);
            double blackPercentage = calculateBlackPercentage(image, bbox);

            // Classify as cue ball (white) if it has more than 20% white pixels and no cue ball is already classified
            if (whitePercentage > 30 && cueBallCount == 0) {
                bbox.category_id = 1;
                cueBallCount++;
            // Classify as eight ball (black) if it has at least 20% black pixels and no eight ball is already classified
            } else if (blackPercentage >= 20 && eightBallCount == 0) {
                bbox.category_id = 2;
                eightBallCount++;
            // Classify as striped ball if it has 2-25% white pixels
            } else if (whitePercentage >= 3  && whitePercentage <= 25) {
                bbox.category_id = 4;
            // Classify as solid color ball for all other cases
            } else {
                bbox.category_id = 3;
            }
        }
    }

    // Function to draw bounding boxes on the image with different colors based on category
    void drawBorderAndBBox(cv::Mat& image, const std::vector<BoundingBox>& boundingBoxes,std::vector<cv::Point> tableCorners) {

        FieldDetection::drawTableBorders(image, tableCorners);

        for (const auto& bbox : boundingBoxes) {
            cv::Scalar color;
            if (bbox.category_id == 1) {
                color = cv::Scalar(255, 255, 255); // White for cue ball
            } else if (bbox.category_id == 2) {
                color = cv::Scalar(0, 0, 0); // Black for eight ball
            } else if (bbox.category_id == 3) {
                color = cv::Scalar(0, 255, 0);  // Green for solid color balls
            } else if (bbox.category_id == 4) {
                color = cv::Scalar(0, 0, 255);  // Red for striped balls
            }

            cv::rectangle(image, cv::Point(bbox.x, bbox.y), cv::Point(bbox.x + bbox.width, bbox.y + bbox.height), color, 2);
        }
    }
    // Function to save bounding box details to a file
    void saveBoundingBoxesToFile(const std::vector<BoundingBox>& boundingBoxes, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return;
        }

        for (const auto& bbox : boundingBoxes) {
            file << bbox.x << " " << bbox.y << " " << bbox.width << " " << bbox.height << " " << bbox.category_id << "\n";
        }

        file.close();
    }
}