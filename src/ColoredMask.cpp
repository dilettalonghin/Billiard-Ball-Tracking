#include "ColoredMask.h"
#include "FieldDetection.h"

namespace ColoredMask{
    // Function to draw a colored mask based on bounding boxes and table mask
    void drawColoredMask(cv::Mat& mask, const std::vector<BoundingBox>& boundingBoxes, const cv::Mat& tableMask) {
        // Set the entire mask to background (0)
        mask.setTo(cv::Scalar(0));

        // Convert maskedFrame to grayscale
         cv::Mat grayMaskedFrame;
        cv::cvtColor(tableMask, grayMaskedFrame, cv::COLOR_BGR2GRAY);

        // Convert grayscale image to binary
        cv::Mat binaryMask;
        cv::threshold(grayMaskedFrame, binaryMask, 1, 255, cv::THRESH_BINARY);

        // Fill the playing field with label 5
        mask.setTo(cv::Scalar(5), binaryMask);


        // Draw circles representing balls with their respective labels
        for (const BoundingBox& bbox : boundingBoxes) {
            int label = bbox.category_id;
            cv::Point center(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
            cv::circle(mask, center, bbox.width / 2, cv::Scalar(label), -1);
        }
    }

    // Function to apply colors to the final image based on the mask labels
    void applyColorsAndContours(cv::Mat& finalImage, const cv::Mat& mask, std::vector<cv::Point> tableCorners) {
        for (int y = 0; y < mask.rows; y++) {
            for (int x = 0; x < mask.cols; x++) {
                int label = mask.at<uchar>(y, x);
                switch (label) {
                    case 0:
                        finalImage.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // Background - Black
                        break;
                    case 1:
                        finalImage.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255); // White cue ball - White
                        break;
                    case 2:
                        finalImage.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // Black 8-ball - Black
                        break;
                    case 3:
                        finalImage.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255); // Solid color balls - Red
                        break;
                    case 4:
                        finalImage.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0); // Striped balls - Blue
                        break;
                    case 5:
                        finalImage.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 128, 0); // Playing field - Green
                        break;
                }
            }
        }

        FieldDetection::drawTableBorders(finalImage, tableCorners);
    }
}