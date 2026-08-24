/**
 * @file BallDetection.h
 *
 * @brief functions for the ball detection
 *
 * @author Longhin Diletta
 */

#ifndef BALL_DETECTION_H
#define BALL_DETECTION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "BoundingBox.h"

namespace BallDetection {

    /**
     * @brief Preprocesses the input image by applying CLAHE and color space conversion.
     * @param image The input image to preprocess.
     * @return The preprocessed image.
     */
    cv::Mat preprocessImage(const cv::Mat& image);

    /**
     * @brief Segments the table area from the input image.
     * @param image The input image.
     * @return The mask of the segmented table.
     */
    cv::Mat segmentTable(const cv::Mat& image);

    /**
     * @brief Removes overlapping Hough circles based on a specified overlap threshold.
     * @param circles The vector of detected circles.
     * @param overlapThreshold The threshold for determining overlap.
    */
    void removeOverlappingCircles(std::vector<cv::Vec3f>& circles, float overlapThreshold);

    /**
     * @brief Detects balls within the input image and keeps the ones inside the playing field
     * @param image The input image.
     * @param tableCorners The corners of the table.
     * @return A vector of bounding boxes for the detected balls.
    */
    std::vector<BoundingBox> detectBalls(const cv::Mat& image, std::vector<cv::Point> tableCorners);

}

#endif