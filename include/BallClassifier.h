/**
 * @file BallCLassifier.h
 *
 * @brief functions for the ball classification
 *
 * @author Nichifor Antonela
 */

#ifndef BALL_CLASSIFIER_H
#define BALL_CLASSIFIER_H

#include "BoundingBox.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace BallClassification{

    /**
     * @brief Denoises the input image.
     * @param image The input image to be denoised.
     * @return The denoised image.
    */
    cv::Mat denoise(const cv::Mat& image);

    /**
     * @brief Corrects the lighting of the input image.
     * @param image The input image for lighting correction.
     * @return The image with corrected lighting.
    */
    cv::Mat correctLighting(const cv::Mat& image);

    /**
     * @brief Calculates the percentage of white pixels within a bounding box.
     * @param image The input image.
     * @param bbox The bounding box within which to calculate the white percentage.
     * @return The percentage of white pixels within the bounding box.
    */
    double calculateWhitePercentage(const cv::Mat& image, const BoundingBox& bbox);

    /**
     * @brief Calculates the percentage of black pixels within a bounding box.
     * @param image The input image.
     * @param bbox The bounding box within which to calculate the black percentage.
     * @return The percentage of black pixels within the bounding box.
    */
    double calculateBlackPercentage(const cv::Mat& image, const BoundingBox& bbox);

    /**
     * @brief Classifies balls based on the bounding boxes and the input image.
     * @param boundingBoxes A vector of bounding boxes to classify.
     * @param image The input image.
    */
    void classifyBalls(std::vector<BoundingBox>& boundingBoxes, const cv::Mat& image);

    /**
     * @brief Draws borders and bounding boxes on the image.
     * @param image The input image on which to draw.
     * @param boundingBoxes A vector of bounding boxes to draw.
     * @param tableCorners A vector of points representing the corners of the table.
    */
    void drawBorderAndBBox(cv::Mat& image, const std::vector<BoundingBox>& boundingBoxes,std::vector<cv::Point> tableCorners);

    /**
     * @brief Saves the bounding boxes to a file.
     * @param boundingBoxes A vector of bounding boxes to save.
     * @param filename The name of the file to save the bounding boxes to.
    */
    void saveBoundingBoxesToFile(const std::vector<BoundingBox>& boundingBoxes, const std::string& filename);

}

#endif
