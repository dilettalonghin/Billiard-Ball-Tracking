/**
 * @file FieldDetection.h
 *
 * @brief functions for the field detection and segmentation
 *
 * @author Poci Ortisa
*/

#ifndef FIELD_DETECTION_H
#define FIELD_DETECTION_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace FieldDetection {
    /**
     * @brief Draws the table borders on the image based on the provided points..
     * @param image The image on which the table borders will be drawn.
     * @param points A vector of points representing the corners of the table.
    */
    void drawTableBorders(cv::Mat& image, const std::vector<cv::Point>& points);

    /**
     * @brief Calculates the intersection point of two lines given their parameters.
     * @param line1 The parameters of the first line.
     * @param line2 The parameters of the second line.
     * @return The intersection point of the two lines.
    */
    cv::Point getIntersectionPoint(cv::Vec2f line1, cv::Vec2f line2);

    /**
     * @brief Filters and draws the main lines on the image and returns the points representing the table borders.
     * @param lines A vector of detected lines.
     * @param image The image on which the lines will be drawn.
     * @return A vector of points representing the table borders.
    */
    std::vector<cv::Point> findTableBorders(const std::vector<cv::Vec2f>& lines, cv::Mat& image);

    /**
     * @brief Processes the input image to segment the table and detect its borders.
     * @param image The input image.
     * @return A vector of points representing the corners of the table.
    */
    std::vector<cv::Point> tableBorders(const cv::Mat& image);

    /**
     * @brief Creates a mask for the table using the provided points representing the table corners.
     * @param image The input image.
     * @param points A vector of points representing the corners of the table.
     * @return The masked image.
    */
    cv::Mat createTableMask(const cv::Mat& image, const std::vector<cv::Point>& points);

}

#endif // FIELD_DETECTION_H
