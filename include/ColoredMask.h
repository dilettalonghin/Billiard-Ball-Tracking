/**
 * @file ColoredMask.h
 *
 * @brief functions for colored mask implementation
 *
 * @author Nichifor Antonela
 */


#ifndef COLOREDMASK_H
#define COLOREDMASK_H

#include <opencv2/opencv.hpp>
#include "BoundingBox.h"
#include <vector>

namespace ColoredMask{


    /**
     * @brief Draws colored masks for the bounding boxes on the given mask.
     * @param mask The mask image on which to draw the colored masks.
     * @param boundingBoxes A vector of bounding boxes for which colored masks will be drawn.
     * @param tableMask The table mask used for reference.
    */
    void drawColoredMask(cv::Mat& mask, const std::vector<BoundingBox>& boundingBoxes, const cv::Mat& tableMask);

    /**
     * @brief Applies colors and contours to the final image based on the given mask.
     * @param finalImage The final image to which colors and contours will be applied.
     * @param mask The mask used for applying colors and drawing contours.
     * @param tableCorners A vector of points representing the corners of the table.
    */
    void applyColorsAndContours(cv::Mat& finalImage, const cv::Mat& mask, std::vector<cv::Point> tableCorners);

}

#endif
