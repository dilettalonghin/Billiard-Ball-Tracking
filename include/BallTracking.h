/**
 * @file BallTracking.h
 *
 * @brief functions for the ball tracking and minimap implementation
 *
 * @author Longhin Diletta
 */

#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include "BoundingBox.h"
#include <vector>
#include <string>
namespace BallTracking {

    /**
     * @brief Loads bounding box annotations from a file.
     * @param annotationFile The path to the annotation file.
     * @return A pair containing a vector of bounding boxes and their corresponding labels.
     */
    std::vector<BoundingBox> loadBoundingBoxes(const std::string& annotationFile);

    /**
     * @brief Calculates the distance between two points.
     * @param p1 The first point.
     * @param p2 The second point.
     * @return The distance between the two points.
     */
    double calculateDistance(const cv::Point& p1, const cv::Point& p2);

    /**
     * @brief Ensures the correct order of points and identifies the long sides.
     * @param points A vector of points to be ordered.
    */
    void ensureCorrectOrder(std::vector<cv::Point>& points);

    /**
     * @brief Creates a minimap of the table with ball positions and trajectories.
     * @param tableCorners The corners of the table.
     * @param bboxes The bounding boxes of the balls.
     * @param labels The labels of the balls.
     * @param originalImage The original image.
     * @param trajectories The trajectories of the balls.
     * @return The minimap image.
    */
    cv::Mat createMinimap(const std::vector<cv::Point>& tableCorners, const std::vector<cv::Rect>& bboxes, const std::vector<int>& labels, const cv::Mat& originalImage, const std::vector<std::vector<cv::Point>>& trajectories);

    /**
     * @brief Tracks balls and updates the minimap for each frame in a video.
     * @param video The video capture object.
     * @param trackers The vector of trackers.
     * @param bboxes The bounding boxes of the balls.
     * @param labels The labels of the balls.
     * @param tableCorners The corners of the table.
     * @param trajectories The trajectories of the balls.
     * @return A vector of processed frames with the minimap overlay.
    */
    std::vector<cv::Mat> trackAndUpdateMinimap(cv::VideoCapture& video, std::vector<cv::Ptr<cv::Tracker>>& trackers, std::vector<cv::Rect>& bboxes, std::vector<int>& labels, const std::vector<cv::Point>& tableCorners, std::vector<std::vector<cv::Point>>& trajectories);

    /**
     * @brief Creates and initializes trackers for the given bounding boxes.
     * @param frame_first_path The path to the first frame image.
     * @param bboxesPairs A pair containing bounding boxes and their labels.
     * @param tableCornersFirst The corners of the table in the first frame.
     * @param video The video capture object.
     * @return A vector of processed frames with the minimap overlay.
     */
     std::vector<cv::Mat>  createTracker(const std::string& frame_first_path, std::vector<BoundingBox> bboxesPairs , const std::vector<cv::Point>& tableCornersFirst, cv::VideoCapture video) ;
}

#endif
