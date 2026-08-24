#include "BallTracking.h"
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <vector>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include "FieldDetection.h"
#include "BoundingBox.h"

namespace BallTracking {

    // Function to load bounding box annotations
   std::vector<BoundingBox> loadBoundingBoxes(const std::string& annotationFile) {
        std::ifstream file(annotationFile);

        if (!file.is_open()) {
            std::cerr << "Unable to open file: " << annotationFile << std::endl;
            return {};
        }

        std::vector<BoundingBox> bboxes;

        int x, y, w, h, ID;
        while (file >> x >> y >> w >> h >> ID) {
            BoundingBox bbox(x, y, w, h, ID);
            bboxes.push_back(bbox);
        }
        return bboxes;
    }

    // Function to calculate the distance between two points
    double calculateDistance(const cv::Point& p1, const cv::Point& p2) {
        return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
    }

    // Function to ensure correct order of points and identify the long sides
    void ensureCorrectOrder(std::vector<cv::Point>& points) {
        if (points.size() != 4) {
            throw std::runtime_error("Expected exactly 4 corner points");
        }

        // Step 1: Sort points by x-coordinate (ascending)
        std::sort(points.begin(), points.end(), [](const cv::Point& a, const cv::Point& b) {
            return a.x < b.x;
        });

        // Step 2: Extract the two rightmost points (with largest x-values)
        std::vector<cv::Point> rightmostPoints = {points[2], points[3]};
        points.erase(points.begin() + 2, points.end());

        // Step 3: Identify top-right and bottom-right from the two rightmost points
        cv::Point topRight = (rightmostPoints[0].y < rightmostPoints[1].y) ? rightmostPoints[0] : rightmostPoints[1];
        cv::Point bottomRight = (rightmostPoints[0].y > rightmostPoints[1].y) ? rightmostPoints[0] : rightmostPoints[1];

        // Step 4: Identify bottom-left and top-left from the remaining points
        cv::Point bottomLeft = (points[0].y > points[1].y) ? points[0] : points[1];
        cv::Point topLeft = (points[0].y < points[1].y) ? points[0] : points[1];

        // Step 5: Reorder points as bottom-right, top-right, top-left, bottom-left
        points = {bottomRight, topRight, topLeft, bottomLeft};
    }

    // Function to create a minimap
    cv::Mat createMinimap(const std::vector<cv::Point>& tableCorners, const std::vector<cv::Rect>& bboxes, const std::vector<int>& labels, const cv::Mat& originalImage, const std::vector<std::vector<cv::Point>>& trajectories) {
        if (tableCorners.size() != 4) {
            throw std::runtime_error("Expected exactly 4 table corner points");
        }

        // Apply ensureCorrectOrder on the transformed points
        std::vector<cv::Point> sortedCorners = tableCorners;
        ensureCorrectOrder(sortedCorners);

        // Calculate distances between the corners
        double d0 = calculateDistance(sortedCorners[0], sortedCorners[1]); // Distance between bottom-right and top-right
        double d1 = calculateDistance(sortedCorners[1], sortedCorners[2]); // Distance between top-right and top-left

        int minimapWidth = 500;
        int minimapHeight = 250;
        cv::Mat minimap = cv::Mat::zeros(minimapHeight, minimapWidth, CV_8UC3);
        minimap.setTo(cv::Scalar(255, 255, 255)); // White background for the table

        // Define the destination points for the minimap pockets based on distances
        double threshold = 150;
        std::vector<cv::Point2f> dstPoints;
        if (d0 > d1 - threshold) {
            dstPoints = {
                cv::Point2f(minimapWidth - 1, 0),
                cv::Point2f(0, 0),
                cv::Point2f(0, minimapHeight - 1),
                cv::Point2f(minimapWidth - 1, minimapHeight - 1)
            };
        } else {
            dstPoints = {
                cv::Point2f(minimapWidth - 1, minimapHeight - 1),
                cv::Point2f(minimapWidth - 1, 0),
                cv::Point2f(0, 0),
                cv::Point2f(0, minimapHeight - 1)
            };
        }

        // Compute the perspective transformation matrix again for the minimap
        std::vector<cv::Point2f> orderedCorners = {
            cv::Point2f(static_cast<float>(sortedCorners[0].x), static_cast<float>(sortedCorners[0].y)),
            cv::Point2f(static_cast<float>(sortedCorners[1].x), static_cast<float>(sortedCorners[1].y)),
            cv::Point2f(static_cast<float>(sortedCorners[2].x), static_cast<float>(sortedCorners[2].y)),
            cv::Point2f(static_cast<float>(sortedCorners[3].x), static_cast<float>(sortedCorners[3].y))
        };
        cv::Mat transformMatrix = cv::getPerspectiveTransform(orderedCorners, dstPoints);

        // Transform the corners
        std::vector<cv::Point2f> transformedCorners;
        cv::perspectiveTransform(orderedCorners, transformedCorners, transformMatrix);

        // Draw the pockets on the minimap
        for (const auto& pt : dstPoints) {
            cv::circle(minimap, pt, 10, cv::Scalar(0, 0, 0), -1);
        }

        // Add middle pockets on the long sides of the table
        cv::Point2f middleTop = (dstPoints[0] + dstPoints[1]) * 0.5;
        cv::Point2f middleBottom = (dstPoints[2] + dstPoints[3]) * 0.5;
        cv::circle(minimap, middleTop, 7, cv::Scalar(0, 0, 0), -1);
        cv::circle(minimap, middleBottom, 7, cv::Scalar(0, 0, 0), -1);

        // Draw the balls on the minimap
        for (size_t i = 0; i < bboxes.size(); ++i) {
            cv::Point2f center(bboxes[i].x + bboxes[i].width / 2.0f, bboxes[i].y + bboxes[i].height / 2.0f);
            std::vector<cv::Point2f> srcPoint = {center};
            std::vector<cv::Point2f> dstPoint;
            cv::perspectiveTransform(srcPoint, dstPoint, transformMatrix);

            // Ensure the points are within the table bounds
            if (dstPoint[0].x < 0 || dstPoint[0].x >= minimapWidth || dstPoint[0].y < 0 || dstPoint[0].y >= minimapHeight) {
                continue; // Skip points that are out of bounds
            }

            int label = labels[i];
            cv::Scalar color;
            switch (label) {
                case 1: color = cv::Scalar(255, 255, 255); break; // White cue ball
                case 2: color = cv::Scalar(0, 0, 0); break; // Black 8-ball
                case 3: color = cv::Scalar(0, 0, 255); break; // Solid color
                case 4: color = cv::Scalar(255, 0, 0); break; // Striped
                default: color = cv::Scalar(255, 255, 255); break;
            }
            cv::circle(minimap, dstPoint[0], 5, color, -1);
        }

        // Draw the trajectories on the minimap
        for (const auto& trajectory : trajectories) {
            for (size_t i = 1; i < trajectory.size(); i++) {
                cv::Point2f srcPrevPoint = cv::Point2f(trajectory[i - 1]);
                cv::Point2f srcCurrPoint = cv::Point2f(trajectory[i]);
                std::vector<cv::Point2f> srcPoints = {srcPrevPoint, srcCurrPoint};
                std::vector<cv::Point2f> dstPoints;
                cv::perspectiveTransform(srcPoints, dstPoints, transformMatrix);

                if (i % 2 == 0) {
                    cv::line(minimap, dstPoints[0], dstPoints[1], cv::Scalar(0, 0, 0), 1);
                }
            }
        }

        return minimap;
    }

    std::vector<cv::Mat> trackAndUpdateMinimap(cv::VideoCapture& video, std::vector<cv::Ptr<cv::Tracker>>& trackers, std::vector<cv::Rect>& bboxes, std::vector<int>& labels, const std::vector<cv::Point>& tableCorners, std::vector<std::vector<cv::Point>>& trajectories) {
        cv::Mat frame;
        std::vector<cv::Mat> processedFrames;

        while (video.read(frame)) {
            for (size_t i = 0; i < trackers.size(); i++) {
                cv::Rect bbox;
                if (trackers[i]->update(frame, bbox)) {
                    cv::Point center(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
                    trajectories[i].push_back(center);
                } else {
                    // Reinitialize the tracker if it fails
                    trackers[i] = cv::TrackerCSRT::create();
                    trackers[i]->init(frame, bboxes[i]);
                }
            }

            // Update the minimap with the new trajectories
            cv::Mat updatedMinimap = createMinimap(tableCorners, bboxes, labels, frame, trajectories);

            // Resize the minimap to fit the bottom-left corner of the frame
            cv::Mat resizedMinimap;
            cv::resize(updatedMinimap, resizedMinimap, cv::Size(frame.cols / 4, frame.rows / 4));

            // Copy the resized minimap to the bottom-left corner of the frame
            resizedMinimap.copyTo(frame(cv::Rect(0, frame.rows - resizedMinimap.rows, resizedMinimap.cols, resizedMinimap.rows)));

            // Store the processed frame
            processedFrames.push_back(frame.clone());

            cv::imshow("Tracking", frame);

            if (cv::waitKey(30)==27) {
                break;
            }
        }

        return processedFrames;
    }

    std::vector<cv::Mat>  createTracker(const std::string& frame_first_path, std::vector<BoundingBox> bboxesPairs , const std::vector<cv::Point>& tableCornersFirst, cv::VideoCapture video) {

        // Initialize the trackers
        std::vector<cv::Rect> bboxes;
        std::vector<int> labels;

        // Convert BoundingBox to cv::Rect and labels
        for (const auto& bboxPair : bboxesPairs) {
            bboxes.push_back(cv::Rect(bboxPair.x, bboxPair.y, bboxPair.width, bboxPair.height));
            labels.push_back(bboxPair.category_id);
        }

        cv::Mat image = cv::imread(frame_first_path);
        std::vector<cv::Ptr<cv::Tracker>> trackers;
        for (const auto& bbox : bboxes) {
            cv::Ptr<cv::Tracker> tracker = cv::TrackerCSRT::create();
            tracker->init(image, bbox);
            trackers.push_back(tracker);
        }

        // Variables to track the trajectories
        std::vector<std::vector<cv::Point>> trajectories(bboxes.size());

        // Track and update the minimap
        std::vector<cv::Mat> processedFrames = trackAndUpdateMinimap(video, trackers, bboxes, labels, tableCornersFirst, trajectories);
        video.release();
        cv::destroyAllWindows();

        return processedFrames;
    }
}