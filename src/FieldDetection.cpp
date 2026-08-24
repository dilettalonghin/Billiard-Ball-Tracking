
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

namespace FieldDetection{

    // Function to draw the table borders on the image
    void drawTableBorders(cv::Mat& image, const std::vector<cv::Point>& points) {

        if (points.size() == 4) {
            line(image, points[0], points[1], cv::Scalar(0, 255, 0), 2);
            line(image, points[1], points[3], cv::Scalar(0, 255, 0), 2);
            line(image, points[3], points[2], cv::Scalar(0, 255, 0), 2);
            line(image, points[2], points[0], cv::Scalar(0, 255, 0), 2);
        }
    }

    // Function to get the intersection point of two lines
    cv::Point getIntersectionPoint(cv::Vec2f line1, cv::Vec2f line2) {

        float rho1 = line1[0], theta1 = line1[1];
        float rho2 = line2[0], theta2 = line2[1];

        float cosTheta1 = cos(theta1), sinTheta1 = sin(theta1);
        float cosTheta2 = cos(theta2), sinTheta2 = sin(theta2);

        float det = cosTheta1 * sinTheta2 - sinTheta1 * cosTheta2;
        if (fabs(det) < 1e-10) {
            return cv::Point(-1, -1);
        }

        int x = cvRound((sinTheta2 * rho1 - sinTheta1 * rho2) / det);
        int y = cvRound((cosTheta1 * rho2 - cosTheta2 * rho1) / det);

        return cv::Point(x, y);
    }

    // Function to filter and draw the main lines on the image
    std::vector<cv::Point> findTableBorders(const std::vector<cv::Vec2f>& lines, const cv::Mat& image) {

        //Subdivide lines into horizontal and vertical
        std::vector<cv::Vec2f> horizontalLines, verticalLines;
        for (size_t i = 0; i < lines.size(); i++) {
            float theta = lines[i][1];
            if (theta <= CV_PI / 3 || theta >= 2 * CV_PI / 3) {
                verticalLines.push_back(lines[i]);
            } else {
                horizontalLines.push_back(lines[i]);
            }
        }

        // Check if there are enough lines to draw the borders
        if (horizontalLines.size() < 2 || verticalLines.size() < 2) {
            cout << "Not enough lines to draw borders" << endl;
            return {}; // Exit if there are not enough lines
        }

        // Sort vertical lines by the projection of fabs(rho) on the x-axis (considering the sign of rho)
        std::sort(verticalLines.begin(), verticalLines.end(), [](const cv::Vec2f& a, const cv::Vec2f& b) {
            return fabs(a[0]) * fabs(cos(a[1])) < fabs(b[0]) * fabs(cos(b[1]));
        });

        // Sort horizontal lines by the projection of fabs(rho) on the y-axis (sin(theta) * fabs(rho))
        std::sort(horizontalLines.begin(), horizontalLines.end(), [](const cv::Vec2f& a, const cv::Vec2f& b) {
            return fabs(a[0]) * sin(a[1]) < fabs(b[0]) * sin(b[1]);
        });

        //Compute image center point and filter table border lines
        int imgCenterX = image.cols / 2;
        int imgCenterY = image.rows / 2;

        cv::Vec2f leftLine, rightLine, topLine, bottomLine;
        bool foundLeftLine = false, foundRightLine = false, foundTopLine = false, foundBottomLine = false;
        float maxLeftProjectionX = -1, minRightProjectionX = image.cols;
        float maxTopProjectionY = -1, minBottomProjectionY = image.rows;

        // Find the vertical lines closest to the center of the image
        for (const auto& line : verticalLines) {
            float projectionX;
            if (line[0] < 0) {
                projectionX = fabs(line[0]) / cos(CV_PI - line[1]);
            } else {
                projectionX = fabs(line[0]) * cos(line[1]);
            }

            if (projectionX <= imgCenterX && projectionX > maxLeftProjectionX) {
                leftLine = line;
                foundLeftLine = true;
                maxLeftProjectionX = projectionX;
            }
            if (projectionX >= imgCenterX && projectionX < minRightProjectionX) {
                rightLine = line;
                foundRightLine = true;
                minRightProjectionX = projectionX;
            }

        }

        // Find the horizontal lines closest to the center of the image
        for (const auto& line : horizontalLines) {
            float projectionY = fabs(line[0]) * sin(line[1]);
            if (projectionY <= imgCenterY && projectionY > maxTopProjectionY) {
                topLine = line;
                foundTopLine = true;
                maxTopProjectionY = projectionY;
            } else if (projectionY >= imgCenterY && projectionY < minBottomProjectionY) {
                bottomLine = line;
                foundBottomLine = true;
                minBottomProjectionY = projectionY;
            }
        }

        // Calculate intersection points of these lines
        cv::Point pt;
        std::vector<cv::Point> points;
        pt = getIntersectionPoint(topLine, leftLine);
        if (pt!= cv::Point(-1, -1)) points.push_back(pt);
        pt = getIntersectionPoint(topLine, rightLine);
        if (pt != cv::Point(-1, -1)) points.push_back(pt);
        pt = getIntersectionPoint(bottomLine, leftLine);
        if (pt != cv::Point(-1, -1)) points.push_back(pt);
        pt = getIntersectionPoint(bottomLine, rightLine);
        if (pt!= cv::Point(-1, -1)) points.push_back(pt);

        return points;
    }

    // Function to compute the table corners and detect its borders
    std::vector<cv::Point> tableBorders(const cv::Mat& image) {

        cv::Mat hsv, gray, bilateral, gaussian, canny;

        // Convert to HSV color space
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

        // Convert to grayscale
        cv::cvtColor(hsv, gray, cv::COLOR_BGR2GRAY);

        // Apply bilateral filter
        cv::bilateralFilter(gray, bilateral, 9, 25, 75);

        // Apply Gaussian blur
        cv::GaussianBlur(bilateral, gaussian, cv::Size(9, 9), 2);

        // Apply Canny edge detection
        Canny(gaussian, canny, 60, 120);

        // Detect lines using Hough transform
        std::vector<cv::Vec2f> lines;
        HoughLines(canny, lines, 1, CV_PI / 180, 110);

        // Find the table borders
        std::vector<cv::Point> tableCorners;
        tableCorners = findTableBorders(lines, image);

        // Check if we have exactly 4 corners
        if (tableCorners.size() != 4) {
            cout << "Detected corners: " << tableCorners.size() << endl;
            for (const auto& corner : tableCorners) {
                cout << "Corner: " << corner << endl;
            }
        }

        return tableCorners;
    }

    //Function to create table Mask
   cv::Mat createTableMask(const cv::Mat& image, const std::vector<cv::Point>& points) {

    // Create a mask from table corners
    cv::Mat tableMask = cv::Mat::zeros(image.size(), CV_8UC1);
    std::vector<cv::Point> hull;
    cv::convexHull(points, hull);
    std::vector<std::vector<cv::Point>> contours = {hull};
    cv::fillPoly(tableMask, contours, cv::Scalar(255));

    // Apply the mask to the image
    cv::Mat coloredMask;
    cv::cvtColor(tableMask, coloredMask, cv::COLOR_GRAY2BGR);

    cv::Mat maskedFrame;
    cv::bitwise_and(image, coloredMask, maskedFrame);
    return maskedFrame;
}
}