
#include "BallDetection.h"
#include "BoundingBox.h"

using namespace cv;
using namespace std;

namespace BallDetection {

    Mat preprocessImage(const Mat& image) {
        //Convert imahe to LAB color space
        Mat labImage, labChannels[3];
        cvtColor(image, labImage, COLOR_BGR2Lab);
        split(labImage, labChannels);

        //Apply CLAHE to L channel

        Ptr<CLAHE> clahe = createCLAHE();
        clahe->setClipLimit(2.0);
        clahe->apply(labChannels[0], labChannels[0]);
        merge(labChannels, 3, labImage);

        //convert to BGR
        Mat result;
        cvtColor(labImage, result, COLOR_Lab2BGR);

        return result;
    }

    Mat segmentTable(const Mat& image) {

        //Do pre-processing to improve image contrast
        Mat preprocessed = preprocessImage(image);

        //Convert image to HSV
        Mat hsvImage;
        cvtColor(preprocessed, hsvImage, COLOR_BGR2HSV);

        //Calculate avarage color in the ROI localized in the center of the image
        int centerX = image.cols / 2;
        int centerY = image.rows / 2;
        int kernelSize = 40;

        int roiX = max(0, centerX - kernelSize / 2);
        int roiY = max(0, centerY - kernelSize / 2);
        int roiWidth = min(kernelSize, image.cols - roiX);
        int roiHeight = min(kernelSize, image.rows - roiY);

        if (roiWidth <= 0 || roiHeight <= 0) {
            cerr << "Invalid ROI dimensions." << endl;
            return Mat();
        }

        Mat roi = hsvImage(Rect(roiX, roiY, roiWidth, roiHeight));
        Scalar avgColor = mean(roi);

        //Threshold image based on average color and thresholding deltas parameters
        int hDelta = 8, sDelta = 82, vDelta = 154;
        Scalar lowerBound(avgColor[0] - hDelta, avgColor[1] - sDelta, avgColor[2] - vDelta);
        Scalar upperBound(avgColor[0] + hDelta, avgColor[1] + sDelta, avgColor[2] + vDelta);

        Mat mask;
        inRange(hsvImage, lowerBound, upperBound, mask);

        // Morphological operations to clean up the mask
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(mask, mask, MORPH_GRADIENT, kernel); // Fill small holes
        Mat kernel1 = getStructuringElement(MORPH_RECT, Size(1, 1));
        dilate(mask, mask, kernel1, Point(-1, -1), 1);

        return mask;
    }

    void removeOverlappingCircles(vector<Vec3f>& circles, float overlapThreshold) {

        vector<bool> keep(circles.size(), true);
        //Compare current circle with subsequent circle
        for (size_t i = 0; i < circles.size(); ++i) {
            if (!keep[i]) continue;

            for (size_t j = i + 1; j < circles.size(); ++j) {
                if (!keep[j]) continue;
                //Calculate the horizontal and vertical distance between the 2 circles
                float dx = circles[i][0] - circles[j][0];
                float dy = circles[i][1] - circles[j][1];
                float distance = sqrt(dx * dx + dy * dy);
                float radiusSum = circles[i][2] + circles[j][2];
                //Check if the distance is less than the threshold times the sum of radii.
                if (distance < overlapThreshold * radiusSum) {
                    //Mark the smaller cirlces for removal
                    if (circles[i][2] < circles[j][2]) {
                        keep[i] = false;
                    } else {
                        keep[j] = false;
                    }
                }
            }
        }

        vector<Vec3f> filteredCircles;
        for (size_t i = 0; i < circles.size(); ++i) {
            if (keep[i]) {
                filteredCircles.push_back(circles[i]);
            }
        }

        circles = filteredCircles;
    }

    std::vector<BoundingBox> detectBalls(const Mat& image, std::vector<cv::Point> tableCorners) {

        Mat gray;
        cvtColor(image, gray, COLOR_BGR2GRAY);

        //Apply segment table function
        Mat mask = segmentTable(image);

        //Apply Hough circles
        vector<Vec3f> circles;
        HoughCircles(mask, circles, HOUGH_GRADIENT, 1, 10, 37, 13, 4, 16);

        // Remove overlapping circles
        removeOverlappingCircles(circles, 0.9);

        int fixedSize = 20; // Set the size of the bounding box

        std::vector<BoundingBox> boundingBoxes;
        for (size_t i = 0; i < circles.size(); ++i) {
            cv::Vec3i c = circles[i];
            cv::Point center = cv::Point(c[0], c[1]);

            // Calculate the top-left corner of the bounding box
            cv::Point topLeft = cv::Point(center.x - fixedSize / 2, center.y - fixedSize / 2);

            // Create the bounding box with fixed size
            BoundingBox bbox(topLeft.x, topLeft.y, fixedSize, fixedSize, 0);
            boundingBoxes.push_back(bbox);
        }

        // Filter out balls that are not within the table area
        std::vector<BoundingBox> ballsWithinTable;
        std::vector<cv::Point> hull;
        cv::convexHull(tableCorners, hull);
        for (const auto& ball : boundingBoxes) {
            cv::Rect ballRect(ball.x, ball.y, ball.width, ball.height);
            if (cv::pointPolygonTest(hull, cv::Point2f(ballRect.x + ballRect.width / 2, ballRect.y + ballRect.height / 2), false) >= 0) {
                ballsWithinTable.push_back(ball);
            }
        }

        return ballsWithinTable;
    }

}