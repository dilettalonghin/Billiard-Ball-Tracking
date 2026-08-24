#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <vector>

#include "FieldDetection.h"
#include "BallTracking.h"
#include "BallDetection.h"
#include "BallClassifier.h"
#include "BoundingBox.h"
#include "ColoredMask.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    if (argc != 2) {
        return -1;
    }

    // Input images path
    std::string base_path = argv[1];

    size_t pos = base_path.find_last_of("/\\");
    std::string folder_name = base_path.substr(pos + 1);

    std::string frame_first_path ="frame_first";
    std::string frame_last_path = "frame_last";
    std::vector<std::string> images= {frame_first_path,frame_last_path};

    std::vector<cv::Point> tableCorners;

    //Loop on input images
    for(int i=0;i<2;i++){

        // Load the image
        cv::Mat image = cv::imread(base_path + "/frames/"+images[i]+".png");
        if (image.empty()) {
            return -1;
        }

        /////FIELD SEGMENTATION

        //Compute table corners sing FieldDetection's tableBorders
        tableCorners = FieldDetection::tableBorders(image);
        if (tableCorners.empty()) {
            return -1;
        }
        //Compute table mask
        Mat tableMask= FieldDetection::createTableMask(image,tableCorners);

        //////BALL DETECTION

        // Detect balls using BallDetection's detectBalls
        vector<BoundingBox> boundingBoxes = BallDetection::detectBalls(image,tableCorners);

        ////BALL CLASSIFICATION

        // Classify the detected balls within the table area
        BallClassification::classifyBalls(boundingBoxes, tableMask);

        // Draw bounding boxes and categories on the image
        BallClassification::drawBorderAndBBox(image, boundingBoxes,tableCorners);

        //STORAGE

        // Save bounding boxes to file for tracking
        std::string output_bbox_file = base_path + "/output/"+images[i]+"_output_bounding_boxes.txt";
        BallClassification::saveBoundingBoxesToFile(boundingBoxes, output_bbox_file);

        // Save the image with bounding boxes and table borders
        std::string output_bboxes_image_path = base_path + "/output/"+images[i]+"_output_bboxes_image.png";
        cv::imwrite(output_bboxes_image_path, image);

        ////BALLS AND PLAYING FIELD SEGMENTATION MASK

        // Create and display the colored mask
        cv::Mat coloredMask(image.size(), CV_8UC1);
        ColoredMask::drawColoredMask(coloredMask, boundingBoxes, tableMask);

        // Combine original image with colored mask
        cv::Mat finalImage(image.size(), CV_8UC3);
        ColoredMask::applyColorsAndContours(finalImage, coloredMask,tableCorners);

        //// STORAGE AND OUTPUT DISPLAY

        // Save the final image and colored mask
        std::string output_final_image_path = base_path + "/output/"+images[i]+"_output_final_image.png";
        cv::imwrite(output_final_image_path, finalImage);

        std::string output_colored_mask_path = base_path + "/output/"+images[i]+"_output_colored_mask.png";
        cv::imwrite(output_colored_mask_path, coloredMask);

        // Display the final image
        cv::imshow("Final Image with Colored Mask - "+images[i], finalImage);
        cv::waitKey(0);

        // Display the original result
        cv::imshow("Detected Balls - "+images[i], image);
        cv::waitKey(0);

    }

    // Input video path
    std::string video_path = base_path +"/"+ folder_name + ".mp4";
    // Load the video
    cv::VideoCapture video(video_path);
    if (!video.isOpened()) {
        return -1;
    }

    //VIDEO TRACKING

    //Read bounding boxes from the first frame file
    vector<BoundingBox> bboxes = BallTracking::loadBoundingBoxes(base_path + "/output/frame_first_output_bounding_boxes.txt");
    std::vector<cv::Mat> outputFrames = BallTracking::createTracker(base_path + "/frames/frame_first.png",bboxes,tableCorners, video);

    ////STORAGE
    std::string output_frames_tracking_path;
    for (int count = 0, i = 0; i < outputFrames.size() && count != 4; i += 40, ++count) {
       output_frames_tracking_path =base_path + "/output/frame"+std::to_string(i)+"_output_tracking.png";
       cv::imwrite(output_frames_tracking_path, outputFrames[i]);

    }
    return 0;
}
