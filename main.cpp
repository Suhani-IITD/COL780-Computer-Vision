#include <iostream>
#include <opencv2/opencv.hpp>
#include "image_processing.h" // Our custom functions (currently unused, but kept for future)
#include "test_functions.h"   // OpenCV's built-in functions for comparison and pipeline demo

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <video_path>" << std::endl;
        return -1;
    }

    std::cout << "Attempting to open video: " << argv[1] << std::endl;
    cv::VideoCapture cap(argv[1]);

    if (!cap.isOpened())
    {
        std::cout << "Error: Could not open video file: " << argv[1] << std::endl;
        std::flush(std::cout); // Explicitly flush output
        return -1;
    }

    std::cout << "Video opened successfully." << std::endl;
    std::flush(std::cout); // Explicitly flush output

    
    cv::Mat frame;
    while (true)
    {
        cap >> frame; // Read a new frame from the video

        if (frame.empty())
        {
            std::cout << "Frame empty, breaking loop." << std::endl;
            std::flush(std::cout); // Explicitly flush output
            break;
        }

        std::cout << "Processing frame..." << std::endl;
        std::flush(std::cout); // Explicitly flush output

        // --- Parameters ---
        int blur_kernel_size = 5;
        double blur_sigma = 1.4;
        uchar threshold_val = 50; // For simple thresholding

        // --- CUSTOM PIPELINE DEMONSTRATION ---

        cv::Mat custome_gray = rgbToGray(frame);
        cv::Mat custom_blurred = custom_blur_separable(custome_gray, blur_kernel_size, blur_sigma);
        cv::Mat custom_sobel_edges = sobel_edge_detection(custom_blurred);
        cv::Mat custom_binary_for_contours = custom_threshold(custom_sobel_edges, threshold_val);
        std::vector<std::vector<cv::Point>> custom_detected_contours = detect_contours_opencv_style(custom_binary_for_contours);

        // polygon approximation using rdp

        cv::Mat custom_contours_display_frame = frame.clone();

        std::vector<Contour> simplified_contours;
        for (const auto &current_raw_contour : custom_detected_contours)
        {

            // A. Calculate Epsilon (Threshold)
            // "I want the simplified shape to be within 2% error of the original."
            double perimeter = get_contour_perimeter(current_raw_contour);
            double epsilon = 0.02 * perimeter;

            // B. Simplify (Reduce 1000 points -> 4 points)
            std::vector<cv::Point> approx_curve;
            rdp_simplify(current_raw_contour, approx_curve, epsilon);
            // cv::approxPolyDP(current_raw_contour, approx_curve, epsilon, true);
            // C. Store it
            // We only care if it simplified to a Triangle(3), Quad(4), or Hexagon(6) etc.
            // For AR tags, we specifically look for 4 points.
            simplified_contours.push_back(approx_curve);

            std::cout << "Contour with " << current_raw_contour.size() << " points simplified to " << approx_curve.size() << " points." << std::endl;
            std::flush(std::cout); // Explicitly flush output
                                   // Visual Debugging
            for (const auto &p : approx_curve)
            {
                cv::circle(custom_contours_display_frame, p, 5, cv::Scalar(0, 0, 255), -1); // Red dots on corners
            }
            // Show the text of how many points it found
            cv::putText(custom_contours_display_frame, std::to_string(approx_curve.size()), approx_curve[0],
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
            // Optional: Filter immediately
            if (approx_curve.size() >= 4 && approx_curve.size() <= 6 &&
                cv::isContourConvex(approx_curve) &&
                cv::contourArea(approx_curve) > 500)
            { // Lower area threshold

                // FORCE TO 4 CORNERS: Use Convex Hull or Bounding Box logic
                // Quick Fix: If size > 4, re-run RDP with higher epsilon
                if (approx_curve.size() > 4)
                {
                    rdp_simplify(current_raw_contour, approx_curve, epsilon * 1.5); // Try harder
                    // cv::approxPolyDP(current_raw_contour, approx_curve, epsilon * 1.5, true);
                }
                std::cout << "Found a potential tag with 4 corners lesgoo!" << std::endl;
                std::flush(std::cout); // Explicitly flush output

                // 2. ORDER CORNERS (TL, TR, BR, BL)
                sort_corners(approx_curve);

                // 3. PREPARE INPUTS
                // Convert to float for Homography math
                std::vector<cv::Point2f> src_points;
                for (const auto &p : approx_curve)
                    src_points.push_back(cv::Point2f(p));

                // Define Destination (Flat 200x200 Square)
                std::vector<cv::Point2f> dst_points = {
                    cv::Point2f(0, 0), cv::Point2f(200, 0), cv::Point2f(200, 200), cv::Point2f(0, 200)};

                std::cout << "Source Points:" << std::endl;
                for (const auto &p : src_points)
                {
                    std::cout << p << std::endl;
                }

                // draw dots on the src points and cv::imshow
                // cv::Mat src_points_display = frame.clone();
                // for (const auto &p : src_points){
                //     cv::circle(src_points_display, p, 5, cv::Scalar(255, 0, 255), -1); // Magenta dots
                // }
                // cv::imshow("Source Points", src_points_display);

                cv::Mat H = custom_compute_homography(src_points, dst_points);

                // 5. WARP (Get the flat tag image)
                if (!H.empty())
                {
                    cv::Mat warped_tag;
                    // OpenCV:
                    custom_warp_perspective(frame, warped_tag, H, cv::Size(200, 200));
                    cv::Mat binary_warped_tag = custom_threshold(custom_blur_separable(rgbToGray(warped_tag), 5, 1.0), 150); // Optional: Threshold to make it more binary and clear
                    cv::imshow("Binary Warped Tag", binary_warped_tag);
                    // warped_tag = custom_threshold(warped_tag, 150); // Optional: Threshold to make it more binary and clear

                    // Custom:
                    // warped_tag = custom_warpPerspective(frame, H, cv::Size(200, 200));

                    // Show the result!
                    cv::imshow("Detected Tag", warped_tag);
                }
            }
        }

        draw_contours_custom(custom_contours_display_frame, simplified_contours, 2);
        // draw_contours_custom(custom_contours_display_frame, custom_detected_contours, 2);
        // cv::drawContours(custom_contours_display_frame, simplified_contours, -1, cv::Scalar(255, 0, 0), 2); // Draw in blue for visibility
        cv::imshow("Custom Native Contours", custom_contours_display_frame);

        // --- OPENCV NATIVE PIPELINE DEMONSTRATION ---

        // 1. Grayscale & Blur
        cv::Mat cv_gray = cv_rgbToGray(frame);
        cv::Mat cv_blurred = cv_custom_blur_separable(cv_gray, blur_kernel_size, blur_sigma);

        // 2. Edge Detection (Sobel)
        cv::Mat cv_sobel_edges = cv_sobel_edge_detection(cv_blurred);
        // cv::imshow("OpenCV Sobel Edges", cv_sobel_edges);

        // 3. Thresholding for Contours (to get binary image)
        cv::Mat cv_binary_for_contours = cv_custom_threshold(cv_sobel_edges, threshold_val);
        // cv::imshow("OpenCV Binary for Contours", cv_binary_for_contours);

        // 4. Contour Detection
        std::vector<std::vector<cv::Point>> cv_own_detected_contours = cv_find_contours(cv_binary_for_contours);

        std::vector<std::vector<cv::Point>> cv_detected_contours = cv_find_contours(cv_binary_for_contours);
        cv::Mat cv_contours_display_frame = frame.clone();

        std::vector<Contour> cv_simplified_contours;
        for (const auto &current_raw_contour : cv_detected_contours)
        {

            // A. Calculate Epsilon (Threshold)
            // "I want the simplified shape to be within 2% error of the original."
            double perimeter = get_contour_perimeter(current_raw_contour);
            double epsilon = 0.02 * perimeter;

            // B. Simplify (Reduce 1000 points -> 4 points)
            std::vector<cv::Point> approx_curve;
            rdp_simplify(current_raw_contour, approx_curve, epsilon);
            // cv::approxPolyDP(current_raw_contour, approx_curve, epsilon, true);
            // C. Store it
            // We only care if it simplified to a Triangle(3), Quad(4), or Hexagon(6) etc.
            // For AR tags, we specifically look for 4 points.
            cv_simplified_contours.push_back(approx_curve);

            std::cout << "CV Contour with " << current_raw_contour.size() << " points simplified to " << approx_curve.size() << " points." << std::endl;
            std::flush(std::cout); // Explicitly flush output
                                   // Visual Debugging
            for (const auto &p : approx_curve)
            {
                cv::circle(cv_contours_display_frame, p, 5, cv::Scalar(0, 0, 255), -1); // Red dots on corners
            }
            // Show the text of how many points it found
            cv::putText(cv_contours_display_frame, std::to_string(approx_curve.size()), approx_curve[0],
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
            // Optional: Filter immediately
            if (approx_curve.size() >= 4 && approx_curve.size() <= 6 &&
                cv::isContourConvex(approx_curve) &&
                cv::contourArea(approx_curve) > 500)
            { // Lower area threshold

                // FORCE TO 4 CORNERS: Use Convex Hull or Bounding Box logic
                // Quick Fix: If size > 4, re-run RDP with higher epsilon
                if (approx_curve.size() > 4)
                {
                    rdp_simplify(current_raw_contour, approx_curve, epsilon * 1.5); // Try harder
                    // cv::approxPolyDP(current_raw_contour, approx_curve, epsilon * 1.5, true);
                }
                std::cout << "Found a potential tag with 4 corners!" << std::endl;
                std::flush(std::cout); // Explicitly flush output

                // 2. ORDER CORNERS (TL, TR, BR, BL)
                sort_corners(approx_curve);

                // 3. PREPARE INPUTS
                // Convert to float for Homography math
                std::vector<cv::Point2f> src_points;
                for (const auto &p : approx_curve)
                    src_points.push_back(cv::Point2f(p));

                // Define Destination (Flat 200x200 Square)
                std::vector<cv::Point2f> dst_points = {
                    cv::Point2f(0, 0), cv::Point2f(200, 0), cv::Point2f(200, 200), cv::Point2f(0, 200)};

                // 4. COMPUTE HOMOGRAPHY
                // Using OpenCV:
                cv::Mat H = custom_compute_homography(src_points, dst_points);

                // Using Custom (if you implement the SVD solver described previously):
                // cv::Mat H = custom_computeHomography(src_points, dst_points);

                // 5. WARP (Get the flat tag image)
                if (!H.empty())
                {
                    cv::Mat cv_warped_tag;
                    // OpenCV:
                    custom_warp_perspective(frame, cv_warped_tag, H, cv::Size(200, 200));

                    // Custom:
                    // warped_tag = custom_warpPerspective(frame, H, cv::Size(200, 200));

                    // Show the result!
                    cv::imshow("CV Detected Tag", cv_warped_tag);
                }
            }
        }

        cv::Mat cv_drawn_contours_frame = frame.clone();

        draw_contours_custom(cv_contours_display_frame, cv_detected_contours, 2);

        cv::drawContours(cv_drawn_contours_frame, cv_own_detected_contours, -1, cv::Scalar(0, 255, 0), 2); // Draw contours in green
        // cv::imshow("OpenCV Native Contours", cv_contours_display_frame);
        // cv::imshow("OpenCV Own Contours", cv_drawn_contours_frame);

        // 6. Homography Demonstration
        // cv::Mat homography_display_frame = frame.clone();
        // // For demonstration, let's assume a square region on the screen as our 'tag'
        // std::vector<cv::Point2f> src_points_on_frame = {
        //     cv::Point2f(200, 100), // Top-left
        //     cv::Point2f(400, 100), // Top-right
        //     cv::Point2f(400, 300), // Bottom-right
        //     cv::Point2f(200, 300)  // Bottom-left
        // };
        // // Draw source points
        // for(size_t i = 0; i < src_points_on_frame.size(); ++i) {
        //     cv::circle(homography_display_frame, src_points_on_frame[i], 5, cv::Scalar(0, 0, 255), -1); // Red
        // }
        // cv::polylines(homography_display_frame, std::vector<std::vector<cv::Point>>(1, std::vector<cv::Point>(src_points_on_frame.begin(), src_points_on_frame.end())), true, cv::Scalar(0, 0, 255), 2);

        // // Target points (a standard square, e.g., for mapping a tag)
        // std::vector<cv::Point2f> dst_points_for_homography = {
        //     cv::Point2f(0, 0),
        //     cv::Point2f(sample_image.cols, 0),
        //     cv::Point2f(sample_image.cols, sample_image.rows),
        //     cv::Point2f(0, sample_image.rows)
        // };

        // cv::Mat H = cv_computeHomography(src_points_on_frame, dst_points_for_homography);
        // if (!H.empty()) {
        //     cv::Mat warped_sample_image;
        //     // Warp the sample image onto the homography_display_frame
        //     cv::warpPerspective(sample_image, warped_sample_image, H, homography_display_frame.size());

        //     // Blend the warped image with the original frame
        //     // For blending, convert original frame to float, add, convert back
        //     cv::Mat gray_warped;
        //     cv::cvtColor(warped_sample_image, gray_warped, cv::COLOR_BGR2GRAY);
        //     cv::Mat mask = (gray_warped > 0); // Create mask of non-black pixels from warped image

        //     homography_display_frame.setTo(cv::Scalar(0,0,0), mask); // Black out the area where warped image will be
        //     cv::add(homography_display_frame, warped_sample_image, homography_display_frame);
        // }
        // cv::imshow("Homography Demo (Warped Object)", homography_display_frame);

        if (cv::waitKey(1) == 'q')
        { // Wait for 1ms and check for 'q' key press
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    std::cout << "Program finished." << std::endl;
    std::flush(std::cout); // Explicitly flush output

    return 0;
}
