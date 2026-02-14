#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>

struct TagInfo {
    int id;                          // Tag ID (0-15)
    int orientation;                 // Degrees: 0, 90, 180, 270
    std::vector<cv::Point2f> corners; // 4 corners (TL, TR, BR, BL)
    cv::Point2f center;              // Tag center point
};


typedef std::vector<cv::Point> Contour;
// Custom image processing functions implemented from scratch
cv::Mat rgbToGray(const cv::Mat &src);
cv::Mat custom_blur_separable(const cv::Mat &src, int kernel_size, double sigma);
cv::Mat custom_threshold(const cv::Mat &src, uchar thresh);
cv::Mat sobel_edge_detection(const cv::Mat &input_img);
cv::Mat custom_dilate(const cv::Mat &src);
bool is_pixel_white(const cv::Mat &img, int r, int c);
std::vector<Contour> detect_contours(const cv::Mat &edge_img);
std::vector<std::vector<cv::Point>> detect_contours_opencv_style(const cv::Mat& binary_img);
double get_contour_perimeter(const std::vector<cv::Point> &contour);
double perpendicular_dist(cv::Point p, cv::Point a, cv::Point b);
void sort_corners(std::vector<cv::Point> &corners);
void sort_corners_float(std::vector<cv::Point2f> &corners);
void custome_sort_corners(std::vector<cv::Point> &corners);
void rdp_simplify(const std::vector<cv::Point> &points, std::vector<cv::Point> &out_points, double epsilon);
void draw_contours_custom(cv::Mat &image,
                          const std::vector<std::vector<cv::Point>> &contours,
                          int thickness);
cv::Mat custom_compute_homography(const std::vector<cv::Point2f> &src_points, const std::vector<cv::Point2f> &dst_points);
void custom_warp_perspective(const cv::Mat &src, cv::Mat &dst, const cv::Mat &H, cv::Size size);
TagInfo decode_ar_tag(const cv::Mat& frame, const std::vector<cv::Point>& corners);


#endif // IMAGE_PROCESSING_H
