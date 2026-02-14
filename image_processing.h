#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>

typedef std::vector<cv::Point> Contour;

struct TagDecodeResult
{
    bool valid = false;
    int id = -1;
    int clockwise_rotations_to_canonical = 0;
};

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
void custom_sort_corners(std::vector<cv::Point> &corners);
void rdp_simplify(const std::vector<cv::Point> &points, std::vector<cv::Point> &out_points, double epsilon);
void draw_contours_custom(cv::Mat &image,
                          const std::vector<std::vector<cv::Point>> &contours,
                          int thickness);
double contour_area(const std::vector<cv::Point> &polygon);
bool is_convex_polygon(const std::vector<cv::Point> &polygon);
cv::Mat custom_compute_homography(const std::vector<cv::Point2f> &src_points, const std::vector<cv::Point2f> &dst_points);
void custom_warp_perspective(const cv::Mat &src, cv::Mat &dst, const cv::Mat &H, cv::Size size);
cv::Mat rotate_binary_90_cw(const cv::Mat &src);
TagDecodeResult decode_ar_tag_8x8(const cv::Mat &warped_binary);

#endif // IMAGE_PROCESSING_H
