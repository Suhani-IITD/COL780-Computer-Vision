#include "image_processing.h"
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

// Custom RGB to grayscale conversion
cv::Mat rgbToGray(const cv::Mat &src)
{
    // Ensure the input image is 3-channel (BGR)
    if (src.channels() != 3)
    {
        if (src.channels() == 1)
        {
            return src.clone();
        }
        CV_Error(cv::Error::StsBadArg, "Input image must be 3-channel BGR for rgbToGray.");
    }

    cv::Mat gray(src.rows, src.cols, CV_8UC1); // Output grayscale image (1 channel, 8-bit unsigned)

    // Weights for RGB to grayscale conversion (standard NTSC/PAL luma formula)
    // Note: OpenCV typically stores BGR, so weights are adjusted for B, G, R channels
    const double B_WEIGHT = 0.114; // Matches Python's 0.114 for B channel
    const double G_WEIGHT = 0.587; // Matches Python's 0.587 for G channel
    const double R_WEIGHT = 0.299; // Matches Python's 0.299 for R channel

    for (int i = 0; i < src.rows; ++i)
    {
        const cv::Vec3b *src_row = src.ptr<cv::Vec3b>(i);
        uchar *gray_row = gray.ptr<uchar>(i);
        for (int j = 0; j < src.cols; ++j)
        {
            // Access BGR channels (OpenCV's default order)
            cv::Vec3b pixel = src_row[j];
            // Calculate grayscale value
            gray_row[j] = static_cast<uchar>(
                B_WEIGHT * pixel[0] + G_WEIGHT * pixel[1] + R_WEIGHT * pixel[2]);
        }
    }
    return gray;
}

// Custom Gaussian blur using separable filters
cv::Mat custom_blur_separable(const cv::Mat &src, int kernel_size, double sigma)
{
    if (src.empty() || src.channels() != 1)
    {
        CV_Error(cv::Error::StsBadArg, "Input image must be a non-empty, 1-channel grayscale image.");
    }
    if (kernel_size % 2 == 0 || kernel_size <= 0)
    {
        CV_Error(cv::Error::StsBadArg, "Kernel size must be odd and positive.");
    }

    cv::Mat float_src;
    src.convertTo(float_src, CV_32F); // Convert to float for calculations

    int radius = kernel_size / 2;

    // Generate 1D Gaussian kernel
    std::vector<float> kernel_1d(kernel_size);
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i)
    {
        int x = i - radius;
        kernel_1d[i] = std::exp(-0.5 * (x / sigma) * (x / sigma));
        sum += kernel_1d[i];
    }
    // Normalize kernel
    for (int i = 0; i < kernel_size; ++i)
    {
        kernel_1d[i] /= sum;
    }

    cv::Mat h_result(src.rows, src.cols, CV_32F, cv::Scalar(0)); // Intermediate result after horizontal pass
    cv::Mat v_result(src.rows, src.cols, CV_32F, cv::Scalar(0)); // Final result after vertical pass

    // --- Horizontal Pass ---
    for (int i = 0; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            float pixel_sum = 0.0f;
            for (int k = 0; k < kernel_size; ++k)
            {
                int col = j + k - radius;
                // edge handling
                col = std::max(0, std::min(col, src.cols - 1));
                pixel_sum += float_src.at<float>(i, col) * kernel_1d[k];
            }
            h_result.at<float>(i, j) = pixel_sum;
        }
    }

    // --- Vertical Pass ---
    for (int i = 0; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            float pixel_sum = 0.0f;
            for (int k = 0; k < kernel_size; ++k)
            {
                int row = i + k - radius;
                // edge handling
                row = std::max(0, std::min(row, src.rows - 1));
                pixel_sum += h_result.at<float>(row, j) * kernel_1d[k];
            }
            v_result.at<float>(i, j) = pixel_sum;
        }
    }

    cv::Mat final_result_8U;
    v_result.convertTo(final_result_8U, CV_8UC1); // Convert back to 8-bit unsigned and clip

    return final_result_8U;
}

// Custom binary threshold
cv::Mat custom_threshold(const cv::Mat &src, uchar thresh)
{
    if (src.empty() || src.channels() != 1)
    {
        CV_Error(cv::Error::StsBadArg, "Input image must be a non-empty, 1-channel grayscale image.");
    }

    cv::Mat binary(src.rows, src.cols, CV_8UC1); // Output binary image

    for (int i = 0; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            if (src.at<uchar>(i, j) > thresh)
            {
                binary.at<uchar>(i, j) = 255; // White
            }
            else
            {
                binary.at<uchar>(i, j) = 0; // Black
            }
        }
    }
    return binary;
}

// convolve function for repeated use in any function like blur, edge detection

// sobel operator for edge detection
// takes in an image and returns the image with edges detected

cv::Mat sobel_edge_detection(const cv::Mat &input_img)
{
    // check if input image is empty or not single channel
    if (input_img.empty() || input_img.channels() != 1)
    {
        CV_Error(cv::Error::StsBadArg, "Input image must be a non-empty, 1-channel grayscale image.");
    }

    // sobel kernel x
    cv::Mat kernel_x = (cv::Mat_<float>(3, 3) << -1, 0, 1,
                        -2, 0, 2,
                        -1, 0, 1);

    // sobel kernel y
    cv::Mat kernel_y = (cv::Mat_<float>(3, 3) << -1, -2, -1,
                        0, 0, 0,
                        1, 2, 1);

    // x and y gradient holding matrices
    cv::Mat mag_grad(input_img.rows, input_img.cols, CV_32F, cv::Scalar(0));

    // perform the convolution with kernel x first

    for (int i = 0; i < input_img.rows; ++i)
    {
        for (int j = 0; j < input_img.cols; ++j)
        {
            float sum_x = 0.0f;
            float sum_y = 0.0f;

            // apply the kernels to the neighborhood
            for (int k = -1; k <= 1; ++k)
            {
                int row = std::max(0, std::min(i + k, input_img.rows - 1));
                for (int l = -1; l <= 1; ++l)
                {
                    int col = std::max(0, std::min(j + l, input_img.cols - 1));
                    sum_x += input_img.at<uchar>(row, col) * kernel_x.at<float>(k + 1, l + 1);
                    sum_y += input_img.at<uchar>(row, col) * kernel_y.at<float>(k + 1, l + 1);
                }
            }
            float mag = sqrt(sum_x * sum_x + sum_y * sum_y);
            if (mag > 255)
            {
                mag = 255; // Clip to max value for 8-bit images
            }
            else if (mag < 0)
            {
                mag = 0; // Clip to min value
            }
            mag_grad.at<float>(i, j) = mag;
        }
    }

    // convert the magnitude gradient to 8-bit unsigned and clip
    cv::Mat edge_img;
    mag_grad.convertTo(edge_img, CV_8UC1); // Convert to 8-bit unsigned and clip
    return edge_img;
}

// morphological dilation

// Custom Dilation (Expands white regions to close gaps)
cv::Mat custom_dilate(const cv::Mat &src)
{
    // Output image must be same size/type
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC1);

    // Iterate over every pixel (ignoring border for simplicity)
    for (int r = 1; r < src.rows - 1; ++r)
    {
        const uchar *prev_row = src.ptr<uchar>(r - 1);
        const uchar *curr_row = src.ptr<uchar>(r);
        const uchar *next_row = src.ptr<uchar>(r + 1);
        uchar *dst_row = dst.ptr<uchar>(r);

        for (int c = 1; c < src.cols - 1; ++c)
        {
            // Check 3x3 neighborhood. If ANY pixel is 255, set dst to 255.
            // Optimization: Check center first, then neighbors.
            if (curr_row[c] == 255 ||
                curr_row[c - 1] == 255 || curr_row[c + 1] == 255 ||
                prev_row[c] == 255 || prev_row[c - 1] == 255 || prev_row[c + 1] == 255 ||
                next_row[c] == 255 || next_row[c - 1] == 255 || next_row[c + 1] == 255)
            {

                dst_row[c] = 255;
            }
        }
    }
    return dst;
}

// find the contours using moore neighbour algorithm based on the edge detected image and return the contours

// Helper to check if a pixel is white and within bounds
bool is_pixel_white(const cv::Mat &img, int r, int c)
{
    if (r < 0 || r >= img.rows || c < 0 || c >= img.cols)
        return false;
    return img.at<uchar>(r, c) == 255;
}

std::vector<std::vector<cv::Point>> detect_contours(const cv::Mat& binary_img) {
    if (binary_img.empty() || binary_img.type() != CV_8UC1) {
        throw std::runtime_error("Input must be 8-bit single-channel binary image.");
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::Mat visited = cv::Mat::zeros(binary_img.size(), CV_8UC1);

    // 8-neighbor directions (N, NE, E, SE, S, SW, W, NW)
    int dr[] = {-1, -1, 0, 1, 1, 1, 0, -1};
    int dc[] = {0, 1, 1, 1, 0, -1, -1, -1};

    // Scan image for starting points
    for (int r = 0; r < binary_img.rows; ++r) {
        const uchar* row_ptr = binary_img.ptr<uchar>(r);
        for (int c = 0; c < binary_img.cols; ++c) {
            
            // Found unvisited white pixel
            if (row_ptr[c] == 255 && visited.at<uchar>(r, c) == 0) {
                
                std::vector<cv::Point> contour;
                int start_r = r;
                int start_c = c;
                
                // Add starting point
                contour.push_back(cv::Point(c, r));
                visited.at<uchar>(r, c) = 255;

                // ✅ FIX: Check if isolated pixel
                bool has_neighbors = false;
                for (int k = 0; k < 8; ++k) {
                    if (is_pixel_white(binary_img, r + dr[k], c + dc[k])) {
                        has_neighbors = true;
                        break;
                    }
                }

                if (!has_neighbors) {
                    // Isolated pixel - contour complete
                    if (contour.size() > 50) {
                        contours.push_back(contour);
                    }
                    continue;
                }

                // Moore-Neighbor Tracing
                int curr_r = r;
                int curr_c = c;
                int backtrack_idx = 6;  // Entered from West
                
                int max_iterations = binary_img.rows * binary_img.cols;  // Safety limit
                int iterations = 0;
                
                while (iterations++ < max_iterations) {
                    int found_neighbor = -1;
                    
                    // Search clockwise from (backtrack + 1)
                    for (int k = 0; k < 8; ++k) {
                        int idx = (backtrack_idx + 1 + k) % 8;
                        int nr = curr_r + dr[idx];
                        int nc = curr_c + dc[idx];
                        
                        if (is_pixel_white(binary_img, nr, nc)) {
                            found_neighbor = idx;
                            curr_r = nr;
                            curr_c = nc;
                            break;
                        }
                    }
                    
                    if (found_neighbor == -1) {
                        // Dead end (shouldn't happen for closed contours)
                        break;
                    }

                    // ✅ FIX: Update backtrack FIRST
                    backtrack_idx = (found_neighbor + 4) % 8;

                    // ✅ FIX: Check stop condition BEFORE adding duplicate start
                    if (curr_r == start_r && curr_c == start_c) {
                        // Completed the loop
                        break;
                    }

                    // Add point to contour
                    contour.push_back(cv::Point(curr_c, curr_r));
                    visited.at<uchar>(curr_r, curr_c) = 255;
                }
                
                // Filter and store
                if (contour.size() > 50) {
                    contours.push_back(contour);
                }
            }
        }
    }
    
    return contours;
}

std::vector<std::vector<cv::Point>> detect_contours_opencv_style(const cv::Mat& binary_img) {
    if (binary_img.empty() || binary_img.type() != CV_8UC1) {
        throw std::runtime_error("Input must be 8-bit single-channel binary image.");
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::Mat labeled = cv::Mat::zeros(binary_img.size(), CV_32S);  // -1=traced, 0=bg, >0=contour_id
    
    int contour_id = 1;

    // 8-neighbor offsets
    int dr[] = {-1, -1, 0, 1, 1, 1, 0, -1};
    int dc[] = {0, 1, 1, 1, 0, -1, -1, -1};

    for (int r = 0; r < binary_img.rows; ++r) {
        for (int c = 0; c < binary_img.cols; ++c) {
            
            // Check if this is a new contour starting point
            if (binary_img.at<uchar>(r, c) == 255 && labeled.at<int>(r, c) == 0) {
                
                // Determine if outer or hole boundary
                bool is_outer = (c == 0 || binary_img.at<uchar>(r, c-1) == 0);
                
                std::vector<cv::Point> contour;
                
                // Trace boundary
                int curr_r = r;
                int curr_c = c;
                int dir = is_outer ? 7 : 3;  // Starting search direction
                
                do {
                    contour.push_back(cv::Point(curr_c, curr_r));
                    labeled.at<int>(curr_r, curr_c) = contour_id;
                    
                    // Find next boundary pixel
                    bool found = false;
                    for (int i = 0; i < 8; ++i) {
                        int check_dir = (dir + i) % 8;
                        int nr = curr_r + dr[check_dir];
                        int nc = curr_c + dc[check_dir];
                        
                        if (nr >= 0 && nr < binary_img.rows && 
                            nc >= 0 && nc < binary_img.cols &&
                            binary_img.at<uchar>(nr, nc) == 255) {
                            
                            curr_r = nr;
                            curr_c = nc;
                            dir = (check_dir + 5) % 8;  // Update search start direction
                            found = true;
                            break;
                        }
                    }
                    
                    if (!found) break;
                    
                } while (curr_r != r || curr_c != c);
                
                if (contour.size() > 50) {
                    contours.push_back(contour);
                    contour_id++;
                }
            }
        }
    }
    
    return contours;
}


double get_contour_perimeter(const std::vector<cv::Point> &contour)
{
    double perimeter = 0.0;
    for (size_t i = 0; i < contour.size(); ++i)
    {
        cv::Point p1 = contour[i];
        cv::Point p2 = contour[(i + 1) % contour.size()]; // Wrap around to start
        perimeter += cv::norm(p1 - p2);                   // Euclidean distance
    }
    return perimeter;
}

// Helper: Calculate perpendicular distance of Point P from Line AB
double perpendicular_dist(cv::Point p, cv::Point a, cv::Point b)
{
    double den = std::sqrt(std::pow(b.y - a.y, 2) + std::pow(b.x - a.x, 2));

    // BUG FIX: If line AB is actually a single point (A==B),
    // return the Euclidean distance from P to A.
    if (den == 0)
    {
        return std::sqrt(std::pow(p.y - a.y, 2) + std::pow(p.x - a.x, 2));
    }

    double num = std::abs((b.y - a.y) * p.x - (b.x - a.x) * p.y + b.x * a.y - b.y * a.x);
    return num / den;
}

// Helper function to sort corners: TL, TR, BR, BL
void sort_corners(std::vector<cv::Point> &corners)
{
    // Calculate center

    cv::Point2f center(0, 0);
    for (const auto &p : corners)
        center += cv::Point2f(p);
    center *= (1.0 / corners.size());

    // Separate into Top and Bottom based on Y
    std::vector<cv::Point> top, bot;
    for (const auto &p : corners)
    {
        if (p.y < center.y)
            top.push_back(p);
        else
            bot.push_back(p);
    }

    // Sort Top by X (TL, TR)
    // Note: Use a robust sort for production, this is a simple heuristic
    if (top[0].x > top[1].x)
        std::swap(top[0], top[1]);

    // Sort Bottom by X (BL, BR) -> We want BR, BL ordering for standard CV homography?
    // Actually standard is TL, TR, BR, BL.
    if (bot[0].x > bot[1].x)
        std::swap(bot[0], bot[1]);

    corners.clear();
    corners.push_back(top[0]); // TL
    corners.push_back(top[1]); // TR
    corners.push_back(bot[1]); // BR
    corners.push_back(bot[0]); // BL
}

// Ramer-Douglas-Peucker Algorithm
void rdp_simplify(const std::vector<cv::Point> &points, std::vector<cv::Point> &out_points, double epsilon)
{
    if (points.size() < 3)
    {
        out_points = points;
        return;
    }

    // 1. Find the point with the maximum distance
    double dmax = 0;
    int index = 0;
    int end = points.size() - 1;

    for (int i = 1; i < end; ++i)
    {
        double d = perpendicular_dist(points[i], points[0], points[end]);
        if (d > dmax)
        {
            index = i;
            dmax = d;
        }
    }

    // 2. If max distance > epsilon, recursively simplify both sides
    if (dmax > epsilon)
    {
        std::vector<cv::Point> res1;
        std::vector<cv::Point> res2;

        // Recursive call on first half (0 to index)
        std::vector<cv::Point> first_half(points.begin(), points.begin() + index + 1);
        rdp_simplify(first_half, res1, epsilon);

        // Recursive call on second half (index to end)
        std::vector<cv::Point> second_half(points.begin() + index, points.end());
        rdp_simplify(second_half, res2, epsilon);

        // 3. Merge results
        // res1 ends with 'index', res2 starts with 'index'. Remove duplicate.
        out_points = res1;
        out_points.pop_back();
        out_points.insert(out_points.end(), res2.begin(), res2.end());
    }
    else
    {
        // Base case: All points are close to line. Keep only start and end.
        out_points.clear();
        out_points.push_back(points[0]);
        out_points.push_back(points[end]);
    }
}

void custome_sort_corners(std::vector<cv::Point> &corners)
{
    if (corners.size() != 4)
        return;

    std::vector<cv::Point> sorted(4);
    std::vector<int> sums, diffs;

    for (const auto &p : corners)
    {
        sums.push_back(p.x + p.y);
        diffs.push_back(p.y - p.x);
    }

    // Top-Left: Minimum Sum
    sorted[0] = corners[std::distance(sums.begin(), std::min_element(sums.begin(), sums.end()))];

    // Top-Right: Minimum Difference (y - x) -> effectively (small y, large x)
    sorted[1] = corners[std::distance(diffs.begin(), std::min_element(diffs.begin(), diffs.end()))];

    // Bottom-Right: Maximum Sum
    sorted[2] = corners[std::distance(sums.begin(), std::max_element(sums.begin(), sums.end()))];

    // Bottom-Left: Maximum Difference (y - x) -> effectively (large y, small x)
    sorted[3] = corners[std::distance(diffs.begin(), std::max_element(diffs.begin(), diffs.end()))];

    corners = sorted;
}

void draw_contours_custom(cv::Mat &image,
                          const std::vector<std::vector<cv::Point>> &contours,
                          int thickness)
{

    // Validate input
    if (image.empty())
    {
        std::cerr << "Error: Input image is empty." << std::endl;
        return;
    }

    // cv::RNG rng(12345);
    cv::Scalar color(0, 255, 0); // Green color for contours

    // Iterate through each contour (each shape)
    for (size_t i = 0; i < contours.size(); ++i)
    {
        // cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        const std::vector<cv::Point> &contour = contours[i];

        // Skip empty contours
        if (contour.empty())
            continue;

        // Iterate through points in the current contour
        for (size_t j = 0; j < contour.size(); ++j)
        {
            cv::Point p1 = contour[j];

            // Connect to the next point (Wrap around to 0 for the last point)
            cv::Point p2 = contour[(j + 1) % contour.size()];

            if (thickness == 1)
            {
                // SIMPLE METHOD: Pixel access (Faster, good for 1px thickness)
                // Note: This only sets the exact points. If points are far apart, use line().
                if (p1.x >= 0 && p1.x < image.cols && p1.y >= 0 && p1.y < image.rows)
                {
                    if (image.channels() == 3)
                    {
                        image.at<cv::Vec3b>(p1) = cv::Vec3b(color[0], color[1], color[2]);
                    }
                    else
                    {
                        image.at<uchar>(p1) = (uchar)color[0];
                    }
                }
            }
            else
            {
                // ROBUST METHOD: Draw Lines (Handles gaps and thickness)
                // We use cv::line because implementing Bresenham's algorithm
                // from scratch for thickness > 1 is complex and unnecessary
                // unless strictly forbidden.
                cv::line(image, p1, p2, color, thickness, cv::LINE_8);
            }
        }
    }
}

cv::Mat custom_compute_homography(const std::vector<cv::Point2f> &src_points, const std::vector<cv::Point2f> &dst_points)
{
    // We need at least 4 points
    if (src_points.size() < 4 || dst_points.size() < 4)
    {
        return cv::Mat();
    }

    // 1. Construct Matrix A (8x9)
    cv::Mat A = cv::Mat::zeros(8, 9, CV_64F);

    for (int i = 0; i < 4; ++i)
    {
        double x = src_points[i].x;
        double y = src_points[i].y;
        double u = dst_points[i].x;
        double v = dst_points[i].y;

        // Equation 1 (Row 2*i)
        A.at<double>(2 * i, 0) = -x;
        A.at<double>(2 * i, 1) = -y;
        A.at<double>(2 * i, 2) = -1;
        A.at<double>(2 * i, 3) = 0;
        A.at<double>(2 * i, 4) = 0;
        A.at<double>(2 * i, 5) = 0;
        A.at<double>(2 * i, 6) = u * x;
        A.at<double>(2 * i, 7) = u * y;
        A.at<double>(2 * i, 8) = u;

        // Equation 2 (Row 2*i + 1)
        A.at<double>(2 * i + 1, 0) = 0;
        A.at<double>(2 * i + 1, 1) = 0;
        A.at<double>(2 * i + 1, 2) = 0;
        A.at<double>(2 * i + 1, 3) = -x;
        A.at<double>(2 * i + 1, 4) = -y;
        A.at<double>(2 * i + 1, 5) = -1;
        A.at<double>(2 * i + 1, 6) = v * x;
        A.at<double>(2 * i + 1, 7) = v * y;
        A.at<double>(2 * i + 1, 8) = v;
    }

    // 2. Solve using SVD (Singular Value Decomposition)
    cv::Mat w, u, vt;
    // A = U * W * Vt
    cv::SVD::compute(A, w, u, vt, cv::SVD::FULL_UV);

    // The solution 'h' is the last row of Vt (corresponding to the smallest singular value)
    std::cout << "Vt Matrix shape: " << vt.rows << "x" << vt.cols << std::endl; // Debug: Print Vt shape
    cv::Mat H = vt.row(vt.rows - 1).reshape(0, 3);
    std::cout << "Unnormalized Homography Matrix: " << std::endl
              << H << std::endl; // Debug: Print unnormalized H

    // 3. Normalize: Make H[2][2] = 1 (if possible)
    if (H.at<double>(2, 2) != 0)
    {
        H = H / H.at<double>(2, 2);
    }

    return H;
}

void custom_warp_perspective(const cv::Mat &src, cv::Mat &dst, const cv::Mat &H, cv::Size size)
{
    // 1. Initialize destination image
    dst = cv::Mat::zeros(size, src.type());

    // 2. Compute Inverse Homography (We map Dst -> Src)
    cv::Mat H_inv = H.inv();

    // 3. Iterate over every pixel in the DESTINATION image
    for (int v = 0; v < size.height; v++)
    {
        for (int u = 0; u < size.width; u++)
        {

            // Apply Homography: P_src = H_inv * P_dst
            // P_dst is (u, v, 1)
            double den = H_inv.at<double>(2, 0) * u +
                         H_inv.at<double>(2, 1) * v +
                         H_inv.at<double>(2, 2);

            // Avoid division by zero
            if (den == 0)
                continue;

            double x_src = (H_inv.at<double>(0, 0) * u +
                            H_inv.at<double>(0, 1) * v +
                            H_inv.at<double>(0, 2)) /
                           den;

            double y_src = (H_inv.at<double>(1, 0) * u +
                            H_inv.at<double>(1, 1) * v +
                            H_inv.at<double>(1, 2)) /
                           den;

            // 4. Nearest Neighbor Interpolation (Round to nearest integer)
            int x_int = static_cast<int>(std::round(x_src));
            int y_int = static_cast<int>(std::round(y_src));

            // 5. Boundary Check: Is the source pixel inside the image?
            if (x_int >= 0 && x_int < src.cols && y_int >= 0 && y_int < src.rows)
            {
                if (src.channels() == 3)
                {
                    dst.at<cv::Vec3b>(v, u) = src.at<cv::Vec3b>(y_int, x_int);
                }
                else
                {
                    dst.at<uchar>(v, u) = src.at<uchar>(y_int, x_int);
                }
            }
        }
    }
}