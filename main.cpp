#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <set>
#include <array>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "image_processing.h"

enum class RunMode
{
    Task1,
    Task2,
    Task3,
    All
};

static RunMode parse_mode_string(const std::string &mode_text)
{
    std::string m = mode_text;
    std::transform(m.begin(), m.end(), m.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    if (m == "task1" || m == "1")
        return RunMode::Task1;
    if (m == "task2" || m == "2")
        return RunMode::Task2;
    if (m == "task3" || m == "3")
        return RunMode::Task3;
    return RunMode::All;
}

static void save_debug_image(const std::string &debug_dir,
                             int frame_idx,
                             const std::string &stage,
                             const cv::Mat &img)
{
    if (debug_dir.empty() || img.empty())
    {
        return;
    }
    cv::utils::fs::createDirectories(debug_dir);
    const std::string path = debug_dir + "/frame_" + std::to_string(frame_idx) + "_" + stage + ".png";
    cv::imwrite(path, img);
}

struct ObjModel
{
    std::vector<cv::Point3f> vertices;
    std::vector<std::pair<int, int>> edges;
    bool loaded = false;
};

struct PoseSmoother
{
    bool initialized = false;
    cv::Mat R;
    cv::Mat t;
    double alpha_rotation = 0.30;
    double alpha_translation = 0.30;

    void update(const cv::Mat &R_new, const cv::Mat &t_new, cv::Mat &R_out, cv::Mat &t_out)
    {
        if (!initialized)
        {
            R = R_new.clone();
            t = t_new.clone();
            initialized = true;
        }
        else
        {
            R = (1.0 - alpha_rotation) * R + alpha_rotation * R_new;
            cv::SVD svd(R);
            R = svd.u * svd.vt;
            if (cv::determinant(R) < 0.0)
            {
                R.col(2) = -R.col(2);
            }

            t = (1.0 - alpha_translation) * t + alpha_translation * t_new;
        }

        R_out = R;
        t_out = t;
    }
};

static cv::Mat default_intrinsics_for_frame(const cv::Size &frame_size)
{
    const double fx = static_cast<double>(frame_size.width);
    const double fy = static_cast<double>(frame_size.width);
    const double cx = static_cast<double>(frame_size.width) * 0.5;
    const double cy = static_cast<double>(frame_size.height) * 0.5;

    return (cv::Mat_<double>(3, 3) <<
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0);
}

static cv::Mat load_intrinsics_matrix(const std::string &path)
{
    std::ifstream probe(path);
    if (!probe.good())
    {
        return cv::Mat();
    }

    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        return cv::Mat();
    }

    cv::Mat K;
    fs["K"] >> K;
    if (K.empty())
    {
        fs["camera_matrix"] >> K;
    }
    if (K.rows != 3 || K.cols != 3)
    {
        return cv::Mat();
    }

    cv::Mat K64;
    K.convertTo(K64, CV_64F);
    return K64;
}

static bool estimate_pose_from_homography(const cv::Mat &H,
                                          const cv::Mat &K,
                                          cv::Mat &R,
                                          cv::Mat &t)
{
    if (H.empty() || K.empty())
    {
        return false;
    }

    cv::Mat H64, K64;
    H.convertTo(H64, CV_64F);
    K.convertTo(K64, CV_64F);

    cv::Mat K_inv = K64.inv();
    cv::Mat h1 = H64.col(0);
    cv::Mat h2 = H64.col(1);
    cv::Mat h3 = H64.col(2);

    cv::Mat r1 = K_inv * h1;
    cv::Mat r2 = K_inv * h2;
    cv::Mat t_raw = K_inv * h3;

    const double n1 = cv::norm(r1);
    const double n2 = cv::norm(r2);
    if (n1 < 1e-9 || n2 < 1e-9)
    {
        return false;
    }

    const double scale = 2.0 / (n1 + n2);
    r1 = scale * r1;
    r2 = scale * r2;
    t = scale * t_raw;

    cv::Mat r3 = r1.cross(r2);
    cv::Mat R_approx(3, 3, CV_64F);
    r1.copyTo(R_approx.col(0));
    r2.copyTo(R_approx.col(1));
    r3.copyTo(R_approx.col(2));

    cv::SVD svd(R_approx);
    R = svd.u * svd.vt;
    if (cv::determinant(R) < 0.0)
    {
        R.col(2) = -R.col(2);
    }

    return true;
}

static bool load_obj_wireframe(const std::string &path, ObjModel &model)
{
    model = ObjModel();
    std::ifstream in(path);
    if (!in.is_open())
    {
        return false;
    }

    std::vector<std::vector<int>> faces;
    std::string line;
    while (std::getline(in, line))
    {
        if (line.size() < 2)
        {
            continue;
        }

        if (line[0] == 'v' && std::isspace(static_cast<unsigned char>(line[1])))
        {
            std::istringstream ss(line.substr(2));
            float x = 0.0f, y = 0.0f, z = 0.0f;
            if (ss >> x >> y >> z)
            {
                model.vertices.push_back(cv::Point3f(x, y, z));
            }
        }
        else if (line[0] == 'f' && std::isspace(static_cast<unsigned char>(line[1])))
        {
            std::istringstream ss(line.substr(2));
            std::vector<int> face;
            std::string token;
            while (ss >> token)
            {
                const size_t slash = token.find('/');
                const std::string index_part = (slash == std::string::npos) ? token : token.substr(0, slash);
                if (index_part.empty())
                {
                    continue;
                }

                int idx = 0;
                try
                {
                    idx = std::stoi(index_part);
                }
                catch (...)
                {
                    continue;
                }
                if (idx > 0)
                {
                    idx -= 1;
                }
                else if (idx < 0)
                {
                    idx = static_cast<int>(model.vertices.size()) + idx;
                }

                if (idx >= 0 && idx < static_cast<int>(model.vertices.size()))
                {
                    face.push_back(idx);
                }
            }
            if (face.size() >= 2)
            {
                faces.push_back(face);
            }
        }
    }

    if (model.vertices.empty() || faces.empty())
    {
        return false;
    }

    std::set<std::pair<int, int>> unique_edges;
    for (const auto &face : faces)
    {
        for (size_t i = 0; i < face.size(); ++i)
        {
            int a = face[i];
            int b = face[(i + 1) % face.size()];
            if (a == b)
            {
                continue;
            }
            if (a > b)
            {
                std::swap(a, b);
            }
            unique_edges.emplace(a, b);
        }
    }

    model.edges.assign(unique_edges.begin(), unique_edges.end());
    model.loaded = !model.edges.empty();
    return model.loaded;
}

static void normalize_obj_to_tag_space(ObjModel &model)
{
    if (!model.loaded || model.vertices.empty())
    {
        return;
    }

    cv::Point3f min_pt = model.vertices[0];
    cv::Point3f max_pt = model.vertices[0];
    cv::Point3f centroid(0.0f, 0.0f, 0.0f);

    for (const auto &v : model.vertices)
    {
        min_pt.x = std::min(min_pt.x, v.x);
        min_pt.y = std::min(min_pt.y, v.y);
        min_pt.z = std::min(min_pt.z, v.z);
        max_pt.x = std::max(max_pt.x, v.x);
        max_pt.y = std::max(max_pt.y, v.y);
        max_pt.z = std::max(max_pt.z, v.z);
        centroid += v;
    }
    centroid *= (1.0f / static_cast<float>(model.vertices.size()));

    const float span_x = max_pt.x - min_pt.x;
    const float span_y = max_pt.y - min_pt.y;
    const float span_z = max_pt.z - min_pt.z;
    const float max_span = std::max(span_x, std::max(span_y, span_z));
    const float safe_span = (max_span > 1e-6f) ? max_span : 1.0f;
    const float scale = 0.8f / safe_span;

    for (auto &v : model.vertices)
    {
        v = (v - centroid) * scale;
    }
}

static bool project_point(const cv::Point3f &world,
                          const cv::Mat &R,
                          const cv::Mat &t,
                          const cv::Mat &K,
                          cv::Point &pixel)
{
    cv::Mat X = (cv::Mat_<double>(3, 1) << static_cast<double>(world.x),
                 static_cast<double>(world.y),
                 static_cast<double>(world.z));

    cv::Mat camera = R * X + t;
    const double z = camera.at<double>(2, 0);
    if (z <= 1e-6)
    {
        return false;
    }

    cv::Mat uvw = K * camera;
    const int u = static_cast<int>(std::lround(uvw.at<double>(0, 0) / uvw.at<double>(2, 0)));
    const int v = static_cast<int>(std::lround(uvw.at<double>(1, 0) / uvw.at<double>(2, 0)));
    pixel = cv::Point(u, v);
    return true;
}

static void draw_cube(cv::Mat &image,
                      const cv::Mat &R,
                      const cv::Mat &t,
                      const cv::Mat &K,
                      float cube_size)
{
    const float cx = 0.5f;
    const float cy = 0.5f;
    const float h = cube_size * 0.5f;
    const std::vector<cv::Point3f> cube_points = {
        cv::Point3f(cx - h, cy - h, 0.0f),
        cv::Point3f(cx + h, cy - h, 0.0f),
        cv::Point3f(cx + h, cy + h, 0.0f),
        cv::Point3f(cx - h, cy + h, 0.0f),
        cv::Point3f(cx - h, cy - h, -cube_size),
        cv::Point3f(cx + h, cy - h, -cube_size),
        cv::Point3f(cx + h, cy + h, -cube_size),
        cv::Point3f(cx - h, cy + h, -cube_size)};

    std::vector<cv::Point> projected(8);
    for (size_t i = 0; i < cube_points.size(); ++i)
    {
        if (!project_point(cube_points[i], R, t, K, projected[i]))
        {
            return;
        }
    }

    const cv::Scalar base_color(0, 255, 255);
    const cv::Scalar pillar_color(0, 0, 255);
    const cv::Scalar top_color(255, 0, 0);

    for (int i = 0; i < 4; ++i)
    {
        cv::line(image, projected[i], projected[(i + 1) % 4], base_color, 2);
        cv::line(image, projected[i + 4], projected[4 + (i + 1) % 4], top_color, 2);
        cv::line(image, projected[i], projected[i + 4], pillar_color, 2);
    }
}

static void draw_obj_wireframe(cv::Mat &image,
                               const ObjModel &model,
                               const cv::Mat &R,
                               const cv::Mat &t,
                               const cv::Mat &K)
{
    if (!model.loaded || model.vertices.empty() || model.edges.empty())
    {
        return;
    }

    std::vector<cv::Point> projected(model.vertices.size());
    std::vector<bool> valid(model.vertices.size(), false);

    for (size_t i = 0; i < model.vertices.size(); ++i)
    {
        cv::Point3f world;
        world.x = 0.5f + model.vertices[i].x;
        world.y = 0.5f + model.vertices[i].y;
        world.z = -model.vertices[i].z;
        valid[i] = project_point(world, R, t, K, projected[i]);
    }

    for (const auto &e : model.edges)
    {
        const int a = e.first;
        const int b = e.second;
        if (a < 0 || b < 0 || a >= static_cast<int>(projected.size()) || b >= static_cast<int>(projected.size()))
        {
            continue;
        }
        if (valid[a] && valid[b])
        {
            cv::line(image, projected[a], projected[b], cv::Scalar(255, 128, 0), 2);
        }
    }
}

static std::string orientation_label(int clockwise_rotations_to_canonical)
{
    const int r = ((clockwise_rotations_to_canonical % 4) + 4) % 4;
    if (r == 0)
        return "UP";
    if (r == 1)
        return "ROT-90";
    if (r == 2)
        return "ROT-180";
    return "ROT-270";
}

static void overlay_template_on_frame(const cv::Mat &template_img,
                                      const std::vector<cv::Point> &quad_corners,
                                      cv::Mat &display)
{
    if (template_img.empty() || quad_corners.size() != 4 || display.empty())
    {
        return;
    }

    std::vector<cv::Point2f> src_template = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(template_img.cols - 1), 0.0f),
        cv::Point2f(static_cast<float>(template_img.cols - 1), static_cast<float>(template_img.rows - 1)),
        cv::Point2f(0.0f, static_cast<float>(template_img.rows - 1))};

    std::vector<cv::Point2f> dst_frame;
    dst_frame.reserve(4);
    for (const auto &p : quad_corners)
    {
        dst_frame.push_back(cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y)));
    }

    cv::Mat H_template_to_frame = custom_compute_homography(src_template, dst_frame);
    if (H_template_to_frame.empty())
    {
        return;
    }

    cv::Mat warped_template;
    custom_warp_perspective(template_img, warped_template, H_template_to_frame, display.size());
    if (warped_template.empty())
    {
        return;
    }

    cv::Mat warped_gray = rgbToGray(warped_template);
    cv::Mat mask = custom_threshold(warped_gray, 10);

    for (int r = 0; r < display.rows; ++r)
    {
        for (int c = 0; c < display.cols; ++c)
        {
            if (mask.at<uchar>(r, c) == 255)
            {
                display.at<cv::Vec3b>(r, c) = warped_template.at<cv::Vec3b>(r, c);
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0]
                  << " <video_path|0> [template_image] [intrinsics_yml] [obj_model] [--mode task1|task2|task3|all] [--debug-dir out_dir]" << std::endl;
        std::cout << "Tip: pass '-' for template_image if you only want intrinsics/obj arguments." << std::endl;
        return -1;
    }

    cv::VideoCapture cap;
    const std::string source_arg = argv[1];
    if (source_arg == "0")
    {
        cap.open(0);
    }
    else
    {
        cap.open(source_arg);
    }

    if (!cap.isOpened())
    {
        std::cout << "Error: Could not open video file: " << argv[1] << std::endl;
        return -1;
    }

    cv::Mat template_img;
    std::string intrinsics_path = "camera_intrinsics.yml";
    std::string obj_path;
    RunMode run_mode = RunMode::All;
    std::string debug_dir;

    std::vector<std::string> positional_args;
    for (int i = 2; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc)
        {
            run_mode = parse_mode_string(argv[i + 1]);
            ++i;
            continue;
        }
        if (arg == "--debug-dir" && i + 1 < argc)
        {
            debug_dir = argv[i + 1];
            ++i;
            continue;
        }
        positional_args.push_back(arg);
    }

    if (positional_args.size() >= 1)
    {
        std::string template_path = positional_args[0];
        if (template_path != "-")
        {
            template_img = cv::imread(template_path, cv::IMREAD_COLOR);
            if (template_img.empty())
            {
                std::cout << "Warning: Could not open template image: " << template_path << std::endl;
            }
        }
    }
    if (positional_args.size() >= 2)
    {
        std::string arg3 = positional_args[1];
        if (arg3.size() >= 4 && arg3.substr(arg3.size() - 4) == ".obj")
        {
            obj_path = arg3;
        }
        else
        {
            intrinsics_path = arg3;
        }
    }
    if (positional_args.size() >= 3)
    {
        obj_path = positional_args[2];
    }

    cv::Mat K = load_intrinsics_matrix(intrinsics_path);
    bool intrinsics_logged = false;
    ObjModel obj_model;
    if (!obj_path.empty())
    {
        if (load_obj_wireframe(obj_path, obj_model))
        {
            normalize_obj_to_tag_space(obj_model);
            std::cout << "Loaded OBJ wireframe: " << obj_path
                      << " (v=" << obj_model.vertices.size()
                      << ", e=" << obj_model.edges.size() << ")" << std::endl;
        }
        else
        {
            std::cout << "Warning: Could not load OBJ file: " << obj_path
                      << ". Falling back to cube rendering." << std::endl;
        }
    }

    const int blur_kernel_size = 5;
    const double blur_sigma = 1.4;
    const uchar threshold_val = 50;
    const int min_quad_area = 500;
    const cv::Size canonical_tag_size(200, 200);
    std::unordered_map<int, PoseSmoother> pose_smoothers_by_id;
    const bool enable_task2_overlay = (run_mode == RunMode::All || run_mode == RunMode::Task2);
    const bool enable_task3_render = (run_mode == RunMode::All || run_mode == RunMode::Task3);

    cv::Mat frame;
    int frame_idx = 0;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            break;
        }
        if (K.empty())
        {
            K = default_intrinsics_for_frame(frame.size());
            if (!intrinsics_logged)
            {
                std::cout << "Using default camera intrinsics. Provide " << intrinsics_path
                          << " for calibrated Task 3 results." << std::endl;
                intrinsics_logged = true;
            }
        }
        else if (!intrinsics_logged)
        {
            std::cout << "Loaded camera intrinsics from: " << intrinsics_path << std::endl;
            intrinsics_logged = true;
        }

        cv::Mat gray = rgbToGray(frame);
        cv::Mat blurred = custom_blur_separable(gray, blur_kernel_size, blur_sigma);
        cv::Mat sobel_edges = sobel_edge_detection(blurred);
        cv::Mat binary_for_contours = custom_threshold(sobel_edges, threshold_val);
        std::vector<std::vector<cv::Point>> detected_contours = detect_contours(binary_for_contours);
        save_debug_image(debug_dir, frame_idx, "gray", gray);
        save_debug_image(debug_dir, frame_idx, "blurred", blurred);
        save_debug_image(debug_dir, frame_idx, "sobel", sobel_edges);
        save_debug_image(debug_dir, frame_idx, "binary", binary_for_contours);

        cv::Mat display = frame.clone();
        cv::Mat contour_debug = frame.clone();
        draw_contours_custom(contour_debug, detected_contours, 1);
        save_debug_image(debug_dir, frame_idx, "contours", contour_debug);
        bool warped_shown = false;
        int quad_counter = 0;
        std::vector<cv::Point2f> accepted_tag_centers;
        const std::array<double, 8> epsilon_factors = {0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060};

        std::sort(detected_contours.begin(), detected_contours.end(),
                  [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b)
                  { return a.size() > b.size(); });

        for (const auto &raw_contour : detected_contours)
        {
            if (raw_contour.size() < 30)
            {
                continue;
            }

            const double perimeter = get_contour_perimeter(raw_contour);
            if (perimeter <= 1.0)
            {
                continue;
            }

            std::vector<cv::Point> approx_curve;
            std::vector<cv::Point> closed_contour = raw_contour;
            closed_contour.push_back(raw_contour.front());

            bool quad_found = false;
            for (const double eps_factor : epsilon_factors)
            {
                std::vector<cv::Point> approx_candidate;
                rdp_simplify(closed_contour, approx_candidate, eps_factor * perimeter);

                if (!approx_candidate.empty() && approx_candidate.front() == approx_candidate.back())
                {
                    approx_candidate.pop_back();
                }

                std::vector<cv::Point> dedup_points;
                dedup_points.reserve(approx_candidate.size());
                for (const auto &p : approx_candidate)
                {
                    if (dedup_points.empty() || p != dedup_points.back())
                    {
                        dedup_points.push_back(p);
                    }
                }

                if (dedup_points.size() != 4)
                {
                    continue;
                }
                if (!is_convex_polygon(dedup_points))
                {
                    continue;
                }
                if (std::abs(contour_area(dedup_points)) < min_quad_area)
                {
                    continue;
                }

                approx_curve = dedup_points;
                quad_found = true;
                break;
            }

            if (!quad_found)
            {
                continue;
            }

            custom_sort_corners(approx_curve);

            double min_edge = 1e12;
            double max_edge = 0.0;
            for (size_t i = 0; i < approx_curve.size(); ++i)
            {
                const cv::Point p1 = approx_curve[i];
                const cv::Point p2 = approx_curve[(i + 1) % approx_curve.size()];
                const double edge_len = cv::norm(p1 - p2);
                min_edge = std::min(min_edge, edge_len);
                max_edge = std::max(max_edge, edge_len);
            }
            if (min_edge < 8.0 || max_edge / min_edge > 2.8)
            {
                continue;
            }
            quad_counter++;

            std::vector<cv::Point2f> src_points;
            src_points.reserve(4);
            for (const auto &p : approx_curve)
            {
                src_points.push_back(cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y)));
            }

            std::vector<cv::Point2f> dst_points = {
                cv::Point2f(0.0f, 0.0f),
                cv::Point2f(static_cast<float>(canonical_tag_size.width - 1), 0.0f),
                cv::Point2f(static_cast<float>(canonical_tag_size.width - 1), static_cast<float>(canonical_tag_size.height - 1)),
                cv::Point2f(0.0f, static_cast<float>(canonical_tag_size.height - 1))};

            cv::Mat H = custom_compute_homography(src_points, dst_points);
            if (H.empty())
            {
                continue;
            }

            cv::Mat warped_tag;
            custom_warp_perspective(frame, warped_tag, H, canonical_tag_size);
            if (warped_tag.empty())
            {
                continue;
            }

            cv::Mat warped_gray = rgbToGray(warped_tag);
            cv::Mat warped_binary = custom_threshold(custom_blur_separable(warped_gray, 5, 1.0), 150);
            save_debug_image(debug_dir, frame_idx, "warped_tag_" + std::to_string(quad_counter), warped_tag);
            save_debug_image(debug_dir, frame_idx, "warped_binary_" + std::to_string(quad_counter), warped_binary);

            TagDecodeResult tag = decode_ar_tag_8x8(warped_binary);
            if (!tag.valid)
            {
                continue;
            }

            cv::Point2f quad_center(0.0f, 0.0f);
            for (const auto &p : approx_curve)
            {
                quad_center += cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y));
            }
            quad_center *= 0.25f;

            bool is_duplicate = false;
            for (const auto &existing_center : accepted_tag_centers)
            {
                if (cv::norm(quad_center - existing_center) < 35.0f)
                {
                    is_duplicate = true;
                    break;
                }
            }
            if (is_duplicate)
            {
                continue;
            }
            accepted_tag_centers.push_back(quad_center);

            if (enable_task2_overlay && tag.valid && !template_img.empty())
            {
                overlay_template_on_frame(template_img, approx_curve, display);
            }
            if (enable_task3_render && tag.valid)
            {
                const std::vector<cv::Point2f> tag_plane = {
                    cv::Point2f(0.0f, 0.0f),
                    cv::Point2f(1.0f, 0.0f),
                    cv::Point2f(1.0f, 1.0f),
                    cv::Point2f(0.0f, 1.0f)};

                cv::Mat H_tag_to_frame = custom_compute_homography(tag_plane, src_points);
                cv::Mat R, t;
                if (!H_tag_to_frame.empty() && estimate_pose_from_homography(H_tag_to_frame, K, R, t))
                {
                    cv::Mat R_smooth, t_smooth;
                    PoseSmoother &smoother = pose_smoothers_by_id[tag.id];
                    smoother.update(R, t, R_smooth, t_smooth);
                    if (obj_model.loaded)
                    {
                        draw_obj_wireframe(display, obj_model, R_smooth, t_smooth, K);
                    }
                    else
                    {
                        draw_cube(display, R_smooth, t_smooth, K, 0.5f);
                    }
                }
            }

            const cv::Scalar quad_color = cv::Scalar(0, 255, 0);
            cv::polylines(display, std::vector<std::vector<cv::Point>>(1, approx_curve), true, quad_color, 2);

            std::string label = "ID: " + std::to_string(tag.id) + " " +
                                orientation_label(tag.clockwise_rotations_to_canonical);
            cv::putText(display, label, approx_curve[0], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);

            for (const auto &p : approx_curve)
            {
                cv::circle(display, p, 4, cv::Scalar(0, 0, 255), -1);
            }

            if (!warped_shown)
            {
                cv::imshow("Warped Binary Tag", warped_binary);
                cv::imshow("Warped Tag", warped_tag);
                warped_shown = true;
            }
        }

        std::string window_title = "AR Tag Detection (Custom Pipeline)";
        if (run_mode == RunMode::Task1)
            window_title += " - Task1";
        else if (run_mode == RunMode::Task2)
            window_title += " - Task2";
        else if (run_mode == RunMode::Task3)
            window_title += " - Task3";
        else
            window_title += " - All";
        cv::imshow(window_title, display);
        save_debug_image(debug_dir, frame_idx, "final", display);

        if (cv::waitKey(1) == 'q')
        {
            break;
        }
        frame_idx++;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
