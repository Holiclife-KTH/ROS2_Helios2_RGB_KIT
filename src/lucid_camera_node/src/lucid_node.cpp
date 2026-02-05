#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>

#include "ArenaApi.h"
#include "SaveApi.h"

#define TAB1 "  "
#define TAB2 "    "

#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "cv_bridge/cv_bridge.h"

using namespace std::chrono_literals;

#define TIMEOUT 200
#define FILE_NAME_IN "/home/irol/workspace/ros2_helios2_rgb_kit/src/lucid_camera_node/resource/orientation.yml"

bool isApplicableDeviceTriton(Arena::DeviceInfo deviceInfo)
{
    return ((deviceInfo.ModelName().find("TRI") != GenICam::gcstring::npos) && (deviceInfo.ModelName().find("-C") != GenICam::gcstring::npos));
}

bool isApplicableDeviceHelios2(Arena::DeviceInfo deviceInfo)
{
    return ((deviceInfo.ModelName().find("HLT") != GenICam::gcstring::npos) || (deviceInfo.ModelName().find("HT") != GenICam::gcstring::npos));
}

// 포인트클라우드 처리용 데이터 구조체
struct PointCloudData
{
    cv::Mat imageMatrixXYZ;
    cv::Mat imageMatrixRGB;
    rclcpp::Time timestamp;
};

class MinimalPublisher : public rclcpp::Node
{
public:
    MinimalPublisher()
    : Node("lucid_camera_node"), pSystem_(nullptr), pDeviceTRI_(nullptr), pDeviceHLT_(nullptr),
      running_(true)
    {
        image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("triton/image_raw", 10);
        pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("helios/pointcloud_rgb", 10);
        
        try
        {
            loadCalibrationData();
            
            pSystem_ = Arena::OpenSystem();
            pSystem_->UpdateDevices(100);
            std::vector<Arena::DeviceInfo> deviceInfos = pSystem_->GetDevices();
            if (deviceInfos.size() == 0)
            {
                RCLCPP_ERROR(this->get_logger(), "No camera connected");
                Arena::CloseSystem(pSystem_);
                pSystem_ = nullptr;
                throw std::runtime_error("No camera connected");
            }

            for (auto& deviceInfo : deviceInfos)
            {
                if (!pDeviceTRI_ && isApplicableDeviceTriton(deviceInfo))
                {
                    pDeviceTRI_ = pSystem_->CreateDevice(deviceInfo);
                    RCLCPP_INFO(this->get_logger(), "Found Triton camera: %s", deviceInfo.ModelName().c_str());

                    Arena::SetNodeValue<bool>(pDeviceTRI_->GetTLStreamNodeMap(), "StreamAutoNegotiatePacketSize", true);
                    Arena::SetNodeValue<bool>(pDeviceTRI_->GetTLStreamNodeMap(), "StreamPacketResendEnable", true);
                    
                    #if defined(__linux__)
                        Arena::SetNodeValue<GenICam::gcstring>(pDeviceTRI_->GetNodeMap(), "PixelFormat", "BGR8");
                    #else
                        Arena::SetNodeValue<GenICam::gcstring>(pDeviceTRI_->GetNodeMap(), "PixelFormat", "RGB8");
                    #endif
                }
                else if (!pDeviceHLT_ && isApplicableDeviceHelios2(deviceInfo))
                {
                    pDeviceHLT_ = pSystem_->CreateDevice(deviceInfo);
                    RCLCPP_INFO(this->get_logger(), "Found Helios2 camera: %s", deviceInfo.ModelName().c_str());

                    Arena::SetNodeValue<bool>(pDeviceHLT_->GetTLStreamNodeMap(), "StreamAutoNegotiatePacketSize", true);
                    Arena::SetNodeValue<bool>(pDeviceHLT_->GetTLStreamNodeMap(), "StreamPacketResendEnable", true);
                    Arena::SetNodeValue<GenICam::gcstring>(pDeviceHLT_->GetNodeMap(), "PixelFormat", "Coord3D_ABCY16");
                }
            }

            if (!pDeviceTRI_)
            {
                RCLCPP_ERROR(this->get_logger(), "No applicable Triton devices found");
                cleanup();
                throw std::logic_error("No applicable Triton devices");
            }

            if (!pDeviceHLT_)
            {
                RCLCPP_ERROR(this->get_logger(), "No applicable Helios2 devices found");
                cleanup();
                throw std::logic_error("No applicable Helios2 devices");
            }

            GenApi::INodeMap* node_map = pDeviceHLT_->GetNodeMap();
            xyz_scale_mm_ = Arena::GetNodeValue<double>(node_map, "Scan3dCoordinateScale");
            Arena::SetNodeValue<GenICam::gcstring>(node_map, "Scan3dCoordinateSelector", "CoordinateA");
            x_offset_mm_ = Arena::GetNodeValue<double>(node_map, "Scan3dCoordinateOffset");
            Arena::SetNodeValue<GenICam::gcstring>(node_map, "Scan3dCoordinateSelector", "CoordinateB");
            y_offset_mm_ = Arena::GetNodeValue<double>(node_map, "Scan3dCoordinateOffset");
            Arena::SetNodeValue<GenICam::gcstring>(node_map, "Scan3dCoordinateSelector", "CoordinateC");
            z_offset_mm_ = Arena::GetNodeValue<double>(node_map, "Scan3dCoordinateOffset");

            pDeviceTRI_->StartStream();
            pDeviceHLT_->StartStream();
            RCLCPP_INFO(this->get_logger(), "Started streaming from both cameras");
            
            // 포인트클라우드 처리 스레드 시작
            pointcloud_thread_ = std::thread(&MinimalPublisher::pointcloud_processing_thread, this);
            
            // 이미지 캡처 타이머 (30Hz - 이미지만)
            timer_ = this->create_wall_timer(
                33ms, std::bind(&MinimalPublisher::timer_callback, this));
        }
        catch (GenICam::GenericException& ge)
        {
            RCLCPP_ERROR(this->get_logger(), "GenICam exception: %s", ge.what());
            cleanup();
            throw;
        }
        catch (std::exception& ex)
        {
            RCLCPP_ERROR(this->get_logger(), "Standard exception: %s", ex.what());
            cleanup();
            throw;
        }
    }
    
    ~MinimalPublisher()
    {
        running_ = false;
        cv_.notify_all();
        
        if (pointcloud_thread_.joinable())
        {
            pointcloud_thread_.join();
        }
        
        cleanup();
    }

private:
    void loadCalibrationData()
    {
        cv::FileStorage fs(FILE_NAME_IN, cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open calibration file: %s", FILE_NAME_IN);
            throw std::runtime_error("Calibration file not found");
        }
        
        fs["cameraMatrix"] >> cameraMatrix_;
        fs["distCoeffs"] >> distCoeffs_;
        fs["rotationVector"] >> rotationVector_;
        fs["translationVector"] >> translationVector_;
        fs.release();
        
        RCLCPP_INFO(this->get_logger(), "Loaded calibration data from %s", FILE_NAME_IN);
    }
    
    // 이미지 캡처 및 publish (메인 스레드)
    void timer_callback()
    {
        if (!pDeviceTRI_ || !pDeviceHLT_) return;
        
        try
        {
            Arena::IImage* pImageHLT = pDeviceHLT_->GetImage(TIMEOUT);
            Arena::IImage* pImageTRI = pDeviceTRI_->GetImage(TIMEOUT);
            
            rclcpp::Time capture_time = this->now();
            if (pImageHLT && pImageTRI)
            {
                size_t hlt_width = pImageHLT->GetWidth();
                size_t hlt_height = pImageHLT->GetHeight();
                
                cv::Mat imageMatrixXYZ((int)hlt_height, (int)hlt_width, CV_32FC3);
                const uint16_t* input_data = reinterpret_cast<const uint16_t*>(pImageHLT->GetData());
                
                for (unsigned int ir = 0; ir < hlt_height; ++ir)
                {
                    for (unsigned int ic = 0; ic < hlt_width; ++ic)
                    {
                        ushort x_u16 = input_data[0];
                        ushort y_u16 = input_data[1];
                        ushort z_u16 = input_data[2];

                        imageMatrixXYZ.at<cv::Vec3f>(ir, ic)[0] = (float)(x_u16 * xyz_scale_mm_ + x_offset_mm_);
                        imageMatrixXYZ.at<cv::Vec3f>(ir, ic)[1] = (float)(y_u16 * xyz_scale_mm_ + y_offset_mm_);
                        imageMatrixXYZ.at<cv::Vec3f>(ir, ic)[2] = (float)(z_u16 * xyz_scale_mm_ + z_offset_mm_);

                        input_data += 4;
                    }
                }
                
                size_t tri_width = pImageTRI->GetWidth();
                size_t tri_height = pImageTRI->GetHeight();
                cv::Mat imageMatrixRGB((int)tri_height, (int)tri_width, CV_8UC3);
                memcpy(imageMatrixRGB.data, pImageTRI->GetData(), tri_height * tri_width * 3);
                
                // 이미지 즉시 publish (빠름)
                std_msgs::msg::Header header;
                header.stamp = capture_time;
                header.frame_id = "triton_camera";
                auto ros_image = cv_bridge::CvImage(header, "bgr8", imageMatrixRGB).toImageMsg();
                image_publisher_->publish(*ros_image);
                
                // 포인트클라우드 데이터를 큐에 추가 (별도 스레드에서 처리)
                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    
                    // 큐가 너무 커지면 오래된 데이터 제거
                    while (pointcloud_queue_.size() >= 3)
                    {
                        pointcloud_queue_.pop();
                    }
                    
                    PointCloudData data;
                    data.imageMatrixXYZ = imageMatrixXYZ.clone();
                    data.imageMatrixRGB = imageMatrixRGB.clone();
                    data.timestamp = capture_time;
                    pointcloud_queue_.push(std::move(data));
                }
                cv_.notify_one();
                
                pDeviceHLT_->RequeueBuffer(pImageHLT);
                pDeviceTRI_->RequeueBuffer(pImageTRI);
            }
        }
        catch (GenICam::GenericException& ge)
        {
            RCLCPP_ERROR(this->get_logger(), "Error capturing image: %s", ge.what());
        }
    }
    
    // 포인트클라우드 처리 스레드 (별도 스레드)
    void pointcloud_processing_thread()
    {
        while (running_)
        {
            PointCloudData data;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_.wait(lock, [this] { return !pointcloud_queue_.empty() || !running_; });
                
                if (!running_) break;
                
                data = std::move(pointcloud_queue_.front());
                pointcloud_queue_.pop();
            }
            
            // 포인트클라우드 처리 (시간 소요)
            process_and_publish_pointcloud(data);
        }
    }
    
    void process_and_publish_pointcloud(const PointCloudData& data)
    {
        size_t hlt_width = data.imageMatrixXYZ.cols;
        size_t hlt_height = data.imageMatrixXYZ.rows;
        
        int size = data.imageMatrixXYZ.rows * data.imageMatrixXYZ.cols;
        cv::Mat xyzPoints = data.imageMatrixXYZ.reshape(3, size);
        
        cv::Mat projectedPointsTRI;
        cv::projectPoints(
            xyzPoints, 
            rotationVector_, 
            translationVector_, 
            cameraMatrix_, 
            distCoeffs_, 
            projectedPointsTRI);
        
        sensor_msgs::msg::PointCloud2 cloud_msg;
        cloud_msg.header.stamp = data.timestamp;
        cloud_msg.header.frame_id = "helios_camera";
        cloud_msg.height = 1;
        cloud_msg.is_dense = false;
        cloud_msg.is_bigendian = false;
        
        std::vector<std::tuple<float, float, float, uint8_t, uint8_t, uint8_t>> valid_points;
        valid_points.reserve(hlt_width * hlt_height);
        
        for (int i = 0; i < static_cast<int>(hlt_width * hlt_height); i++)
        {
            float X = data.imageMatrixXYZ.at<cv::Vec3f>(i / hlt_width, i % hlt_width)[0];
            float Y = data.imageMatrixXYZ.at<cv::Vec3f>(i / hlt_width, i % hlt_width)[1];
            float Z = data.imageMatrixXYZ.at<cv::Vec3f>(i / hlt_width, i % hlt_width)[2];
            
            if (Z <= 0) continue;
            
            float projX = projectedPointsTRI.at<cv::Vec2f>(i, 0)[0];
            float projY = projectedPointsTRI.at<cv::Vec2f>(i, 0)[1];
            
            unsigned int colTRI = (unsigned int)std::round(projX);
            unsigned int rowTRI = (unsigned int)std::round(projY);
            
            if (rowTRI >= static_cast<unsigned int>(data.imageMatrixRGB.rows) ||
                colTRI >= static_cast<unsigned int>(data.imageMatrixRGB.cols))
                continue;
            
            uint8_t R = data.imageMatrixRGB.at<cv::Vec3b>(rowTRI, colTRI)[2];
            uint8_t G = data.imageMatrixRGB.at<cv::Vec3b>(rowTRI, colTRI)[1];
            uint8_t B = data.imageMatrixRGB.at<cv::Vec3b>(rowTRI, colTRI)[0];
            
            valid_points.emplace_back(X / 1000.0f, Y / 1000.0f, Z / 1000.0f, R, G, B);
        }
        
        cloud_msg.width = valid_points.size();
        
        sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
        modifier.setPointCloud2Fields(
            5,
            "x", 1, sensor_msgs::msg::PointField::FLOAT32,
            "y", 1, sensor_msgs::msg::PointField::FLOAT32,
            "z", 1, sensor_msgs::msg::PointField::FLOAT32,
            "rgb", 1, sensor_msgs::msg::PointField::UINT32,
            "intensity", 1, sensor_msgs::msg::PointField::FLOAT32
        );
        modifier.resize(valid_points.size());
        
        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_rgb(cloud_msg, "rgb");
        sensor_msgs::PointCloud2Iterator<float> iter_intensity(cloud_msg, "intensity");
        
        for (const auto& pt : valid_points)
        {
            *iter_x = std::get<0>(pt);
            *iter_y = std::get<1>(pt);
            *iter_z = std::get<2>(pt);
            
            iter_rgb[0] = std::get<5>(pt);
            iter_rgb[1] = std::get<4>(pt);
            iter_rgb[2] = std::get<3>(pt);
            iter_rgb[3] = 255;
            
            *iter_intensity = 0.5f;
            
            ++iter_x; ++iter_y; ++iter_z; ++iter_rgb; ++iter_intensity;
        }
        
        pointcloud_publisher_->publish(cloud_msg);
        
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                             "Publishing pointcloud with %zu points", valid_points.size());
    }
    
    void cleanup()
    {
        try
        {
            if (pDeviceTRI_)
            {
                pDeviceTRI_->StopStream();
                pSystem_->DestroyDevice(pDeviceTRI_);
                pDeviceTRI_ = nullptr;
            }
            if (pDeviceHLT_)
            {
                pDeviceHLT_->StopStream();
                pSystem_->DestroyDevice(pDeviceHLT_);
                pDeviceHLT_ = nullptr;
            }
            if (pSystem_)
            {
                Arena::CloseSystem(pSystem_);
                pSystem_ = nullptr;
            }
        }
        catch (...)
        {
        }
    }
    
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher_;
    
    Arena::ISystem* pSystem_;
    Arena::IDevice* pDeviceTRI_;
    Arena::IDevice* pDeviceHLT_;
    
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
    cv::Mat rotationVector_;
    cv::Mat translationVector_;
    
    double xyz_scale_mm_;
    double x_offset_mm_;
    double y_offset_mm_;
    double z_offset_mm_;
    
    // 멀티스레드용 멤버
    std::thread pointcloud_thread_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::queue<PointCloudData> pointcloud_queue_;
    std::atomic<bool> running_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalPublisher>());
    rclcpp::shutdown();
    return 0;
}