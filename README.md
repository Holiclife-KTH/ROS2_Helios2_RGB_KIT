# ROS2_Helios2_RGB_KIT
Lucid Vision 3D+RGB IP67 Kit (Helios2 / Helios2+ / Helios2 Ray & Triton 3.2MP Kit) ROS2 Driver

This package provides a ROS 2 driver for the Lucid Vision 3D+RGB Kit, enabling the acquisition of synchronized PointCloud data (from Helios2 series) and Raw Image data (from Triton cameras). It is built upon the Lucid Arena SDK.

## 📋 Prerequisites
### 1. Arena SDK
You must install the Arena SDK before building this package.

Download: Lucid Vision Downloads Hub

Note: Ensure the SDK is properly sourced or installed in the default system path so the compiler can find the headers and libraries.

## 🚀 Key Features
PointCloud Publishing: Publishes high-precision 3D point clouds from Helios2/2+/Ray.

Image Streaming: Captures and publishes raw images from the Triton 3.2MP camera.

Device Synchronization: Supports bridged operation between the ToF and RGB sensors using the Arena SDK.

## 🛠️ Calibration & Setup
[!IMPORTANT]

Before running the driver, you must calibrate the cameras to obtain the Intrinsics (Distortion Matrix) and Extrinsics for the 3D-RGB registration.

Lens Calibration: Use the camera_calibration ROS 2 package or Arena SDK tools to get the Triton camera's calibration file.

Bridge Configuration: Use the Arena SDK to verify that the Triton and Helios cameras are correctly identified and accessible on the network.
