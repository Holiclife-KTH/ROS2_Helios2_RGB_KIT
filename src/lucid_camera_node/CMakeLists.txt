cmake_minimum_required(VERSION 3.8)
project(lucid_camera_node)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pcl_conversions REQUIRED)

set(PROJECT_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Arena SDK 경로 (lucid_camera_node 기준: ../../ArenaViewMP...)
set(ARENA_SDK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../ArenaViewMP_v_1.0.0.10_Linux_x64/ArenaSDK_Linux_x64")
get_filename_component(ARENA_SDK_PATH "${ARENA_SDK_PATH}" REALPATH)
set(ARENA_LIB_DIR "${ARENA_SDK_PATH}/lib64")
set(GENICAM_LIB_DIR "${ARENA_SDK_PATH}/GenICam/library/lib/Linux64_x64")

message(STATUS "ARENA_SDK_PATH = ${ARENA_SDK_PATH}")
message(STATUS "ARENA_LIB_DIR  = ${ARENA_LIB_DIR}")
message(STATUS "GENICAM_LIB_DIR= ${GENICAM_LIB_DIR}")

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  "${ARENA_SDK_PATH}/include"
  "${ARENA_SDK_PATH}/include/Arena"
  "${ARENA_SDK_PATH}/include/Save"
  "${ARENA_SDK_PATH}/GenICam/library/CPP/include"
)


# 실행 파일 생성 (한 번만)
add_executable(lucid_node src/lucid_node.cpp)

# 라이브러리 절대 경로
set(ARENA_LIB "${ARENA_LIB_DIR}/libarena.so")
set(GCBASE_LIB "${GENICAM_LIB_DIR}/libGCBase_gcc54_v3_3_LUCID.so")
set(GENAPI_LIB "${GENICAM_LIB_DIR}/libGenApi_gcc54_v3_3_LUCID.so")
set(LOG_LIB    "${GENICAM_LIB_DIR}/libLog_gcc54_v3_3_LUCID.so")

ament_target_dependencies(lucid_node
  rclcpp
  sensor_msgs
  cv_bridge
)

target_link_libraries(lucid_node
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${pcl_conversions_LIBRARIES}
  ${ARENA_LIB}
  ${GCBASE_LIB}
  ${GENAPI_LIB}
  ${LOG_LIB}
)

# RPATH 설정도 한 번만
set_target_properties(lucid_node PROPERTIES
  BUILD_RPATH "${ARENA_LIB_DIR}:${GENICAM_LIB_DIR}"
  INSTALL_RPATH "${ARENA_LIB_DIR}:${GENICAM_LIB_DIR}"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS lucid_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
