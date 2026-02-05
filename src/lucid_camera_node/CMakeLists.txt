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

# Arena SDK 경로 (사용자 지정 경로 반영)
set(ARENA_SDK_PATH "/home/irol/workspace/ros2_helios2_rgb_kit/ArenaViewMP_v_1.0.0.10_Linux_x64/ArenaSDK_Linux_x64")
set(ARENA_LIB_DIR "${ARENA_SDK_PATH}/lib64")
set(GENICAM_LIB_DIR "${ARENA_SDK_PATH}/GenICam/library/lib/Linux64_x64")

# 2. 헤더 포함
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  "${ARENA_SDK_PATH}/include/Arena"
  "${ARENA_SDK_PATH}/include/Save"
  "${ARENA_SDK_PATH}/GenICam/library/CPP/include"
)

# 3. 라이브러리 절대 경로 직접 지정 (find_library 실패 방지)
set(ARENA_LIB "${ARENA_LIB_DIR}/libarena.so")
set(GCBASE_LIB "${GENICAM_LIB_DIR}/libGCBase_gcc54_v3_3_LUCID.so")
set(GENAPI_LIB "${GENICAM_LIB_DIR}/libGenApi_gcc54_v3_3_LUCID.so")
set(LOG_LIB    "${GENICAM_LIB_DIR}/libLog_gcc54_v3_3_LUCID.so")

# 4. 실행 파일 생성
add_executable(lucid_node src/lucid_node.cpp)

# ament_target_dependencies 추가
ament_target_dependencies(lucid_node
  rclcpp
  sensor_msgs
  cv_bridge
)

# 5. 링크 설정 (pcl_conversions는 target 이름 대신 라이브러리 변수 사용)
target_link_libraries(lucid_node
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${pcl_conversions_LIBRARIES}
  ${ARENA_LIB}
  ${GCBASE_LIB}
  ${GENAPI_LIB}
  ${LOG_LIB}
)


install(TARGETS lucid_node
  DESTINATION lib/${PROJECT_NAME}
)

# 7. RPATH 설정 (기존 내용 유지)
set_target_properties(lucid_node PROPERTIES 
  INSTALL_RPATH "${ARENA_LIB_DIR}:${GENICAM_LIB_DIR}"
  BUILD_WITH_INSTALL_RPATH TRUE
)
ament_package()