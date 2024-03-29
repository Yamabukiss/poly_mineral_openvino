cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_STANDARD 14)
project(polygon_mineral)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        dynamic_reconfigure
        std_msgs
        cv_bridge
        nodelet
        pluginlib
        )

generate_dynamic_reconfigure_options(
        cfg/dynamic.cfg
)

find_package(OpenCV REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
)


catkin_package(
        INCLUDE_DIRS include
        CATKIN_DEPENDS roscpp dynamic_reconfigure std_msgs cv_bridge dynamic_reconfigure
        LIBRARIES ${PROJECT_NAME}
)

# Declare a C++ library
#add_library(${PROJECT_NAME} src/main.cpp include/polygon_mineral/heal.h)
#add_library(${PROJECT_NAME} src/picodet_openvino.cpp)
add_executable(${PROJECT_NAME} src/picodet_openvino.cpp src/main.cpp)

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${InferenceEngine_LIBRARIES}
        ${NGRAPH_LIBRARIES}
        ${OpenCV_LIBS}
        )
