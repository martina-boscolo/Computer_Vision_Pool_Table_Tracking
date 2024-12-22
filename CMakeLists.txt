cmake_minimum_required(VERSION 3.5)
project(BilliardFinalProject)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Set include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

# Set source directory
set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)

# Define executables and their source files
add_executable(segmentation_and_classification 
    ${SRC_DIR}/Main_Field_Balls.cpp 
    ${SRC_DIR}/Balls_segmentation.cpp 
    ${SRC_DIR}/ball_classification.cpp 
    ${SRC_DIR}/Classes.cpp 
    ${SRC_DIR}/Field_detection.cpp 
    ${SRC_DIR}/Utilities.cpp 
    ${SRC_DIR}/Metrics.cpp
)

add_executable(tracking
   ${SRC_DIR}/main_tracking.cpp
   ${SRC_DIR}/tracking.cpp
   ${SRC_DIR}/Classes.cpp
   ${SRC_DIR}/table_orientation.cpp
)

add_executable(metrics  
    ${SRC_DIR}/Classes.cpp  
    ${SRC_DIR}/Utilities.cpp 
    ${SRC_DIR}/Metrics.cpp 
    ${SRC_DIR}/Main_metrics.cpp
)

add_executable(complete  
    ${SRC_DIR}/Classes.cpp  
    ${SRC_DIR}/Utilities.cpp 
    ${SRC_DIR}/Metrics.cpp 
    ${SRC_DIR}/tracking.cpp
    ${SRC_DIR}/table_orientation.cpp
    ${SRC_DIR}/Balls_segmentation.cpp 
    ${SRC_DIR}/ball_classification.cpp 
    ${SRC_DIR}/Field_detection.cpp 
    ${SRC_DIR}/Utilities.cpp 
    ${SRC_DIR}/main.cpp
)

# Link libraries
target_link_libraries(segmentation_and_classification ${OpenCV_LIBS})
target_link_libraries(tracking ${OpenCV_LIBS})
target_link_libraries(metrics ${OpenCV_LIBS})
target_link_libraries(complete ${OpenCV_LIBS})
