cmake_minimum_required(VERSION 3.10)
project(BilliardVision)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

set(BILLIARD_VISION_SOURCES
    src/main.cpp
    src/BallTracking.cpp
    src/BallDetection.cpp
    src/BallClassifier.cpp
    src/FieldDetection.cpp
    src/ColoredMask.cpp
)

add_executable(BilliardVision ${BILLIARD_VISION_SOURCES})
target_link_libraries(BilliardVision ${OpenCV_LIBS})

add_executable(Metrics src/metrics.cpp)
target_link_libraries(Metrics ${OpenCV_LIBS})
