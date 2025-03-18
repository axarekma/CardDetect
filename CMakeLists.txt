cmake_minimum_required(VERSION 3.10)
project(MTGCardDetector)

# Use C++17
set(CMAKE_CXX_STANDARD 17)

# Set path to data files
set(DATA_DIR "${CMAKE_SOURCE_DIR}/data")
add_definitions(-DDATA_DIR=${DATA_DIR})

# enable symbols 
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")  # Keep debug info in release
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /Zi")  # Debug info with optimizations
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi")  # Debug info in debug mode

# Find OpenCV
# find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} include/)

# Enable specific warnings for MSVC
if(MSVC)
  add_compile_options(
    /W4                  # Set warning level 4 for MSVC (quite strict)
    /WX                  # Treat warnings as errors (optional)
  )
endif()

# Source files
set(SRC_FILES 
    src/CameraCapture.cpp 
    src/FrameProcessor.cpp 
    src/DisplayManager.cpp 
    src/SimilarityModel.cpp 
    main.cpp
)


# Define executable
add_executable(MTGCardDetector ${SRC_FILES})

# Link 
target_link_libraries(MTGCardDetector PRIVATE ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
