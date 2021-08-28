#set(HEADER_LIST "${ModernCMakeExample_SOURCE_DIR}/include/modern/lib.hpp")

#OpenCV
FIND_PACKAGE(OpenCV REQUIRED)

# DLIB
include(FetchContent)
FetchContent_Declare(dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG        v19.18
)
FetchContent_MakeAvailable(dlib)

set(USE_AVX_INSTRUCTIONS ON)
set(USE_SSE2_INSTRUCTIONS OFF)
set(USE_SSE4_INSTRUCTIONS OFF)
set(DLIB_NO_GUI_SUPPORT OFF)

# cxxopts - Lightweight C++ command line option parser   https://github.com/jarro2783/cxxopts
include(FetchContent)
FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG        v2.2.1
        GIT_SHALLOW    TRUE
)
set(CXXOPTS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CXXOPTS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CXXOPTS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(CXXOPTS_ENABLE_WARNINGS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cxxopts)

add_executable(diplomka main.cpp)
target_link_libraries(diplomka ${OpenCV_LIBS} dlib::dlib)
