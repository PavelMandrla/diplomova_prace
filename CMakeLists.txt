cmake_minimum_required(VERSION 3.17)

project(DiplomovaPrace VERSION 1.0
        DESCRIPTION "Diplomová práce - Bc. Pavel Mandrla"
        LANGUAGES CXX)

include(FetchContent)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include_directories(include)
add_subdirectory(src)