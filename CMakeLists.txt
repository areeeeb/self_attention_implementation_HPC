cmake_minimum_required(VERSION 3.10)
project(attention_implementation)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Add external dependencies
include(FetchContent)

# Add nlohmann/json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Add cnpy
FetchContent_Declare(
    cnpy
    GIT_REPOSITORY https://github.com/rogersce/cnpy.git
    GIT_TAG master
)
FetchContent_MakeAvailable(cnpy)

# Add executable
add_executable(main
    main.cpp
    naive.cpp
    optimized.cpp
)

# Link libraries
target_link_libraries(main PRIVATE
    OpenMP::OpenMP_CXX
    nlohmann_json::nlohmann_json
    cnpy
)

# Include directories
target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(main PRIVATE -O2)
endif()