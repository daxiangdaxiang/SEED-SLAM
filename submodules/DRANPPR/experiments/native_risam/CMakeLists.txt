cmake_minimum_required(VERSION 3.16)
project(DranpprNativeRisamAudit LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(GTSAM REQUIRED)
find_package(nlohmann_json REQUIRED)
foreach(path RISAM_SOURCE_DIR RISAM_LIBRARY)
  if(NOT DEFINED ${path})
    message(FATAL_ERROR "Set ${path} to the explicitly pinned upstream source/library")
  endif()
endforeach()
add_executable(native-risam-replay replay.cpp)
target_include_directories(native-risam-replay PRIVATE
  "${RISAM_SOURCE_DIR}/risam/include" "${GTSAM_INCLUDE_DIR}")
target_link_libraries(native-risam-replay PRIVATE
  "${RISAM_LIBRARY}" gtsam nlohmann_json::nlohmann_json)
add_executable(native-risam-checkpoint checkpoint.cpp)
target_include_directories(native-risam-checkpoint PRIVATE
  "${RISAM_SOURCE_DIR}/risam/include" "${GTSAM_INCLUDE_DIR}")
target_link_libraries(native-risam-checkpoint PRIVATE
  "${RISAM_LIBRARY}" gtsam nlohmann_json::nlohmann_json)
