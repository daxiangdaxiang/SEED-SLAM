cmake_minimum_required(VERSION 3.16)
project(DranpprMatchedBackends LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(GTSAM REQUIRED)
find_package(nlohmann_json REQUIRED)
foreach(name IMESA_SOURCE_DIR RIMESA_SOURCE_DIR RISAM_SOURCE_DIR RISAM_LIBRARY)
  if(NOT DEFINED ${name})
    message(FATAL_ERROR "Set the pinned ${name}")
  endif()
endforeach()
set(RIMESA_LINK_TARGET "${RIMESA_LIBRARY}")
set(RIMESA_LOCAL_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}matched_rimesa${CMAKE_STATIC_LIBRARY_SUFFIX}")
if(NOT DEFINED RIMESA_LIBRARY OR RIMESA_LIBRARY STREQUAL RIMESA_LOCAL_LIBRARY)
  add_library(matched_rimesa STATIC
    "${RIMESA_SOURCE_DIR}/rimesa/src/rimesa.cpp"
    "${RIMESA_SOURCE_DIR}/rimesa/src/biased_prior.cpp"
    "${RIMESA_SOURCE_DIR}/rimesa/src/initialization.cpp"
    "${RIMESA_SOURCE_DIR}/rimesa/src/misc.cpp")
  target_include_directories(matched_rimesa PUBLIC
    "${RIMESA_SOURCE_DIR}/rimesa/include" "${RISAM_SOURCE_DIR}/risam/include"
    "${GTSAM_INCLUDE_DIR}")
  target_link_libraries(matched_rimesa PUBLIC "${RISAM_LIBRARY}" gtsam)
  set(RIMESA_LIBRARY "${RIMESA_LOCAL_LIBRARY}" CACHE FILEPATH "Pinned riMESA library")
  set(RIMESA_LINK_TARGET matched_rimesa)
endif()
add_library(matched_imesa STATIC
  "${IMESA_SOURCE_DIR}/src/imesa.cpp"
  "${IMESA_SOURCE_DIR}/src/biased_prior.cpp")
target_include_directories(matched_imesa PUBLIC "${IMESA_SOURCE_DIR}/include" "${GTSAM_INCLUDE_DIR}")
target_link_libraries(matched_imesa PUBLIC gtsam)
add_executable(native-matched-baseline baseline.cpp ../../src/NativeBetweenPose2Factor.cpp)
target_include_directories(native-matched-baseline PRIVATE ../../include
  "${GTSAM_INCLUDE_DIR}" "${RIMESA_SOURCE_DIR}/rimesa/include" "${RISAM_SOURCE_DIR}/risam/include")
target_link_libraries(native-matched-baseline PRIVATE matched_imesa
  "${RIMESA_LINK_TARGET}" "${RISAM_LIBRARY}" gtsam nlohmann_json::nlohmann_json)
add_executable(native-consensus-bias consensus_bias.cpp)
target_include_directories(native-consensus-bias PRIVATE "${GTSAM_INCLUDE_DIR}"
  "${RIMESA_SOURCE_DIR}/rimesa/include" "${RISAM_SOURCE_DIR}/risam/include")
target_link_libraries(native-consensus-bias PRIVATE
  "${RIMESA_LINK_TARGET}" "${RISAM_LIBRARY}" gtsam nlohmann_json::nlohmann_json)
