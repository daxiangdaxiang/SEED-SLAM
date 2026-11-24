find_package(Eigen3 CONFIG REQUIRED)
find_package(GTSAM REQUIRED)
add_library(DPGONativeModel src/IncrementalLocalModel.cpp src/NativeSchurModel.cpp
  src/NativeBetweenPose2Factor.cpp src/NativeLocalOptimizer.cpp src/NativeInterface.cpp
  src/IncrementalConditionalModel.cpp)
target_include_directories(DPGONativeModel PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include" "${GTSAM_INCLUDE_DIR}")
target_link_libraries(DPGONativeModel PUBLIC gtsam)
find_package(nlohmann_json REQUIRED)
add_executable(native-local-audit experiments/native_local/audit.cpp)
target_link_libraries(native-local-audit PRIVATE DPGONativeModel nlohmann_json::nlohmann_json)
add_executable(native-causal-replay experiments/native_local/replay.cpp)
target_link_libraries(native-causal-replay PRIVATE DPGONativeModel nlohmann_json::nlohmann_json)
add_executable(native-joint-audit experiments/native_local/joint.cpp)
target_link_libraries(native-joint-audit PRIVATE DPGONativeModel nlohmann_json::nlohmann_json)
add_executable(native-joint-replay experiments/native_local/joint_replay.cpp)
target_link_libraries(native-joint-replay PRIVATE DPGONativeModel nlohmann_json::nlohmann_json)
add_executable(native-pose3-calibration-audit experiments/native_local/pose3_calibration.cpp)
target_link_libraries(native-pose3-calibration-audit PRIVATE DPGONativeModel nlohmann_json::nlohmann_json)
add_executable(native-conditional-audit experiments/native_local/conditional_audit.cpp)
target_link_libraries(native-conditional-audit PRIVATE DPGONativeModel nlohmann_json::nlohmann_json)
if(BUILD_DPGO_TESTS)
  add_executable(testIncrementalLocalModel
    tests/testIncrementalLocalModel.cpp tests/testNativeSchurModel.cpp
    tests/testNativeLocalOptimizer.cpp tests/testNativeInterface.cpp
    tests/testIncrementalConditionalModel.cpp tests/testNativeChartAudit.cpp
    tests/testNativeLazyRefresh.cpp tests/testNativeSubspace.cpp)
  target_link_libraries(testIncrementalLocalModel PRIVATE DPGONativeModel gtest_main nlohmann_json::nlohmann_json)
  add_test(NAME native_local_model COMMAND testIncrementalLocalModel)
endif()
# Match the pinned GTSAM build's Eigen allocation ABI. With Eigen 3.3.7,
# limiting MAX_ALIGN_BYTES alone does not change DEFAULT_ALIGN_BYTES under AVX.
# Leave legacy targets unchanged; native consumers must use compatible flags.
foreach(native_target DPGONativeModel testIncrementalLocalModel native-local-audit native-causal-replay native-joint-audit native-joint-replay native-conditional-audit native-pose3-calibration-audit)
  if(TARGET ${native_target})
    get_target_property(native_options ${native_target} COMPILE_OPTIONS)
    list(REMOVE_ITEM native_options "-march=native")
    set_target_properties(${native_target} PROPERTIES COMPILE_OPTIONS "${native_options}")
  endif()
endforeach()
