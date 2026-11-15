# Build-tree RPATH so ctest can load libviewfeat.so, libtensorrtbuffer.so, and
# TensorRT without LD_LIBRARY_PATH. --disable-new-dtags stores DT_RPATH, which
# the loader also uses for transitive dependencies.
function(viewfeat_test_runtime target)
  set_target_properties(${target} PROPERTIES
    BUILD_RPATH "$<TARGET_FILE_DIR:viewfeat>;$<TARGET_FILE_DIR:tensorrtbuffer>;${TENSORRT_LIB_DIR}"
  )
  target_link_options(${target} PRIVATE "LINKER:--disable-new-dtags")
endfunction()

add_executable(test_engine_io test_engine_io.cpp)
target_compile_features(test_engine_io PRIVATE cxx_std_14)
target_include_directories(test_engine_io PRIVATE
  "${PROJECT_SOURCE_DIR}/src"
  "${TENSORRT_INCLUDE_DIR}"
  ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(test_engine_io PRIVATE viewfeat "${NVINFER_LIB}")
viewfeat_test_runtime(test_engine_io)

add_test(NAME engine_io COMMAND test_engine_io)
add_test(NAME engine_io_garbage COMMAND test_engine_io --garbage)
set_tests_properties(engine_io PROPERTIES TIMEOUT 60)
set_tests_properties(engine_io_garbage PROPERTIES
  TIMEOUT 90
  SKIP_RETURN_CODE 77
)

add_executable(test_config test_config.cpp)
target_compile_features(test_config PRIVATE cxx_std_14)
target_compile_definitions(test_config PRIVATE VIEWFEAT_SAMPLE_CONFIG="${PROJECT_SOURCE_DIR}/config/config.yaml")
target_include_directories(test_config PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(test_config PRIVATE viewfeat)
viewfeat_test_runtime(test_config)
add_test(NAME config_load COMMAND test_config)
set_tests_properties(config_load PROPERTIES TIMEOUT 60)

add_executable(test_images test_images.cpp)
target_compile_features(test_images PRIVATE cxx_std_14)
target_link_libraries(test_images PRIVATE viewfeat)
if(TARGET OpenCV::imgcodecs)
  target_link_libraries(test_images PRIVATE OpenCV::imgcodecs)
  if(TARGET OpenCV::imgproc)
    target_link_libraries(test_images PRIVATE OpenCV::imgproc)
  endif()
else()
  target_include_directories(test_images PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(test_images PRIVATE ${OpenCV_LIBS})
endif()
viewfeat_test_runtime(test_images)

set(VIEWFEAT_TEST_CONFIG "${PROJECT_SOURCE_DIR}/config/config.yaml")
set(VIEWFEAT_TEST_WEIGHTS "${PROJECT_SOURCE_DIR}/weights")
set(VIEWFEAT_TEST_IMAGE0 "${PROJECT_SOURCE_DIR}/example/image/image0.png")
set(VIEWFEAT_TEST_IMAGE1 "${PROJECT_SOURCE_DIR}/example/image/image1.png")
set(VIEWFEAT_TEST_FRAME0 "${PROJECT_SOURCE_DIR}/example/image/freiburg_sequence/1341847980.722988.png")
set(VIEWFEAT_TEST_FRAME1 "${PROJECT_SOURCE_DIR}/example/image/freiburg_sequence/1341847981.726650.png")

function(viewfeat_add_image_test name family timeout)
  add_test(
    NAME ${name}
    COMMAND test_images ${family}
            "${VIEWFEAT_TEST_CONFIG}" "${VIEWFEAT_TEST_WEIGHTS}"
            "${VIEWFEAT_TEST_IMAGE0}" "${VIEWFEAT_TEST_IMAGE1}"
            "${VIEWFEAT_TEST_FRAME0}" "${VIEWFEAT_TEST_FRAME1}"
  )
  set_tests_properties(${name} PROPERTIES
    TIMEOUT ${timeout}
    SKIP_RETURN_CODE 77
  )
endfunction()

viewfeat_add_image_test(example_superpoint default 180)
viewfeat_add_image_test(example_aliked aliked 180)
viewfeat_add_image_test(example_cosplace cosplace 180)
viewfeat_add_image_test(example_eigenplaces eigenplaces 300)
viewfeat_add_image_test(example_flexglue superpoint_flexglue 300)

add_executable(test_perf test_perf.cpp)
target_compile_features(test_perf PRIVATE cxx_std_14)
target_include_directories(test_perf PRIVATE
  "${PROJECT_SOURCE_DIR}/src"
  ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(test_perf PRIVATE viewfeat ${CUDA_LIBRARIES})
if(TARGET OpenCV::imgcodecs)
  target_link_libraries(test_perf PRIVATE OpenCV::imgcodecs)
  if(TARGET OpenCV::imgproc)
    target_link_libraries(test_perf PRIVATE OpenCV::imgproc)
  endif()
else()
  target_include_directories(test_perf PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(test_perf PRIVATE ${OpenCV_LIBS})
endif()
viewfeat_test_runtime(test_perf)

set(VIEWFEAT_PERF_BASELINE "${PROJECT_SOURCE_DIR}/test/data/perf_baseline.bin")
add_test(
  NAME perf_infer
  COMMAND test_perf
          "${VIEWFEAT_TEST_CONFIG}" "${VIEWFEAT_TEST_WEIGHTS}"
          "${VIEWFEAT_TEST_IMAGE0}" "${VIEWFEAT_TEST_IMAGE1}"
          --baseline "${VIEWFEAT_PERF_BASELINE}"
)
set_tests_properties(perf_infer PROPERTIES
  TIMEOUT 900
  SKIP_RETURN_CODE 77
)
