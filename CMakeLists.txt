# project
cmake_minimum_required(VERSION 3.15.0)
cmake_policy(SET CMP0091 NEW)
project(detector LANGUAGES CXX)

# target
add_executable(detector "")
set_target_properties(detector PROPERTIES OUTPUT_NAME "detector")
set_target_properties(detector PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/linux/x86_64/release")
target_include_directories(detector PRIVATE
    include
    /usr/local/cuda/include
)
target_include_directories(detector SYSTEM PRIVATE
    /pkg/xmake/.xmake/packages/s/spdlog/v1.13.0/cca5b628022e49229205fe5617e4245e/include
    /usr/local/include/opencv4
    /usr/local/include/torch/csrc/api/include
    /usr/local/cuda-12.1/include
)
target_compile_definitions(detector PRIVATE
    USE_C10D_GLOO
    USE_C10D_NCCL
    USE_DISTRIBUTED
    USE_RPC
    USE_TENSORPIPE
)
target_compile_options(detector PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-m64>
    $<$<COMPILE_LANGUAGE:CXX>:-m64>
    $<$<COMPILE_LANGUAGE:C>:-DNDEBUG>
    $<$<COMPILE_LANGUAGE:CXX>:-DNDEBUG>
)
set_target_properties(detector PROPERTIES CXX_EXTENSIONS OFF)
target_compile_features(detector PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(detector PRIVATE $<$<CONFIG:Release>:-Ox -fp:fast>)
else()
    target_compile_options(detector PRIVATE -O3)
endif()
if(MSVC)
else()
    target_compile_options(detector PRIVATE -fvisibility=hidden)
endif()
if(MSVC)
    set_property(TARGET detector PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
target_link_libraries(detector PRIVATE
    opencv_gapi
    opencv_stitching
    opencv_aruco
    opencv_barcode
    opencv_bgsegm
    opencv_bioinspired
    opencv_ccalib
    opencv_cudabgsegm
    opencv_cudafeatures2d
    opencv_cudaobjdetect
    opencv_cudastereo
    opencv_dnn_objdetect
    opencv_dnn_superres
    opencv_dpm
    opencv_face
    opencv_freetype
    opencv_fuzzy
    opencv_hfs
    opencv_img_hash
    opencv_intensity_transform
    opencv_line_descriptor
    opencv_mcc
    opencv_quality
    opencv_rapid
    opencv_reg
    opencv_rgbd
    opencv_saliency
    opencv_stereo
    opencv_structured_light
    opencv_superres
    opencv_surface_matching
    opencv_tracking
    opencv_videostab
    opencv_wechat_qrcode
    opencv_xfeatures2d
    opencv_xobjdetect
    opencv_xphoto
    opencv_shape
    opencv_highgui
    opencv_datasets
    opencv_plot
    opencv_text
    opencv_ml
    opencv_phase_unwrapping
    opencv_cudacodec
    opencv_videoio
    opencv_cudaoptflow
    opencv_cudalegacy
    opencv_cudawarping
    opencv_optflow
    opencv_ximgproc
    opencv_video
    opencv_imgcodecs
    opencv_objdetect
    opencv_calib3d
    opencv_dnn
    opencv_features2d
    opencv_flann
    opencv_photo
    opencv_cudaimgproc
    opencv_cudafilters
    opencv_imgproc
    opencv_cudaarithm
    opencv_core
    opencv_cudev
    onnxruntime
    torch
    cuda
    nvrtc
    c10_cuda
    c10
    cudart
    nvToolsExt
    nvinfer
    pthread
)
target_link_directories(detector PRIVATE
    /usr/local/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda-12.1/lib64
)
target_link_options(detector PRIVATE
    -m64
    -Wl,-rpath,/usr/local/cuda-12.1/lib64:/usr/local/lib
    -Wl,-rpath,/lib/intel64:/lib/intel64_win:/lib/win-x64:/usr/local/cuda-12.1/lib64:/usr/local/lib
    -Wl,--no-as-needed,/usr/local/lib/libtorch_cpu.so
    -Wl,--no-as-needed,/usr/local/lib/libtorch_cuda.so
    -Wl,--no-as-needed,/usr/local/lib/libtorch.so
    -Wl,--as-needed
)
target_sources(detector PRIVATE
    src/main.cpp
    src/detector.cpp
    src/onnxinf.cpp
    src/opencvinf.cpp
    src/torchinf.cpp
)

