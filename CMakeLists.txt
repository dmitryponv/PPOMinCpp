cmake_minimum_required(VERSION 3.11 FATAL_ERROR) 
project(PPOmin)

#############################################
################## OPTIONS ##################
#############################################
set(CAFFE2_USE_CUDNN 1)
set(CAFFE2_USE_CUSPARSELT 1)
set(CAFFE2_USE_CUDSS 1)
set(CAFFE2_USE_CUFILE 1)

#############################################
################## TORCH DIR ################
#############################################
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch_r/share/cmake/Torch")
else()
    set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch_r/share/cmake/Torch")
endif()

# Torch
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/torch.cmake")

#############################################
################## CONTENT ##################
#############################################
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_executable(PPOmin src/main.cpp)
target_link_libraries(PPOmin "${TORCH_LIBRARIES}")
set_property(TARGET PPOmin PROPERTY CXX_STANDARD 17)

#############################################
############ COPY ONLY DLL FILES ############
#############################################
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(LIBTORCH_DLL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch_r/lib")
else()
    set(LIBTORCH_DLL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch_r/lib")
endif()

file(GLOB TORCH_DLLS "${LIBTORCH_DLL_DIR}/*.dll")

add_custom_command(TARGET PPOmin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${TORCH_DLLS}
    $<TARGET_FILE_DIR:PPOmin>
)
