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
    set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch_d/share/cmake/Torch")
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
set(LIBTORCH_DLL_DIR_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch_d/lib")
set(LIBTORCH_DLL_DIR_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch_r/lib")

file(GLOB TORCH_DLLS_DEBUG "${LIBTORCH_DLL_DIR_DEBUG}/*.dll")
file(GLOB TORCH_DLLS_RELEASE "${LIBTORCH_DLL_DIR_RELEASE}/*.dll")

# Define absolute destination folders
set(DEST_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/build/Debug")
set(DEST_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/build/Release")

# Create destination directories
file(MAKE_DIRECTORY "${DEST_DEBUG}")
file(MAKE_DIRECTORY "${DEST_RELEASE}")

# Copy Debug DLLs and print the command
foreach(dll ${TORCH_DLLS_DEBUG})
    message(STATUS "Copying ${dll} to ${DEST_DEBUG}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "${DEST_DEBUG}")
endforeach()

# Copy Release DLLs and print the command
foreach(dll ${TORCH_DLLS_RELEASE})
    message(STATUS "Copying ${dll} to ${DEST_RELEASE}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "${DEST_RELEASE}")
endforeach()