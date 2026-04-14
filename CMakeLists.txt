cmake_minimum_required(VERSION 3.20)
set(CMAKE_C_COMPILER /storage/opt/nvidia-2025-259/Linux_x86_64/2025/compilers/bin/nvc)
set(CMAKE_CUDA_COMPILER /storage/opt/nvidia-2025-259/Linux_x86_64/2025/cuda/12.9/bin/nvcc)
project(ode_art LANGUAGES C CUDA)

set(ACC_INCLUDES
    /usr/lib/gcc/x86_64-redhat-linux/8/include
)
set(ACC_COMPILE_FLAGS "--gcc-toolchain=/usr;-Kieee;-acc=multicore")

set(SYCL_COMPILER clang++)
set(SYCL_CUDA_PATH /storage/opt/cuda/cuda-12.1)
set(SYCL_GPU_ARCH sm_70)
set(SYCL_FLAGS
    -fsycl
    -fsycl-targets=nvptx64-nvidia-cuda,spir64
    --cuda-path=${SYCL_CUDA_PATH}
    -Xsycl-target-backend=nvptx64-nvidia-cuda
    --cuda-gpu-arch=${SYCL_GPU_ARCH}
)

function(add_acc_executable name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${ACC_INCLUDES})
    target_compile_options(${name} PRIVATE ${ACC_COMPILE_FLAGS})
    target_link_options(${name} PRIVATE ${ACC_COMPILE_FLAGS})
endfunction()

function(add_cuda_executable name src)
    add_executable(${name} ${src})
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    )
    target_link_libraries(${name} PRIVATE gomp m stdc++)
endfunction()

function(add_sycl_executable name src)
    add_custom_target(${name} ALL
        COMMAND ${SYCL_COMPILER} ${SYCL_FLAGS} ${src} -o ${CMAKE_BINARY_DIR}/${name}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Compiling SYCL: ${src} -> ${name}"
        VERBATIM
    )
endfunction()

file(GLOB C_FILES "*.c")
foreach(src ${C_FILES})
    get_filename_component(name ${src} NAME_WE)
    if(src MATCHES "_acc_" OR src MATCHES "_seq")
        add_acc_executable(${name} ${src})
    endif()
endforeach()

file(GLOB CU_FILES "*.cu")
foreach(src ${CU_FILES})
    get_filename_component(name ${src} NAME_WE)
    add_cuda_executable(${name} ${src})
endforeach()

file(GLOB DP_FILES "*.dp.cpp")
foreach(src ${DP_FILES})
    get_filename_component(base ${src} NAME)
    string(REPLACE ".dp.cpp" "" stem ${base})
    string(REPLACE "_cuda_" "_sycl_" name ${stem})
    add_sycl_executable(${name} ${src})
endforeach()
