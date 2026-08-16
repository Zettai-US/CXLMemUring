cmake_minimum_required(VERSION 3.10)
project(CiraRuntime VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Find required packages
find_package(Threads REQUIRED)

# Check for NUMA support
find_library(NUMA_LIBRARY numa)
if(NOT NUMA_LIBRARY)
    message(WARNING "libnuma not found, building without NUMA support")
    set(NUMA_SUPPORT OFF)
else()
    set(NUMA_SUPPORT ON)
    message(STATUS "Found libnuma: ${NUMA_LIBRARY}")
endif()

# Vortex SDK (optional, for hardware support)
set(VORTEX_HOME "$ENV{HOME}/vortex" CACHE PATH "Vortex SDK root directory")
if(EXISTS "${VORTEX_HOME}/runtime/include")
    set(VORTEX_FOUND ON)
    message(STATUS "Found Vortex SDK at: ${VORTEX_HOME}")
else()
    set(VORTEX_FOUND OFF)
    message(STATUS "Vortex SDK not found at ${VORTEX_HOME}, building with simulation stubs")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
if(VORTEX_FOUND)
    include_directories(${VORTEX_HOME}/runtime/include)
    include_directories(${VORTEX_HOME}/build/hw)  # For VX_config.h
endif()

# Source files for the runtime library
set(RUNTIME_SOURCES
    src/CiraRuntime.cpp
    vortex_verilator_sim.cpp
    vortex_device.cpp
    vortex_offload_backend.cpp
    cira_jit.cpp                      # JIT cost model (no LLVM dependency)
    cira_mwait.cpp                    # cache-resident wait (UMWAIT/TPAUSE/WFE)
    cira_mmio.cpp                     # CXL Type-2 control window + doorbell
)

# Optional: ORC JIT engine. Built only when LLVM is discoverable so the
# runtime can still be built standalone (e.g., embedded test environments).
find_package(LLVM CONFIG QUIET)
if(LLVM_FOUND)
    message(STATUS "LLVM ${LLVM_PACKAGE_VERSION} found at ${LLVM_DIR} — building cira_jit_engine")
    list(APPEND RUNTIME_SOURCES cira_jit_engine.cpp)
    set(CIRA_JIT_HAS_LLVM ON)
    llvm_map_components_to_libnames(CIRA_JIT_LLVM_LIBS
        Core
        IRReader
        BitReader
        Passes
        Analysis
        OrcJIT
        ExecutionEngine
        Support
        Target
        TargetParser
        native
    )
else()
    message(STATUS "LLVM not found — cira_jit_engine (ORC JIT) will not be built")
    set(CIRA_JIT_HAS_LLVM OFF)
endif()

# Create shared library
add_library(cira_runtime SHARED ${RUNTIME_SOURCES})

# Link libraries
target_link_libraries(cira_runtime
    PUBLIC
        Threads::Threads
        ${CMAKE_DL_LIBS}
)

if(NUMA_SUPPORT)
    target_link_libraries(cira_runtime PUBLIC ${NUMA_LIBRARY})
    target_compile_definitions(cira_runtime PUBLIC NUMA_SUPPORT)
endif()

if(VORTEX_FOUND)
    target_compile_definitions(cira_runtime PUBLIC VORTEX_FOUND)
    target_link_libraries(cira_runtime PUBLIC dl)
    # Link to Vortex runtime library
    target_link_libraries(cira_runtime PUBLIC ${VORTEX_HOME}/build/runtime/libvortex.so)
    target_link_libraries(cira_runtime PUBLIC ${VORTEX_HOME}/build/runtime/libsimx.so)
endif()

if(CIRA_JIT_HAS_LLVM)
    target_include_directories(cira_runtime PUBLIC ${LLVM_INCLUDE_DIRS})
    target_compile_definitions(cira_runtime PUBLIC ${LLVM_DEFINITIONS} CIRA_JIT_HAS_LLVM)
    target_link_libraries(cira_runtime PUBLIC ${CIRA_JIT_LLVM_LIBS})
endif()

# Create static library
add_library(cira_runtime_static STATIC ${RUNTIME_SOURCES})
set_target_properties(cira_runtime_static PROPERTIES OUTPUT_NAME cira_runtime)

target_link_libraries(cira_runtime_static
    PUBLIC
        Threads::Threads
        ${CMAKE_DL_LIBS}
)

if(NUMA_SUPPORT)
    target_link_libraries(cira_runtime_static PUBLIC ${NUMA_LIBRARY})
    target_compile_definitions(cira_runtime_static PUBLIC NUMA_SUPPORT)
endif()

if(VORTEX_FOUND)
    target_compile_definitions(cira_runtime_static PUBLIC VORTEX_FOUND)
    target_link_libraries(cira_runtime_static PUBLIC dl)
    target_link_libraries(cira_runtime_static PUBLIC ${VORTEX_HOME}/build/runtime/libvortex.so)
    target_link_libraries(cira_runtime_static PUBLIC ${VORTEX_HOME}/build/runtime/libsimx.so)
endif()

if(CIRA_JIT_HAS_LLVM)
    target_include_directories(cira_runtime_static PUBLIC ${LLVM_INCLUDE_DIRS})
    target_compile_definitions(cira_runtime_static PUBLIC ${LLVM_DEFINITIONS} CIRA_JIT_HAS_LLVM)
    target_link_libraries(cira_runtime_static PUBLIC ${CIRA_JIT_LLVM_LIBS})
endif()

# Set library properties
set_target_properties(cira_runtime PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER include/CiraRuntime.h
)

# Installation rules
install(TARGETS cira_runtime cira_runtime_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/cira
)

# Install all header files
install(DIRECTORY include/
    DESTINATION include/cira
    FILES_MATCHING PATTERN "*.h"
)

# Tests (opt-out so the runtime can still be built as a bare dependency)
option(CIRA_RUNTIME_BUILD_TESTS "Build the CiraRuntime test suite" ON)
if(CIRA_RUNTIME_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
    enable_testing()
    add_subdirectory(test)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "CiraRuntime Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  NUMA Support: ${NUMA_SUPPORT}")
message(STATUS "  Vortex SDK: ${VORTEX_FOUND}")
message(STATUS "  ORC JIT (LLVM): ${CIRA_JIT_HAS_LLVM}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
