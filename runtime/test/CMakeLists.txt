cmake_minimum_required(VERSION 3.10)

# One helper so every test gets the same include paths and link line.
function(cira_add_runtime_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE cira_runtime)
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# Cost model: pure decision logic, no LLVM and no device required.
cira_add_runtime_test(test_cira_jit test_cira_jit.cpp)

# Dispatch path: cache-resident wait + CXL control window + doorbell protocol.
# Runs against an emulated (anonymous-memory) window, so no FPGA is needed.
cira_add_runtime_test(test_cira_offload_path test_cira_offload_path.cpp)

# Offload engine / two-pass profiling.
cira_add_runtime_test(test_offload_engine test_offload_engine.cpp)
cira_add_runtime_test(test_twopass_runtime test_twopass_runtime.cpp)

# ORC specializer — only built when the runtime was configured with LLVM.
if(CIRA_JIT_HAS_LLVM)
    cira_add_runtime_test(test_cira_jit_engine test_cira_jit_engine.cpp)
endif()

# Cost-model microbenchmark — built, but not part of the test suite.
add_executable(bench_cira_jit bench_cira_jit.cpp)
target_link_libraries(bench_cira_jit PRIVATE cira_runtime)
target_include_directories(bench_cira_jit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)