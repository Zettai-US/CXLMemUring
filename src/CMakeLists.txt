add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Lowering)
# add_subdirectory(Analysis)  # Temporarily disabled

# Gather MLIR dialect and conversion libs collected by AddMLIR
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# Base libs for CIR and Clang
set(CIRA_BASE_LIBS
  MLIRCIR
  clang-cpp
)

# MLIR opt/tooling and core libraries required by MlirOptMain and pass registration
set(CIRA_MLIR_LIBS
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRIR
  MLIRSupport
)

add_llvm_executable(cira driver.cpp)
# llvm_update_compile_flags(cira)

# Link everything needed for the opt-style driver:
# - our CIR/Clang libs
# - MLIR opt/tooling/core
# - all collected dialect and conversion libs (provides pass/dialect registrars)
target_link_libraries(cira PUBLIC
  ${CIRA_BASE_LIBS}
  ${CIRA_MLIR_LIBS}
  # ClangIR lowering through MLIR and helpers (provides createConvertCIRToMLIRPass)
  clangCIRLoweringThroughMLIR
  clangCIRLoweringHelpers
  # Ensure CIR transform registrations are linked
  MLIRCIRTransforms
  ${dialect_libs}
  ${conversion_libs}
)

add_llvm_executable(cira-link tools/cira-link.cpp)
target_link_libraries(cira-link PUBLIC
  ${CIRA_BASE_LIBS}
  MLIRParser
  MLIRIR
  MLIRSupport
  MLIRRemoteMem
  MLIRCIRTransforms
  ${dialect_libs}
)
