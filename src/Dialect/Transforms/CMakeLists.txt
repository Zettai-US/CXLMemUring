add_mlir_dialect_library(MLIRRMEMTransforms
        SearchRemote.cpp
        BackwardSlice.cpp
        RegionFormation.cpp
        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include

        DEPENDS
        RemoteMemPassIncGen

        LINK_LIBS PUBLIC
        MLIRAnalysis
        MLIRDataLayoutInterfaces
        MLIRFuncDialect
        MLIRMemRefDialect
        MLIRLLVMDialect
        MLIRVectorDialect
        MLIRAffineDialect
        MLIRSCFDialect
        MLIRRemoteMem
        MLIRTransforms
)