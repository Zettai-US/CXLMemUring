
#include "Dialect/RemoteMemRef.h"
#include "Dialect/RemoteMemDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::cira;

// 实现 get 方法
RemoteMemRefType RemoteMemRefType::get(Type elementType, unsigned cacheID) {
    return Base::get(elementType.getContext(), elementType, cacheID);
}

// 实现 classof 方法
// NB: must compare TypeIDs directly; llvm::isa<RemoteMemRefType> dispatches
// back to this function and would recurse forever.
bool RemoteMemRefType::classof(Type type) { return type && type.getTypeID() == RemoteMemRefType::getTypeID(); }

// 实现 isValidElementType 方法
bool RemoteMemRefType::isValidElementType(Type elementType) {
    if (!elementType)
        return false;
    if (!llvm::isa<mlir::MemRefType>(elementType) && !llvm::isa<LLVM::LLVMPointerType>(elementType) &&
        !llvm::isa<mlir::UnrankedMemRefType>(elementType))
        return false;
    return true;
}

// 实现 getElementType 方法
Type RemoteMemRefType::getElementType() const {
    auto *storage = static_cast<detail::RemoteMemRefTypeStorage *>(getImpl());
    return storage->elementType;
}

// 实现 getCacheID 方法
unsigned RemoteMemRefType::getCacheID() const {
    auto *storage = static_cast<detail::RemoteMemRefTypeStorage *>(getImpl());
    return storage->cacheID;
}