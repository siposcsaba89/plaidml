// Copyright 2019, Intel Corporation

#include "pmlc/dialect/xsmm/dialect.h"

#include <utility>

#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/DebugStringHelper.h"

#include "base/util/logging.h"
#include "pmlc/dialect/xsmm/ops.h"

#define DEBUG_TYPE "xsmm"

namespace pmlc {
namespace dialect {
namespace xsmm {

Dialect::Dialect(mlir::MLIRContext* ctx) : mlir::Dialect(getDialectNamespace(), ctx) {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/xsmm/ops.cc.inc"
      >();
}

std::string Dialect::getCanonicalOpName(llvm::StringRef name) {
  return llvm::formatv("{0}.{1}", getDialectNamespace(), name).str();
}

static mlir::DialectRegistration<Dialect> XSMMOps;

}  // namespace xsmm
}  // namespace dialect
}  // namespace pmlc
