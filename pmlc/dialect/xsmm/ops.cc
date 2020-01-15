// Copyright 2019, Intel Corporation

#include "pmlc/dialect/xsmm/ops.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "llvm/ADT/StringSwitch.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/util/util.h"

#define DEBUG_TYPE "xsmm"

namespace pmlc {
namespace dialect {
namespace xsmm {

#include "pmlc/dialect/xsmm/interfaces.cc.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/xsmm/ops.cc.inc"

}  // namespace xsmm
}  // namespace dialect
}  // namespace pmlc
