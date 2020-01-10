// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

#include "pmlc/dialect/xsmm/types.h"
#include "pmlc/util/interfaces.h"

namespace pmlc {
namespace dialect {
namespace xsmm {

using mlir::AbstractOperation;
using mlir::ArrayRef;
using mlir::Attribute;
using mlir::Builder;
using mlir::IndexType;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::NamedAttribute;
using mlir::Op;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::OpFoldResult;
using mlir::OpInterface;
using mlir::StringRef;
using mlir::Type;
using mlir::Value;
using mlir::ValueRange;
using util::GenericBuilder;

namespace OpTrait = mlir::OpTrait;

#include "pmlc/dialect/xsmm/interfaces.h.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/xsmm/ops.h.inc"

}  // namespace xsmm
}  // namespace dialect
}  // namespace pmlc
