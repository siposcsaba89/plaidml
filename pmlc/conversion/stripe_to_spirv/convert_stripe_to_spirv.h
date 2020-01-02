// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/Module.h"

namespace pmlc::conversion::stripe_to_spirv {

mlir::OwningModuleRef StripeLowerIntoSPIRV(mlir::ModuleOp workspace);
}  // namespace pmlc::conversion::stripe_to_spirv
