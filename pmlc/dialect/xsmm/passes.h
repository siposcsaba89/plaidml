// Copyright 2020, Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class Pass;
}  // namespace mlir

namespace pmlc::dialect::xsmm {

std::unique_ptr<mlir::Pass> createLowerXSMMToStandardPass();

}  // namespace pmlc::dialect::xsmm
