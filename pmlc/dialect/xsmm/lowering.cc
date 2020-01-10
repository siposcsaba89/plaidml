// Copyright 2020, Intel Corporation

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/DialectConversion.h"

#include "base/util/logging.h"
#include "pmlc/dialect/xsmm/dialect.h"
#include "pmlc/dialect/xsmm/ops.h"
#include "pmlc/dialect/xsmm/passes.h"
#include "pmlc/util/util.h"

namespace pmlc::dialect::xsmm {

namespace {

struct LoweringPass : public mlir::ModulePass<LoweringPass> {
  void runOnModule() override;
};

void LoweringPass::runOnModule() {
  // do work here
}

}  // namespace

std::unique_ptr<mlir::Pass> createLowerXSMMToStandardPass() { return std::make_unique<LoweringPass>(); }

static mlir::PassRegistration<LoweringPass> legalize_pass("xsmm-legalize-to-standard",
                                                          "match xsmm op pattern and convert to standard calls");

}  // namespace pmlc::dialect::xsmm
