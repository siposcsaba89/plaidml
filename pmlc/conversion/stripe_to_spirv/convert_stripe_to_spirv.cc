#include "pmlc/conversion/stripe_to_spirv/convert_stripe_to_spirv.h"
#include "pmlc/conversion/stripe_to_affine/convert_stripe_to_affine.h"

#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Translation.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/FormatVariadic.h"

#include "base/util/logging.h"
#include "pmlc/dialect/eltwise/dialect.h"
#include "pmlc/dialect/eltwise/ops.h"
#include "pmlc/dialect/eltwise/util.h"
#include "pmlc/dialect/stripe/affine_poly.h"
#include "pmlc/dialect/stripe/analysis.h"
#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/ops.h"
#include "pmlc/dialect/stripe/transcode.h"
#include "pmlc/dialect/stripe/util.h"
#include "pmlc/util/util.h"

using mlir::AbstractOperation;
using mlir::ArrayAttr;
using mlir::Block;
using mlir::ConversionPattern;
using mlir::ConversionPatternRewriter;
using mlir::ConversionTarget;
using mlir::FloatAttr;
using mlir::FuncOp;
using mlir::FunctionType;
using mlir::Identifier;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OwningModuleRef;
using mlir::PatternBenefit;
using mlir::PatternMatchResult;
using mlir::ReturnOp;
using mlir::UnknownLoc;
using mlir::Value;

namespace pmlc::conversion::stripe_to_spirv {

struct AffineToSPIRVLoweringPass : public mlir::ModulePass<AffineToSPIRVLoweringPass> {
  void runOnModule() final;
};

void AffineToSPIRVLoweringPass::runOnModule() {
  ConversionTarget target(getContext());
  target.addLegalDialect<mlir::spirv::SPIRVDialect>();
  target.addLegalOp<ModuleOp, mlir::ModuleTerminatorOp>();
  target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) { return true; });

  mlir::OwningRewritePatternList patterns;
  mlir::SPIRVTypeConverter typeconverter;
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  populateStandardToSPIRVPatterns(&getContext(), typeconverter, patterns);

  auto module = getModule();
  if (failed(applyFullConversion(module, target, patterns))) signalPassFailure();
}

std::unique_ptr<mlir::Pass> createAffineToSPIRVLoweringPass() { return std::make_unique<AffineToSPIRVLoweringPass>(); }

OwningModuleRef StripeLowerIntoSPIRV(ModuleOp workspace) {
  OwningModuleRef module(llvm::cast<ModuleOp>(workspace.getOperation()->clone()));
  mlir::PassManager pm(workspace.getContext());
  IVLOG(3, "before:\n" << mlir::debugString(*module->getOperation()));

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createConvertStripeToAffinePass());
  pm.addPass(createAffineToSPIRVLoweringPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  auto result = pm.run(*module);
  if (failed(result)) {
    IVLOG(1, "StripeLowerIntoSPIRV failed: " << mlir::debugString(*module->getOperation()));
    throw std::runtime_error("Lowering to SPIR-V dialect failure");
  }
  return module;
}
}  // namespace pmlc::conversion::stripe_to_spirv
