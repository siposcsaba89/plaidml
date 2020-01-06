#include "pmlc/conversion/stripe_to_spirv/convert_stripe_to_spirv.h"
#include "pmlc/conversion/stripe_to_affine/convert_stripe_to_affine.h"

#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
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

// Standard to SPIR-V pass
struct StandardToSPIRVLoweringPass : public mlir::ModulePass<StandardToSPIRVLoweringPass> {
  void runOnModule() final;
};

void StandardToSPIRVLoweringPass::runOnModule() {
  ConversionTarget target(getContext());
  mlir::SPIRVTypeConverter typeConverter;
  mlir::OwningRewritePatternList patterns;

  target.addLegalDialect<mlir::spirv::SPIRVDialect>();
  target.addLegalOp<ModuleOp, mlir::ModuleTerminatorOp>();
  target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
    // FuncOp is legal only if types have been converted to SPIR-V types.
    // return typeConverter.isSignatureLegal(op.getType());
    return true;
  });

  populateStandardToSPIRVPatterns(&getContext(), typeConverter, patterns);

  auto module = getModule();
  if (failed(applyFullConversion(module, target, patterns))) signalPassFailure();
}
std::unique_ptr<mlir::Pass> createStandardToSPIRVLoweringPass() {
  return std::make_unique<StandardToSPIRVLoweringPass>();
}

// Standard+Loop to full Standard pass
struct LoopToStandardLoweringPass : public mlir::FunctionPass<LoopToStandardLoweringPass> {
  void runOnFunction() final;
};

void LoopToStandardLoweringPass::runOnFunction() {
  ConversionTarget target(getContext());
  target.addLegalDialect<mlir::StandardOpsDialect>();
  target.addLegalOp<ModuleOp, mlir::ModuleTerminatorOp>();

  mlir::TypeConverter typeConverter;
  target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
    // FuncOp is legal only if types have been converted to Std types.
    return typeConverter.isSignatureLegal(op.getType());
  });

  mlir::OwningRewritePatternList patterns;
  populateLoopToStdConversionPatterns(patterns, &getContext());

  auto function = getFunction();
  if (failed(applyFullConversion(function, target, patterns))) signalPassFailure();
}

std::unique_ptr<mlir::Pass> createLoopToStandardLoweringPass() {
  return std::make_unique<LoopToStandardLoweringPass>();
}

// Affine+Eltwise to Standard+Loop pass
struct AffineToStandardLoweringPass : public mlir::ModulePass<AffineToStandardLoweringPass> {
  void runOnModule() final;
};

void AffineToStandardLoweringPass::runOnModule() {
  ConversionTarget target(getContext());
  target.addLegalDialect<mlir::loop::LoopOpsDialect, mlir::StandardOpsDialect>();
  target.addLegalOp<ModuleOp, mlir::ModuleTerminatorOp>();

  mlir::TypeConverter typeConverter;
  target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
    // FuncOp is legal only if types have been converted to Std types.
    return typeConverter.isSignatureLegal(op.getType());
  });

  mlir::OwningRewritePatternList patterns;
  populateAffineToStdConversionPatterns(patterns, &getContext());

  auto module = getModule();
  if (failed(applyFullConversion(module, target, patterns))) signalPassFailure();
}

std::unique_ptr<mlir::Pass> createAffineToStandardLoweringPass() {
  return std::make_unique<AffineToStandardLoweringPass>();
}

OwningModuleRef StripeLowerIntoSPIRV(ModuleOp workspace) {
  OwningModuleRef module(llvm::cast<ModuleOp>(workspace.getOperation()->clone()));
  mlir::PassManager pm(workspace.getContext());
  IVLOG(3, "before:\n" << mlir::debugString(*module->getOperation()));

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createConvertStripeToAffinePass());

  pm.addPass(createAffineToStandardLoweringPass());
  pm.addPass(createLoopToStandardLoweringPass());
  // pm.addPass(createStandardToSPIRVLoweringPass());

  /*
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createLowerToCFGPass());
  pm.addPass(mlir::createConvertStandardToSPIRVPass());
  pm.addPass(mlir::createLegalizeStdOpsForSPIRVLoweringPass());
  */
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
