#include "pmlc/conversion/stripe_to_spirv/convert_stripe_to_spirv.h"
#include "pmlc/conversion/stripe_to_affine/convert_stripe_to_affine.h"

#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/LoopsToGPU/LoopsToGPUPass.h"
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

// test Conversion from Loop+standard to SPIR-V
class testLoopConversion final : public mlir::SPIRVOpLowering<mlir::loop::ForOp> {
 public:
  using SPIRVOpLowering<mlir::loop::ForOp>::SPIRVOpLowering;

  PatternMatchResult matchAndRewrite(mlir::loop::ForOp forop, mlir::ArrayRef<Value*> operands,
                                     ConversionPatternRewriter& rewriter) const override;
};

PatternMatchResult testLoopConversion::matchAndRewrite(mlir::loop::ForOp forop, mlir::ArrayRef<Value*> operands,
                                                       ConversionPatternRewriter& rewriter) const {
  auto valueAttr1 = rewriter.getIntegerAttr(rewriter.getIndexType(), 0);
  auto valueAttr2 = rewriter.getIntegerAttr(rewriter.getIndexType(), 3);
  auto constant_op1 = rewriter.create<mlir::ConstantOp>(forop.getLoc(), rewriter.getIndexType(), valueAttr1);
  auto constant_op2 = rewriter.create<mlir::ConstantOp>(forop.getLoc(), rewriter.getIndexType(), valueAttr2);

  auto induction_var = forop.getInductionVar();
  induction_var->replaceAllUsesWith(constant_op1.getResult());

  auto for_operation = forop.getOperation();
  auto first_inner_op = for_operation->getRegion(0).getBlocks().begin()->getOperations().begin();
  auto inner_forop = llvm::cast<mlir::loop::ForOp>(first_inner_op);
  if (!inner_forop) {
    throw std::runtime_error("First operation in outer for-loop is not ForOp!!");
  }
  auto inner_for_operation = inner_forop.getOperation();

  // mergeBlocks currently is not supported in ConversionPatternRewriter
  // rewriter.mergeBlocks(&newblock, rewriter.getBlock(), argValues);

  // replace block argument
  mlir::Block& inner_block = *(inner_for_operation->getRegion(0).getBlocks().begin());
  auto inner_block_arg = *inner_block.getArguments().begin();
  inner_block_arg->replaceAllUsesWith(constant_op2.getResult());

  auto outer_region = rewriter.getBlock()->getParent();
  rewriter.inlineRegionBefore(inner_for_operation->getRegion(0), *outer_region, outer_region->end());

  // merge block
  auto dest_block = rewriter.getBlock();
  auto it_dest = dest_block->end();
  it_dest--;
  dest_block->getOperations().splice(it_dest, inner_block.getOperations());
  inner_block.dropAllUses();
  inner_block.erase();

  // erase two layers of loop
  rewriter.eraseOp(forop);
  rewriter.eraseOp(inner_forop);
  return matchSuccess();
}

class testLoopTerminatorConversion final : public mlir::SPIRVOpLowering<mlir::loop::TerminatorOp> {
 public:
  using SPIRVOpLowering<mlir::loop::TerminatorOp>::SPIRVOpLowering;

  PatternMatchResult matchAndRewrite(mlir::loop::TerminatorOp terminatorop, mlir::ArrayRef<Value*> operands,
                                     ConversionPatternRewriter& rewriter) const override;
};

PatternMatchResult testLoopTerminatorConversion::matchAndRewrite(mlir::loop::TerminatorOp terminatorop,
                                                                 mlir::ArrayRef<Value*> operands,
                                                                 ConversionPatternRewriter& rewriter) const {
  rewriter.eraseOp(terminatorop);
  return matchSuccess();
}

void populateTestPatterns(MLIRContext* context, mlir::SPIRVTypeConverter& typeConverter,
                          mlir::OwningRewritePatternList& patterns) {
  patterns.insert<testLoopConversion, testLoopTerminatorConversion>(context, typeConverter);
}

struct TestSPIRVPass : public mlir::ModulePass<TestSPIRVPass> {
  void runOnModule() final;
};

void TestSPIRVPass::runOnModule() {
  ConversionTarget target(getContext());
  target.addLegalDialect<mlir::spirv::SPIRVDialect, mlir::StandardOpsDialect>();
  target.addLegalOp<ModuleOp, mlir::ModuleTerminatorOp>();

  // mlir::TypeConverter typeConverter;
  target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
    // FuncOp is legal only if types have been converted to Std types.
    // return typeConverter.isSignatureLegal(op.getType());
    return true;
  });

  mlir::SPIRVTypeConverter typeConverter;
  mlir::OwningRewritePatternList patterns;
  populateTestPatterns(&getContext(), typeConverter, patterns);

  auto module = getModule();
  if (failed(applyFullConversion(module, target, patterns))) signalPassFailure();
}

std::unique_ptr<mlir::Pass> createTestSPIRVPass() { return std::make_unique<TestSPIRVPass>(); }

OwningModuleRef StripeLowerIntoSPIRV(ModuleOp workspace) {
  OwningModuleRef module(llvm::cast<ModuleOp>(workspace.getOperation()->clone()));
  mlir::PassManager pm(workspace.getContext());
  IVLOG(3, "before:\n" << mlir::debugString(*module->getOperation()));

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createConvertStripeToAffinePass());

  pm.addPass(createAffineToStandardLoweringPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  pm.addPass(createTestSPIRVPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // pm.addPass(createLoopToStandardLoweringPass());
  // pm.addPass(createStandardToSPIRVLoweringPass());

  /*
  std::vector<int64_t> wg = {1, 1};
  std::vector<int64_t> gs = {3, 3};
  mlir::ArrayRef<int64_t> numWorkGroups(wg);
  mlir::ArrayRef<int64_t> workGroupSize(gs);
  pm.addPass(mlir::createLoopToGPUPass(numWorkGroups, workGroupSize));
  */

  // pm.addPass(mlir::createCanonicalizerPass());
  // pm.addPass(mlir::createCSEPass());

  // pm.addPass(mlir::createConvertGPUToSPIRVPass(workGroupSize));
  // pm.addPass(mlir::createLowerAffinePass());
  // pm.addPass(mlir::createLowerToCFGPass());
  // pm.addPass(mlir::createConvertStandardToSPIRVPass());
  // pm.addPass(mlir::createLegalizeStdOpsForSPIRVLoweringPass());

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
