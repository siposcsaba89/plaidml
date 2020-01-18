// Copyright 2019, Intel Corporation

#include "pmlc/compiler/compiler.h"

#include <unordered_map>
#include <utility>

#include <boost/filesystem.hpp>

#include "base/util/env.h"
#include "base/util/file.h"
#include "base/util/logging.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/tile_to_pxa/tile_to_pxa.h"
#include "pmlc/tools/pmlc-vulkan-runner/VulkanRuntime.h"
#include "pmlc/tools/pmlc-vulkan-runner/VulkanRuntimeTests.h"

using namespace mlir;  // NOLINT[build/namespaces]
using pmlc::conversion::tile_to_pxa::createLowerTileToPXAPass;

namespace fs = boost::filesystem;

namespace pmlc::compiler {

namespace {

template <typename T, int N>
struct StridedMemRefType {
  T* basePtr;
  T* data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

template <typename StreamType, typename T, int N>
void printMemRefMetaData(StreamType& os, StridedMemRefType<T, N>* memref) {  // NOLINT[runtime/references]
  static_assert(N > 0, "Expected N > 0");
  os << "Memref ptr: " << reinterpret_cast<void*>(memref);
  os << " base: " << reinterpret_cast<void*>(memref->data);
  os << " rank: " << N;
  os << " offset: " << memref->offset;
  os << " sizes: [";
  for (unsigned i = 0; i < N; ++i) {
    if (i) {
      os << ", ";
    }
    os << memref->sizes[i];
  }
  os << "] strides: [";
  for (unsigned i = 0; i < N; ++i) {
    if (i) {
      os << ", ";
    }
    os << memref->strides[i];
  }
  os << "]";
}

template <typename T, int N>
void printMemRef(StridedMemRefType<T, N>* memref) {
  static_assert(N > 0, "Expected N > 0");
  printMemRefMetaData(std::cout, memref);
  std::cout << std::endl;
}

class InjectTracingPass : public mlir::FunctionPass<InjectTracingPass> {
 public:
  void runOnFunction() override {
    auto funcOp = getFunction();
    auto moduleOp = funcOp.getParentOfType<ModuleOp>();

    OpBuilder builder(funcOp.getBody());
    for (auto arg : funcOp.getArguments()) {
      auto memRefType = arg->getType().cast<MemRefType>();
      SmallVector<int64_t, 2> shape(memRefType.getRank(), MemRefType::kDynamicSize);
      auto genericType = MemRefType::get(shape, memRefType.getElementType());
      auto printRef = getOrInsertPrint(moduleOp, genericType);
      auto castOp = builder.create<MemRefCastOp>(builder.getUnknownLoc(), genericType, arg);
      builder.create<CallOp>(builder.getUnknownLoc(), printRef, ArrayRef<mlir::Type>{}, castOp.getResult());
    }
  }

  static FlatSymbolRefAttr getOrInsertPrint(ModuleOp module, MemRefType memRefType) {
    auto* context = module.getContext();
    // TODO: select symbol name based on memRefType
    const char* symbol = "print_memref_2d_f32";
    if (module.lookupSymbol<FuncOp>(symbol)) {
      return SymbolRefAttr::get(symbol, context);
    }
    OpBuilder builder(context);
    builder.setInsertionPointToStart(module.getBody());
    auto funcType = mlir::FunctionType::get(memRefType, {}, context);
    builder.create<FuncOp>(module.getLoc(), symbol, funcType, ArrayRef<NamedAttribute>{});
    return SymbolRefAttr::get(symbol, context);
  }

  static std::unique_ptr<Pass> create() { return std::make_unique<InjectTracingPass>(); }
};

using MemRefTypes = std::vector<MemRefType>;

class ArgumentCollectorPass : public mlir::FunctionPass<ArgumentCollectorPass> {
 public:
  explicit ArgumentCollectorPass(MemRefTypes* into) : into(into) {}

  void runOnFunction() override {
    auto funcOp = getFunction();
    for (auto arg : funcOp.getArguments()) {
      into->emplace_back(arg->getType().cast<MemRefType>());
    }
  }

  static std::unique_ptr<Pass> create(MemRefTypes* into) { return std::make_unique<ArgumentCollectorPass>(into); }

 private:
  MemRefTypes* into;
};

}  // namespace

extern "C" void print_memref_2d_f32(StridedMemRefType<float, 2>* M) {  //
  printMemRef(M);
}

class MemRefDescriptor {
 private:
  struct Base {
    void* basePtr;
    void* data;
    int64_t offset;
  };

 public:
  MemRefDescriptor(void* data, MemRefType type) : memory(computeSize(type)) {
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto maybeStrides = getStridesAndOffset(type, strides, offset);
    if (failed(maybeStrides)) {
      throw std::runtime_error("unexpected non-strided memref");
    }
    auto base = reinterpret_cast<Base*>(memory.data());
    base->basePtr = data;
    base->data = data;
    base->offset = offset;
    auto var = reinterpret_cast<int64_t*>(memory.data() + sizeof(Base));
    auto rank = type.getRank();
    auto sizes = type.getShape();
    for (unsigned i = 0; i < rank; i++) {
      var[i] = sizes[i];
      var[i + rank] = strides[i];
    }
  }

  void* ptr() { return memory.data(); }

 private:
  static unsigned computeSize(MemRefType type) {
    return sizeof(void*) +                     // allocatedPtr
           sizeof(void*) +                     // alignedPtr
           sizeof(int64_t) +                   // offset
           sizeof(int64_t) * type.getRank() +  // sizes
           sizeof(int64_t) * type.getRank();   // strides
  }

  std::vector<char> memory;
};

void Executable::initialize() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  initializeLLVMPasses();
}

Executable::Executable(StringRef entry, StringRef target, ModuleOp programModule, ArrayRef<void*> bufptrs)
    : entry(entry), args(bufptrs.size()), ptrs(bufptrs.size()) {
  auto copy = cast<ModuleOp>(programModule.getOperation()->clone());
  OwningModuleRef module(copy);
  mlir::PassManager manager(module->getContext());

  auto shouldPrintBeforePass = [](auto, auto) { return false; };
  auto shouldPrintAfterPass = [](auto, auto) { return VLOG_IS_ON(3); };
  manager.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass, true, false, llvm::errs());

  manager.addNestedPass<FuncOp>(createCanonicalizerPass());
  manager.addNestedPass<FuncOp>(createCSEPass());

  manager.addPass(createLowerTileToPXAPass());
  manager.addNestedPass<FuncOp>(createCanonicalizerPass());
  manager.addNestedPass<FuncOp>(createCSEPass());

  std::vector<MemRefType> memRefTypes;
  manager.addPass(ArgumentCollectorPass::create(&memRefTypes));
  if (VLOG_IS_ON(6)) {
    manager.addPass(InjectTracingPass::create());
  }

  auto pipelineBuilder = resolveTarget(target);
  pipelineBuilder(&manager);

  if (failed(manager.run(*module))) {
    throw std::runtime_error("conversion to the LLVM IR dialect failed");
  }

  // clone the first spv module from the generated ModuleOp for serialization
  mlir::spirv::ModuleOp spv_ModuleOp;
  auto& blocks = module->getOperation()->getRegion(0).getBlocks();
  for (auto it = blocks.begin(); it != blocks.end(); it++) {
    auto& ops = it->getOperations();
    for (auto it = ops.begin(); it != ops.end(); it++) {
      mlir::Operation& m = *it;
      auto op_name = m.getName().getStringRef().str();
      if (!op_name.compare("spv.module")) {
        spv_ModuleOp = cast<mlir::spirv::ModuleOp>(m.clone());
        break;
      }
    }
  }
  spv_ModuleOp.dump();

  pmlc::vulkan::RuntimeTest rt;
  pmlc::vulkan::NumWorkGroups numWorkGroups;
  numWorkGroups.x = 3;
  numWorkGroups.y = 3;

  auto resOne = rt.createResourceVarFloat(0, 0, 3);
  auto resTwo = rt.createResourceVarFloat(0, 1, 3);
  auto resThree = rt.createResourceVarFloat(1, 0, 3);
  auto resFour = rt.createResourceVarFloat(1, 1, 3);

  if (failed(pmlc::vulkan::runOnVulkan(cast<mlir::ModuleOp>(spv_ModuleOp), rt.vars, numWorkGroups))) {
    std::cout << "runOnVulkan failed" << std::endl;
  } else {
    std::cout << "runOnVulkan success" << std::endl;
  }
}

Executable::~Executable() = default;

void Executable::invoke() {
  auto result = engine->invoke(entry, llvm::MutableArrayRef<void*>(args));
  if (result) {
    throw std::runtime_error("JIT invocation failed");
  }
}

}  // namespace pmlc::compiler
