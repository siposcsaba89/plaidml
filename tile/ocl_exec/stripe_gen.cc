#include "tile/ocl_exec/stripe_gen.h"

#include <stdio.h>

#include <fstream>
#include <istream>
#include <utility>
#include <vector>

#include "base/util/env.h"
#include "base/util/file.h"

#include "llvm/ADT/ArrayRef.h"

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/IR/Module.h"
#include "pmlc/conversion/stripe_to_spirv/convert_stripe_to_spirv.h"
#include "pmlc/dialect/stripe/mlir.h"
#include "pmlc/dialect/stripe/transcode.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/driver.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lang/semprinter.h"
#include "tile/lang/simplifier.h"
#include "tile/ocl_exec/emitsem.h"
#include "tile/targets/targets.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace lang;  // NOLINT

using pmlc::dialect::stripe::IntoMLIR;
using pmlc::conversion::stripe_to_spirv::StripeLowerIntoSPIRV;

lang::KernelList GenerateProgram(                    //
    const std::shared_ptr<stripe::Program>& stripe,  //
    const std::string& cfg_name,                     //
    const std::string& out_dir,                      //
    ConstBufferManager* const_bufs) {
  codegen::OptimizeOptions options;
  options.dump_passes = !out_dir.empty();
  options.dump_passes_proto = !out_dir.empty();
  options.dbg_dir = out_dir + "/passes";
  if (options.dump_passes) {
    IVLOG(2, "Write passes to: " << options.dbg_dir);
  }
  IVLOG(2, *stripe->entry);
  const auto& cfgs = targets::GetConfigs();
  const auto& cfg = cfgs.configs().at(cfg_name);
  const auto& stage = cfg.stages().at("default");
  codegen::CompilerState state(stripe);
  state.const_bufs = const_bufs;
  codegen::Optimize(&state, stage.passes(), options);
  IVLOG(2, *stripe->entry);

  if (vertexai::env::Get("PLAIDML_TEST_SPIRV") == "1") {
    auto env_cache = env::Get("PLAIDML_OPENCL_CACHE");
    mlir::MLIRContext ctx;
    mlir::spirv::ModuleOp module_SPIRV;

    if (vertexai::env::Get("USE_TEST_SPIRV_BINARY") == "1") {
      auto src_path = env_cache + "/out.spv";
      std::ifstream file(src_path, std::ios::in | std::ios::binary);
      std::vector<uint32_t> buffer;
      uint32_t t;
      while (file.read(reinterpret_cast<char*>(&t), sizeof(uint32_t))) {
        buffer.push_back(t);
      }
      llvm::ArrayRef<uint32_t> binary = llvm::ArrayRef<uint32_t>(buffer);
      // a few bugs need to be fixed for deserialize to run
      module_SPIRV = mlir::spirv::deserialize(binary, &ctx).getValue();
    } else {
      auto module_stripe = IntoMLIR(&ctx, *stripe);
      auto module = *StripeLowerIntoSPIRV(*module_stripe);
      module.dump();
      module_SPIRV = llvm::cast<mlir::spirv::ModuleOp>(module);
    }

    //  spirv::serialize
    mlir::SmallVector<uint32_t, 4> binaryout;
    mlir::spirv::serialize(module_SPIRV, binaryout);
    std::stringstream out;
    for (int i = 0; i < static_cast<int>(binaryout.size()); i++) {
      out.write(reinterpret_cast<char*>(&binaryout[i]), sizeof(binaryout[i]));
    }

    auto out_path = env_cache + "/spv_module_out.spv";
    WriteFile(out_path, out.str(), true);
  }

  codegen::SemtreeEmitter emit(codegen::AliasMap{}, 256);
  emit.Visit(*stripe->entry);
  // lang::Simplify(emit.kernels_.kernels);
  if (VLOG_IS_ON(2)) {
    for (const auto ki : emit.kernels_.kernels) {
      sem::Print p(*ki.kfunc);
      IVLOG(2, p.str());
      IVLOG(2, "gids = " << ki.gwork);
      IVLOG(2, "lids = " << ki.lwork);
    }
  }
  auto main = stripe->entry->SubBlock(0);
  AliasMap init_map;
  AliasMap prog_map(init_map, stripe->entry.get());
  AliasMap main_map(prog_map, main.get());
  for (const auto& ref : main->refs) {
    if (ref.dir != stripe::RefDir::None) {
      emit.kernels_.types[ref.from] = ref.interior_shape;
    } else {
      emit.kernels_.types["local_" + ref.into()] = ref.interior_shape;
    }
  }
  for (auto& ki : emit.kernels_.kernels) {
    for (auto& name : ki.inputs) {
      const auto& ai = main_map.at(name);
      name = ai.base_block == stripe->entry.get() ? ai.base_ref->into() : ("local_" + name);
    }
    for (auto& name : ki.outputs) {
      const auto& ai = main_map.at(name);
      name = ai.base_block == stripe->entry.get() ? ai.base_ref->into() : ("local_" + name);
    }
  }
  return emit.kernels_;
}

KernelList GenerateProgram(       //
    const RunInfo& runinfo,       //
    const std::string& cfg_name,  //
    const std::string& out_dir,   //
    ConstBufferManager* const_bufs) {
  IVLOG(2, runinfo.input_shapes);
  IVLOG(2, runinfo.output_shapes);
  IVLOG(2, to_string(runinfo.program));
  auto stripe = GenerateStripe(runinfo);
  return GenerateProgram(stripe, cfg_name, out_dir, const_bufs);
}

}  // End namespace codegen
}  // End namespace tile
}  // End namespace vertexai
