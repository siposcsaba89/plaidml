// Copyright 2019, Intel Corporation

#ifndef __PML_XSMM_INTERFACES__
#define __PML_XSMM_INTERFACES__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

def XSMMOpInterface : OpInterface<"XSMMOp">;

#endif // __PML_XSMM_INTERFACES__
