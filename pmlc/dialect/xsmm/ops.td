// Copyright 2019, Intel Corporation
#ifndef __PML_XSMM_OPS__
#define __PML_XSMM_OPS__

#ifndef __PML_UTIL_INTERFACES__
include "pmlc/util/interfaces.td"
#endif

#ifndef __PML_XSMM_INTERFACES__
include "pmlc/dialect/xsmm/interfaces.td"
#endif

def XSMMDialect : Dialect {
  let name = "xsmm";
  let cppNamespace = "pmlc::dialect::xsmm";
}

class XSMMBuilderImpl {
  code XSMMBuilderImpl_create = [{
    static Operation* create(OpBuilder* builder, Location loc, Type type, ArrayRef<Value*> operands) {
      OperationState state(loc, getOperationName());
      state.addOperands(operands);
      state.addAttribute("type", TypeAttr::get(type));
      state.addTypes(getResultType(operands));
      return builder->createOperation(state);
    }
  }];
}

class XSMMOp<string mnemonic, list<OpTrait> traits = []> :
    Op<XSMMDialect, mnemonic, traits>;

def DispatchOp : XSMMOp<"dispatch"> {
  let summary = "dispatch";
  let arguments = (ins AnyAttr:$value);

}

#endif // __PML_XSMM_OPS__
