//===- LowerAnnotations.cpp - Lower Annotations -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerAnnotations pass.  This pass processes FIRRTL
// annotations, rewriting them, scattering them, and dealing with non-local
// annotations.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringExtras.h"

using namespace circt;
using namespace firrtl;
using namespace chirrtl;

namespace {

/// State threaded through functions for resolving and applying annotations.
struct ApplyState {
  using AddToWorklistFn = llvm::function_ref<void(DictionaryAttr)>;
  ApplyState(CircuitOp circuit, SymbolTable &symTbl,
             AddToWorklistFn addToWorklistFn)
      : circuit(circuit), symTbl(symTbl), addToWorklistFn(addToWorklistFn) {}

  CircuitOp circuit;
  SymbolTable &symTbl;
  AddToWorklistFn addToWorklistFn;

  ModuleNamespace &getNamespace(FModuleLike module) {
    auto &ptr = namespaces[module];
    if (!ptr)
      ptr = std::make_unique<ModuleNamespace>(module);
    return *ptr;
  }

private:
  DenseMap<Operation *, std::unique_ptr<ModuleNamespace>> namespaces;
};

} // namespace

/// Get annotations or an empty set of annotations.
static ArrayAttr getAnnotationsFrom(Operation *op) {
  if (auto annots = op->getAttrOfType<ArrayAttr>(getAnnotationAttrName()))
    return annots;
  return ArrayAttr::get(op->getContext(), {});
}

/// Construct the annotation array with a new thing appended.
static ArrayAttr appendArrayAttr(ArrayAttr array, Attribute a) {
  if (!array)
    return ArrayAttr::get(a.getContext(), ArrayRef<Attribute>{a});
  SmallVector<Attribute> old(array.begin(), array.end());
  old.push_back(a);
  return ArrayAttr::get(a.getContext(), old);
}

/// Update an ArrayAttribute by replacing one entry.
static ArrayAttr replaceArrayAttrElement(ArrayAttr array, size_t elem,
                                         Attribute newVal) {
  SmallVector<Attribute> old(array.begin(), array.end());
  old[elem] = newVal;
  return ArrayAttr::get(array.getContext(), old);
}

/// Apply a new annotation to a resolved target.  This handles ports,
/// aggregates, modules, wires, etc.
static void addAnnotation(AnnoTarget ref, unsigned fieldIdx,
                          ArrayRef<NamedAttribute> anno) {
  auto *context = ref.getOp()->getContext();
  DictionaryAttr annotation;
  if (fieldIdx) {
    SmallVector<NamedAttribute> annoField(anno.begin(), anno.end());
    annoField.emplace_back(
        StringAttr::get(context, "circt.fieldID"),
        IntegerAttr::get(IntegerType::get(context, 32, IntegerType::Signless),
                         fieldIdx));
    annotation = DictionaryAttr::get(context, annoField);
  } else {
    annotation = DictionaryAttr::get(context, anno);
  }

  if (ref.isa<OpAnnoTarget>()) {
    auto newAnno = appendArrayAttr(getAnnotationsFrom(ref.getOp()), annotation);
    ref.getOp()->setAttr(getAnnotationAttrName(), newAnno);
    return;
  }

  auto portRef = ref.cast<PortAnnoTarget>();
  auto portAnnoRaw = ref.getOp()->getAttr(getPortAnnotationAttrName());
  ArrayAttr portAnno = portAnnoRaw.dyn_cast_or_null<ArrayAttr>();
  if (!portAnno || portAnno.size() != getNumPorts(ref.getOp())) {
    SmallVector<Attribute> emptyPortAttr(
        getNumPorts(ref.getOp()),
        ArrayAttr::get(ref.getOp()->getContext(), {}));
    portAnno = ArrayAttr::get(ref.getOp()->getContext(), emptyPortAttr);
  }
  portAnno = replaceArrayAttrElement(
      portAnno, portRef.getPortNo(),
      appendArrayAttr(portAnno[portRef.getPortNo()].dyn_cast<ArrayAttr>(),
                      annotation));
  ref.getOp()->setAttr("portAnnotations", portAnno);
}

/// Make an anchor for a non-local annotation.  Use the expanded path to build
/// the module and name list in the anchor.
static FlatSymbolRefAttr buildNLA(AnnoPathValue target, ApplyState &state) {
  OpBuilder b(state.circuit.getBodyRegion());
  SmallVector<Attribute> insts;
  for (auto inst : target.instances)
    insts.push_back(OpAnnoTarget(inst).getNLAReference(
        state.getNamespace(inst->getParentOfType<FModuleLike>())));

  auto module = dyn_cast<FModuleLike>(target.ref.getOp());
  if (!module)
    module = target.ref.getOp()->getParentOfType<FModuleLike>();
  insts.push_back(target.ref.getNLAReference(state.getNamespace(module)));
  auto instAttr = ArrayAttr::get(state.circuit.getContext(), insts);
  auto nla = b.create<NonLocalAnchor>(state.circuit.getLoc(), "nla", instAttr);
  state.symTbl.insert(nla);
  return FlatSymbolRefAttr::get(nla);
}

/// Scatter breadcrumb annotations corresponding to non-local annotations
/// along the instance path.  Returns symbol name used to anchor annotations to
/// path.
// FIXME: uniq annotation chain links
static FlatSymbolRefAttr scatterNonLocalPath(AnnoPathValue target,
                                             ApplyState &state) {

  FlatSymbolRefAttr sym = buildNLA(target, state);

  NamedAttrList pathmetadata;
  pathmetadata.append("circt.nonlocal", sym);
  pathmetadata.append(
      "class", StringAttr::get(state.circuit.getContext(), "circt.nonlocal"));
  for (auto item : target.instances)
    addAnnotation(OpAnnoTarget(item), 0, pathmetadata);

  return sym;
}

//===----------------------------------------------------------------------===//
// Standard Utility Resolvers
//===----------------------------------------------------------------------===//

/// Always resolve to the circuit, ignoring the annotation.
static Optional<AnnoPathValue> noResolve(DictionaryAttr anno,
                                         ApplyState &state) {
  return AnnoPathValue(state.circuit);
}

/// Implementation of standard resolution.  First parses the target path, then
/// resolves it.
static Optional<AnnoPathValue> stdResolveImpl(StringRef rawPath,
                                              ApplyState &state) {
  auto pathStr = canonicalizeTarget(rawPath);
  StringRef path{pathStr};

  auto tokens = tokenizePath(path);
  if (!tokens) {
    state.circuit->emitError("Cannot tokenize annotation path ") << rawPath;
    return {};
  }

  return resolveEntities(*tokens, state.circuit, state.symTbl);
}

/// (SFC) FIRRTL SingleTargetAnnotation resolver.  Uses the 'target' field of
/// the annotation with standard parsing to resolve the path.  This requires
/// 'target' to exist and be normalized (per docs/FIRRTLAnnotations.md).
static Optional<AnnoPathValue> stdResolve(DictionaryAttr anno,
                                          ApplyState &state) {
  auto target = anno.getNamed("target");
  if (!target) {
    state.circuit.emitError("No target field in annotation ") << anno;
    return {};
  }
  if (!target->getValue().isa<StringAttr>()) {
    state.circuit.emitError(
        "Target field in annotation doesn't contain string ")
        << anno;
    return {};
  }
  return stdResolveImpl(target->getValue().cast<StringAttr>().getValue(),
                        state);
}

/// Resolves with target, if it exists.  If not, resolves to the circuit.
static Optional<AnnoPathValue> tryResolve(DictionaryAttr anno,
                                          ApplyState &state) {
  auto target = anno.getNamed("target");
  if (target)
    return stdResolveImpl(target->getValue().cast<StringAttr>().getValue(),
                          state);
  return AnnoPathValue(state.circuit);
}

//===----------------------------------------------------------------------===//
// Standard Utility Appliers
//===----------------------------------------------------------------------===//

/// An applier which puts the annotation on the target and drops the 'target'
/// field from the annotaiton.  Optionally handles non-local annotations.
static LogicalResult applyWithoutTargetImpl(AnnoPathValue target,
                                            DictionaryAttr anno,
                                            ApplyState &state,
                                            bool allowNonLocal) {
  if (!allowNonLocal && !target.isLocal())
    return failure();
  SmallVector<NamedAttribute> newAnnoAttrs;
  for (auto &na : anno) {
    if (na.getName().getValue() != "target") {
      newAnnoAttrs.push_back(na);
    } else if (!target.isLocal()) {
      auto sym = scatterNonLocalPath(target, state);
      newAnnoAttrs.push_back(
          {StringAttr::get(anno.getContext(), "circt.nonlocal"), sym});
    }
  }
  addAnnotation(target.ref, target.fieldIdx, newAnnoAttrs);
  return success();
}

/// An applier which puts the annotation on the target and drops the 'target'
/// field from the annotaiton.  Optionally handles non-local annotations.
/// Ensures the target resolves to an expected type of operation.
template <bool allowNonLocal, typename T, typename... Tr>
static LogicalResult applyWithoutTarget(AnnoPathValue target,
                                        DictionaryAttr anno,
                                        ApplyState &state) {
  if (!target.isOpOfType<T, Tr...>())
    return failure();
  return applyWithoutTargetImpl(target, anno, state, allowNonLocal);
}

/// An applier which puts the annotation on the target and drops the 'target'
/// field from the annotaiton.  Optionally handles non-local annotations.
template <bool allowNonLocal = false>
static LogicalResult applyWithoutTarget(AnnoPathValue target,
                                        DictionaryAttr anno,
                                        ApplyState &state) {
  return applyWithoutTargetImpl(target, anno, state, allowNonLocal);
}

//===----------------------------------------------------------------------===//
// Driving table
//===----------------------------------------------------------------------===//

namespace {
struct AnnoRecord {
  llvm::function_ref<Optional<AnnoPathValue>(DictionaryAttr, ApplyState &)>
      resolver;
  llvm::function_ref<LogicalResult(AnnoPathValue, DictionaryAttr, ApplyState &)>
      applier;
};
} // end anonymous namespace

static const llvm::StringMap<AnnoRecord> annotationRecords{{

    // Testing Annotation
    {"circt.test", {stdResolve, applyWithoutTarget<true>}},
    {"circt.testNT", {noResolve, applyWithoutTarget<>}},
    {"circt.missing", {tryResolve, applyWithoutTarget<>}}

}};

/// Lookup a record for a given annotation class.  Optionally, returns the
/// record for "circuit.missing" if the record doesn't exist.
static const AnnoRecord *getAnnotationHandler(StringRef annoStr,
                                              bool ignoreUnhandledAnno) {
  auto ii = annotationRecords.find(annoStr);
  if (ii != annotationRecords.end())
    return &ii->second;
  if (ignoreUnhandledAnno)
    return &annotationRecords.find("circt.missing")->second;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerAnnotationsPass
    : public LowerFIRRTLAnnotationsBase<LowerAnnotationsPass> {
  void runOnOperation() override;
  LogicalResult applyAnnotation(DictionaryAttr anno, ApplyState &state);

  bool ignoreUnhandledAnno = false;
  bool ignoreClasslessAnno = false;
  SmallVector<DictionaryAttr> worklistAttrs;
};
} // end anonymous namespace

LogicalResult LowerAnnotationsPass::applyAnnotation(DictionaryAttr anno,
                                                    ApplyState &state) {
  // Lookup the class
  StringRef annoClassVal;
  if (auto annoClass = anno.getNamed("class"))
    annoClassVal = annoClass->getValue().cast<StringAttr>().getValue();
  else if (ignoreClasslessAnno)
    annoClassVal = "circt.missing";
  else
    return state.circuit.emitError("Annotation without a class: ") << anno;

  // See if we handle the class
  auto record = getAnnotationHandler(annoClassVal, ignoreUnhandledAnno);
  if (!record)
    return state.circuit.emitWarning("Unhandled annotation: ") << anno;

  // Try to apply the annotation
  auto target = record->resolver(anno, state);
  if (!target)
    return state.circuit.emitError("Unable to resolve target of annotation: ")
           << anno;
  if (record->applier(*target, anno, state).failed())
    return state.circuit.emitError("Unable to apply annotation: ") << anno;
  return success();
}

// This is the main entrypoint for the lowering pass.
void LowerAnnotationsPass::runOnOperation() {
  CircuitOp circuit = getOperation();
  SymbolTable modules(circuit);
  // Grab the annotations.
  for (auto anno : circuit.annotations())
    worklistAttrs.push_back(anno.cast<DictionaryAttr>());
  // Clear the annotations.
  circuit.annotationsAttr(ArrayAttr::get(circuit.getContext(), {}));
  size_t numFailures = 0;
  ApplyState state{circuit, modules,
                   [&](DictionaryAttr ann) { worklistAttrs.push_back(ann); }

  };
  while (!worklistAttrs.empty()) {
    auto attr = worklistAttrs.pop_back_val();
    if (applyAnnotation(attr, state).failed())
      ++numFailures;
  }
  if (numFailures)
    signalPassFailure();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerFIRRTLAnnotationsPass(
    bool ignoreUnhandledAnnotations, bool ignoreClasslessAnnotations) {
  auto pass = std::make_unique<LowerAnnotationsPass>();
  pass->ignoreUnhandledAnno = ignoreUnhandledAnnotations;
  pass->ignoreClasslessAnno = ignoreClasslessAnnotations;
  return pass;
}
