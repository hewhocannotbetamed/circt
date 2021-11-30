//===- IMConstProp.cpp - Intermodule ConstProp and DCE ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/InstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-imconstprop"

using namespace circt;
using namespace firrtl;

/// Return true if this is either wire, register or similar(subfield and
/// subindex).
static bool isWireOrReg(Operation *op) {
  return isa<WireOp, RegResetOp, RegOp, SubfieldOp, SubindexOp>(op);
}

/// Return true if this is a wire or register we're allowed to delete.
static bool isDeletableWireOrReg(Operation *op) {
  return isWireOrReg(op) && !hasDontTouch(op);
}

/// Return true if a type is possible to track. Currently, we allow passive
/// types.
static bool isTrackableType(Type type) {
  return type.cast<FIRRTLType>().isPassive();
}

/// This function recursively apply `fn` to leaf ground types of `type`.
static void traverseFIRRTLType(Type type,
                               llvm::unique_function<void(FIRRTLType)> fn) {
  std::function<void(FIRRTLType)> recurse = [&](FIRRTLType type) {
    TypeSwitch<FIRRTLType>(type)
        .Case<BundleType>([&](BundleType bundle) {
          for (size_t i = 0, e = bundle.getNumElements(); i < e; ++i)
            recurse(bundle.getElementType(i));
        })
        .Case<FVectorType>([&](FVectorType vector) {
          for (size_t i = 0, e = vector.getNumElements(); i < e; ++i)
            recurse(vector.getElementType());
        })
        .Default([&](auto op) {
          assert(op.isGround() && "only ground types are expected here");
          fn(type.cast<FIRRTLType>());
        });
  };
  recurse(type.cast<FIRRTLType>());
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a single lattice value. A lattive value corresponds to
/// the various different states that a value in the SCCP dataflow analysis can
/// take. See 'Kind' below for more details on the different states a value can
/// take.
class LatticeValue {
  enum Kind {
    /// A value with a yet-to-be-determined value. This state may be changed to
    /// anything, it hasn't been processed by IMConstProp.
    Unknown,

    /// An FIRRTL 'invalidvalue' value, carrying the result of an
    /// InvalidValueOp.  Wires and other stateful values start out in this
    /// state.
    ///
    /// This is named "InvalidValue" instead of "Invalid" to avoid confusion
    /// about whether the lattice value is corrupted.  "InvalidValue" is a
    /// valid lattice state, and a can move up to Constant or Overdefined.
    InvalidValue,

    /// A value that is known to be a constant. This state may be changed to
    /// overdefined.
    Constant,

    /// A value that cannot statically be determined to be a constant. This
    /// state cannot be changed.
    Overdefined
  };

public:
  /// Initialize a lattice value with "Unknown".
  /*implicit*/ LatticeValue() : valueAndTag(nullptr, Kind::Unknown) {}
  /// Initialize a lattice value with a constant.
  /*implicit*/ LatticeValue(IntegerAttr attr)
      : valueAndTag(attr, Kind::Constant) {}

  /// Initialize a lattice value with an InvalidValue constant.
  /*implicit*/ LatticeValue(InvalidValueAttr attr)
      : valueAndTag(attr, Kind::InvalidValue) {}

  static LatticeValue getOverdefined() {
    LatticeValue result;
    result.markOverdefined();
    return result;
  }

  bool isUnknown() const { return valueAndTag.getInt() == Kind::Unknown; }
  bool isInvalidValue() const {
    return valueAndTag.getInt() == Kind::InvalidValue;
  }
  bool isConstant() const { return valueAndTag.getInt() == Kind::Constant; }
  bool isOverdefined() const {
    return valueAndTag.getInt() == Kind::Overdefined;
  }

  /// Mark the lattice value as overdefined.
  void markOverdefined() {
    valueAndTag.setPointerAndInt(nullptr, Kind::Overdefined);
  }

  void markInvalidValue(InvalidValueAttr value) {
    valueAndTag.setPointerAndInt(value, Kind::InvalidValue);
  }

  /// Mark the lattice value as constant.
  void markConstant(IntegerAttr value) {
    valueAndTag.setPointerAndInt(value, Kind::Constant);
  }

  /// If this lattice is constant or invalid value, return the attribute.
  /// Returns nullptr otherwise.
  Attribute getValue() const { return valueAndTag.getPointer(); }

  /// If this is in the constant state, return the IntegerAttr.
  IntegerAttr getConstant() const {
    assert(isConstant());
    return getValue().dyn_cast_or_null<IntegerAttr>();
  }

  /// Merge in the value of the 'rhs' lattice into this one. Returns true if the
  /// lattice value changed.
  bool mergeIn(LatticeValue rhs) {
    // If we are already overdefined, or rhs is unknown, there is nothing to do.
    if (isOverdefined() || rhs.isUnknown())
      return false;

    // If we are unknown, just take the value of rhs.
    if (isUnknown()) {
      valueAndTag = rhs.valueAndTag;
      return true;
    }

    // If the right side is InvalidValue then it won't contribute anything to
    // our state since we're either already InvalidValue or a constant here.
    if (rhs.isInvalidValue())
      return false;

    // If we are an InvalidValue, then upgrade to Constant or Overdefined.
    if (isInvalidValue()) {
      valueAndTag = rhs.valueAndTag;
      return true;
    }

    // Otherwise, if this value doesn't match rhs go straight to overdefined.
    // This happens when we merge "3" and "4" from two different instance sites
    // for example.
    if (valueAndTag != rhs.valueAndTag) {
      markOverdefined();
      return true;
    }
    return false;
  }

  bool operator==(const LatticeValue &other) const {
    return valueAndTag == other.valueAndTag;
  }
  bool operator!=(const LatticeValue &other) const {
    return valueAndTag != other.valueAndTag;
  }
  friend raw_ostream &operator<<(raw_ostream &os, const LatticeValue &dt);

private:
  /// The attribute value if this is a constant and the tag for the element
  /// kind.  The attribute is always an IntegerAttr.
  llvm::PointerIntPair<Attribute, 2, Kind> valueAndTag;
};

raw_ostream &operator<<(raw_ostream &os, const LatticeValue &lattice) {
  os << "<unknown: " << lattice.isUnknown()
     << ", invalid: " << lattice.isInvalidValue()
     << ", constant: " << lattice.isConstant()
     << ", over: " << lattice.isOverdefined() << ">";
  return os;
}

} // end anonymous namespace

// This class defines the data structure associated with lattice values.
// This means basically a pair of value and integer. The integer
// represents the relative offset to the value on leaf level of type.

class ValueAndLeafIndex {
public:
  ValueAndLeafIndex(Value value, unsigned leafId, bool isKnownRoot = false)
      : valueAndFlag(value, isKnownRoot), leafId(leafId) {}
  ValueAndLeafIndex() = default;
  Value getValue() const { return valueAndFlag.getPointer(); }
  unsigned getLeafId() const { return leafId; }

  // isKnownRoot is true if we already know that the value is a root value.
  // If this flag is true, we can skip the translation.
  bool isKnownRoot() const { return valueAndFlag.getInt(); }
  void setKnownRoot() { valueAndFlag.setInt(true); }

  friend raw_ostream &operator<<(raw_ostream &os, const ValueAndLeafIndex &dt);

private:
  // Pair of a value and flag.
  llvm::PointerIntPair<Value, 1, bool> valueAndFlag{};
  unsigned leafId{};
};

raw_ostream &operator<<(raw_ostream &os, const ValueAndLeafIndex &value) {
  os << "<" << value.getValue() << ", index=" << value.getLeafId()
     << ", isKnownRoot=" << value.isKnownRoot() << ">";
  return os;
}

template <>
struct DenseMapInfo<ValueAndLeafIndex> {
  static inline ValueAndLeafIndex getEmptyKey() {
    // We don't care `isKnownRoot` here because keys must be root.
    return {llvm::DenseMapInfo<Value>::getEmptyKey(),
            llvm::DenseMapInfo<unsigned>::getEmptyKey()};
  }
  static inline ValueAndLeafIndex getTombstoneKey() {
    // We don't care `isKnownRoot` here because keys must be root.
    return {llvm::DenseMapInfo<Value>::getTombstoneKey(),
            llvm::DenseMapInfo<unsigned>::getTombstoneKey()};
  }
  static unsigned getHashValue(ValueAndLeafIndex val) {
    assert(val.isKnownRoot() &&
           "All the values must be already known to be a root");
    return llvm::DenseMapInfo<std::pair<Value, unsigned>>::getHashValue(
        {val.getValue(), val.getLeafId()});
  }

  static bool isEqual(const ValueAndLeafIndex &LHS,
                      const ValueAndLeafIndex &RHS) {
    return LHS.getValue() == RHS.getValue() &&
           LHS.getLeafId() == RHS.getLeafId();
  }
};

namespace {
struct IMConstPropPass : public IMConstPropBase<IMConstPropPass> {
  void runOnOperation() override;
  void rewriteModuleBody(FModuleOp module);

  /// Returns true if the given block is executable.
  bool isBlockExecutable(Block *block) const {
    return executableBlocks.count(block);
  }

  bool isOverdefined(ValueAndLeafIndex value) const {
    auto it = find(value);
    return it != latticeValues.end() && it->second.isOverdefined();
  }

  // Value is regarded as overdefined if one of lattice values associated with
  // the value is overdefined.
  bool isOverdefined(Value value) const {
    auto [root, start, size] =
        getRootValueWithCorrespondingChildIndexRange(value);
    for (unsigned i = 0; i < size; i++) {
      if (isOverdefined({root, start + i, true}))
        return true;
    }
    return false;
  }

  /// Mark the given pair of value and leaf index as overdefined. This
  /// means that we cannot refine a specific constant for this value.
  void markOverdefined(ValueAndLeafIndex value) {
    value = translateToRootIndex(value);
    auto &entry = latticeValues[value];
    if (!entry.isOverdefined()) {
      entry.markOverdefined();
      changedLatticeValueWorklist.push_back(value);
    }
  }

  /// Mark the given value as overdefined. This means that we cannot
  /// refine a specific constant for this value. The value might be associated
  /// with some values if the value has a aggregate type so iterate over the
  /// corresponding range.
  void markOverdefined(Value value) {
    auto [root, start, size] =
        getRootValueWithCorrespondingChildIndexRange(value);
    for (unsigned int i = 0; i < size; i++)
      markOverdefined({root, start + i, /*isKnownRoot=*/true});
  }

  /// Merge information from the 'from' lattice value into value.  If it
  /// changes, then users of the value are added to the worklist for
  /// revisitation.
  void mergeLatticeValue(ValueAndLeafIndex value, LatticeValue &valueEntry,
                         LatticeValue source) {
    assert(value.isKnownRoot() && "value must be known to be root beforehand");
    if (!source.isOverdefined() && hasDontTouch(value.getValue()))
      source = LatticeValue::getOverdefined();
    if (valueEntry.mergeIn(source))
      changedLatticeValueWorklist.push_back(value);
  }

  /// Merge two values.
  void mergeLatticeValue(Value value, Value source) {
    auto [valueRoot, valueStart, valueSize] =
        getRootValueWithCorrespondingChildIndexRange(value);
    auto [sourceRoot, sourceStart, sourceSize] =
        getRootValueWithCorrespondingChildIndexRange(source);
    assert(valueSize == sourceSize && "range must match");
    for (unsigned i = 0; i < valueSize; i++)
      mergeLatticeValue({valueRoot, valueStart + i, /*isKnownRoot=*/true},
                        {sourceRoot, sourceStart + i, /*isKnownRoot=*/true});
  }

  void mergeLatticeValue(Value value, LatticeValue source) {
    // Don't even do a map lookup if from has no info in it.
    if (source.isUnknown())
      return;
    auto [valueRoot, valueStart, valueSize] =
        getRootValueWithCorrespondingChildIndexRange(value);
    for (unsigned i = 0; i < valueSize; i++)
      mergeLatticeValue({valueRoot, valueStart + i, /*isKnownRoot=*/true},
                        source);
  }

  void mergeLatticeValue(ValueAndLeafIndex value, LatticeValue source) {
    LLVM_DEBUG({
      llvm::dbgs() << "Lattice Merge Values: \n<" << value.getValue()
                   << ", index=" << value.getLeafId() << ">\n"
                   << " <= <" << source << ">\n";
    });

    // Don't even do a map lookup if from has no info in it.
    if (source.isUnknown())
      return;
    value = translateToRootIndex(value);
    LLVM_DEBUG({
      llvm::dbgs() << "Lattice Merge Values: \n<" << value.getValue()
                   << ", index=" << value.getLeafId() << ">\n"
                   << " <= <" << source << ">\n";
    });
    mergeLatticeValue(value, latticeValues[value], source);
  }

  void mergeLatticeValue(ValueAndLeafIndex result, ValueAndLeafIndex from) {
    LLVM_DEBUG({
      llvm::dbgs() << "Lattice Merge Values: \n<" << result.getValue()
                   << ", index=" << result.getLeafId() << ">\n"
                   << " <= <" << from.getValue()
                   << ", index=" << from.getLeafId() << ">\n";
    });
    // If 'from' hasn't been computed yet, then it is unknown, don't do
    // anything.
    auto it = find(from);
    if (it == latticeValues.end())
      return;
    mergeLatticeValue(result, it->second);
  }

  /// setLatticeValue - This is used when a new LatticeValue is computed for
  /// the result of the specified value that replaces any previous knowledge,
  /// e.g. because a fold() function on an op returned a new thing.  This should
  /// not be used on operations that have multiple contributors to it, e.g.
  /// wires or ports.
  // Don't even do a map lookup if from has no info in it.
  void setLatticeValue(ValueAndLeafIndex value, LatticeValue source) {
    if (source.isUnknown())
      return;

    value = translateToRootIndex(value);
    if (!source.isOverdefined() && hasDontTouch(value.getValue()))
      source = LatticeValue::getOverdefined();
    // If we've changed this value then revisit all the users.
    auto &valueEntry = latticeValues[value];
    if (valueEntry != source) {
      changedLatticeValueWorklist.push_back(value);
      valueEntry = source;
    }
  }

  /// Return the lattice value for the specified SSA value, extended to the
  /// width of the specified destType.  If allowTruncation is true, then this
  /// allows truncating the lattice value to the specified type.
  LatticeValue getExtendedLatticeValue(ValueAndLeafIndex value,
                                       FIRRTLType destType,
                                       bool allowTruncation = false);

  /// Mark the given block as executable.
  void markBlockExecutable(Block *block);
  void markWireOrUnresetableRegOp(Operation *wireOrReg);
  void markRegResetOp(RegResetOp regReset);
  void markRegOp(RegOp reg);
  void markMemOp(MemOp mem);
  void markSubindexOp(SubindexOp subindex);
  void markSubfieldOp(SubfieldOp subfield);
  void markInvalidValueOp(InvalidValueOp invalid);
  void markConstantOp(ConstantOp constant);
  void markSpecialConstantOp(SpecialConstantOp specialConstant);
  void markInstanceOp(InstanceOp instance);

  // Visit when some lattice value is updated. `changedValue` is that changed
  // value.
  void visitConnect(ConnectOp connect, ValueAndLeafIndex changedValue);
  void visitPartialConnect(PartialConnectOp connect,
                           ValueAndLeafIndex changedValue);
  void visitRegResetOp(RegResetOp connect, ValueAndLeafIndex changedValue);
  void visitSubindex(SubindexOp op);
  void visitSubfield(SubfieldOp op);
  void visitOperation(Operation *op, ValueAndLeafIndex changedValue);

  /// Translate the given value and leaf index into the root and leaf
  /// index for the root value.
  ValueAndLeafIndex
  translateToRootIndex(ValueAndLeafIndex valueAndLeafIndex) const {
    // If the value is already known to be root, just return.
    if (valueAndLeafIndex.isKnownRoot())
      return valueAndLeafIndex;
    auto it = valueToRootValueAndLeafIndex.find(valueAndLeafIndex.getValue());

    if (it != valueToRootValueAndLeafIndex.end()) {
      auto root = it->second;
      return {root.getValue(), root.getLeafId() + valueAndLeafIndex.getLeafId(),
              /*isKnownRoot=*/true};
    }

    // If the value is not registered, it must be root then just return.
    valueAndLeafIndex.setKnownRoot();
    return valueAndLeafIndex;
  }

  ValueAndLeafIndex translateToRootIndex(Value value) const {
    return translateToRootIndex({value, 0});
  }

  DenseMap<ValueAndLeafIndex, LatticeValue>::const_iterator
  find(ValueAndLeafIndex valueAndLeafIndex) const {
    return latticeValues.find(translateToRootIndex(valueAndLeafIndex));
  }

  /// Returns the tuple of root value, index, and size of the given value.
  /// This means that `valueAndLeafIndex` represent ground type elements of the
  /// root value in the range [index, index + size).
  std::tuple<Value, unsigned, unsigned>
  getRootValueWithCorrespondingChildIndexRange(
      ValueAndLeafIndex valueAndLeafIndex) const {
    auto rootValueAndLeafIndex = translateToRootIndex(valueAndLeafIndex);

    return {rootValueAndLeafIndex.getValue(), rootValueAndLeafIndex.getLeafId(),
            getNumberOfLeafs(valueAndLeafIndex.getValue().getType())};
  }

  std::tuple<Value, unsigned, unsigned>
  getRootValueWithCorrespondingChildIndexRange(Value value) const {
    return getRootValueWithCorrespondingChildIndexRange({value, 0});
  }

  /// Returns integer array which represent offsets of field index for the given
  /// bundle type.
  const SmallVector<unsigned, 4> &
  getBundleTypeToChildIndexOffset(BundleType bundle) {
    auto &entry = typeToChildIndexOffset[bundle];
    if (!entry.empty())
      return entry;
    unsigned commulativeSumNumberOfLeafs = 0;
    for (size_t i = 0, e = bundle.getNumElements(); i < e; ++i) {
      entry.push_back(commulativeSumNumberOfLeafs);
      commulativeSumNumberOfLeafs += getNumberOfLeafs(bundle.getElementType(i));
    }
    return entry;
  }

  /// Returns an offset at leaf level for given bundle/vector type and index.
  unsigned getChildIndexOffset(Type type, unsigned index) {
    return TypeSwitch<FIRRTLType, unsigned>(type.cast<FIRRTLType>())
        .Case<BundleType>([&](BundleType bundle) {
          auto offset = getBundleTypeToChildIndexOffset(bundle);
          return offset[index];
        })
        .Case<FVectorType>([&](FVectorType vector) {
          return getNumberOfLeafs(vector.getElementType()) * index;
        })
        .Default([&](auto op) {
          llvm_unreachable("bundle and vector types are expected");
          return 0;
        });
  }

  /// Returns the number of ground types in `type`. Results are cached in
  /// `typeToNumberOfGroundTypes`.
  unsigned getNumberOfLeafs(Type type) const {
    auto &entry = typeToNumberOfGroundTypes[type];
    if (entry)
      return entry;
    TypeSwitch<FIRRTLType>(type.cast<FIRRTLType>())
        .Case<BundleType>([&](BundleType bundle) {
          for (size_t i = 0, e = bundle.getNumElements(); i < e; ++i)
            entry += getNumberOfLeafs(bundle.getElementType(i));
        })
        .Case<FVectorType>([&](FVectorType vector) {
          entry += getNumberOfLeafs(vector.getElementType()) *
                   vector.getNumElements();
        })
        .Default([&](auto op) { entry += 1; });
    return entry;
  }

  /// Return all ground types which `type` have.
  const SmallVector<Type, 4> &getLeafGroundTypes(Type type) {
    auto &entry = typeToLeafGroundTypes[type];
    if (!entry.empty())
      return entry;
    auto fn = [&entry](Type type) { entry.push_back(type); };
    traverseFIRRTLType(type, fn);
    return entry;
  }

private:
  /// This is the current instance graph for the Circuit.
  InstanceGraph *instanceGraph = nullptr;

  /// This keeps track of the current state of each tracked value.
  DenseMap<ValueAndLeafIndex, LatticeValue> latticeValues;

  /// A map from a value (subindex and subfield) to its root and leaf index.
  DenseMap<Value, ValueAndLeafIndex> valueToRootValueAndLeafIndex;

  /// A map from a type to its leaf ground types.
  DenseMap<Type, SmallVector<Type, 4>> typeToLeafGroundTypes;

  /// An inverse map of `valueToRootValueAndLeafIndex` to visit all the users of
  /// children.
  DenseMap<ValueAndLeafIndex, SmallVector<Value, 4>> rootToChildrenAccess;

  /// A map from a bundle type to its offsets of index.
  DenseMap<Type, SmallVector<unsigned, 4>> typeToChildIndexOffset;

  /// A cache to compute the number of ground types.
  mutable DenseMap<Type, unsigned> typeToNumberOfGroundTypes;

  /// The set of blocks that are known to execute, or are intrinsically
  /// live.
  SmallPtrSet<Block *, 16> executableBlocks;

  /// A worklist containing blocks that need to be processed.
  SmallVector<Block *, 64> blockWorklist;

  /// A worklist of values whose LatticeValue recently changed, indicating the
  /// users need to be reprocessed.
  SmallVector<ValueAndLeafIndex, 64> changedLatticeValueWorklist;

  /// This keeps track of users the instance results that correspond to output
  /// ports.
  DenseMap<BlockArgument, llvm::TinyPtrVector<Value>>
      resultPortToInstanceResultMapping;
};
} // end anonymous namespace

// TODO: handle annotations: [[OptimizableExtModuleAnnotation]]
void IMConstPropPass::runOnOperation() {
  auto circuit = getOperation();

  instanceGraph = &getAnalysis<InstanceGraph>();

  // If the top level module is an external module, mark the input ports
  // overdefined.
  if (auto module = dyn_cast<FModuleOp>(circuit.getMainModule())) {
    markBlockExecutable(module.getBody());
    for (auto port : module.getBody()->getArguments())
      markOverdefined(port);
  } else {
    // Otherwise, mark all module ports as being overdefined.
    for (auto &circuitBodyOp : circuit.getBody()->getOperations()) {
      if (auto module = dyn_cast<FModuleOp>(circuitBodyOp)) {
        markBlockExecutable(module.getBody());
        for (auto port : module.getBody()->getArguments())
          markOverdefined(port);
      }
    }
  }

  // If a value changed lattice state then reprocess any of its users.
  while (!changedLatticeValueWorklist.empty()) {
    auto changedValue = changedLatticeValueWorklist.pop_back_val();
    LLVM_DEBUG(llvm::dbgs()
                   << "Lattice Worklist pop: <" << changedValue.getValue()
                   << ", index=" << changedValue.getLeafId()
                   << "> = " << latticeValues[changedValue] << "\n";);

    for (Operation *user : changedValue.getValue().getUsers()) {
      if (isBlockExecutable(user->getBlock())) {
        LLVM_DEBUG(llvm::dbgs() << "Dep direct<" << changedValue.getValue()
                                << ", index=" << changedValue.getLeafId()
                                << "> = " << *user << "\n";);
        visitOperation(user, changedValue);
      }
    }

    auto changedValueAll = rootToChildrenAccess[changedValue];
    for (auto changedVal : changedValueAll)
      for (Operation *user : changedVal.getUsers()) {
        if (isBlockExecutable(user->getBlock())) {
          LLVM_DEBUG(llvm::dbgs() << "Dep indirect <" << changedValue.getValue()
                                  << ", index=" << changedValue.getLeafId()
                                  << "> = " << *user << "\n";);
          visitOperation(user, changedValue);
        }
      }
  }

  // Rewrite any constants in the modules.
  mlir::parallelForEach(circuit.getContext(),
                        circuit.getBody()->getOps<FModuleOp>(),
                        [&](auto op) { rewriteModuleBody(op); });

  // Clean up our state for next time.
  instanceGraph = nullptr;
  latticeValues.clear();
  executableBlocks.clear();
  resultPortToInstanceResultMapping.clear();
}

/// Return the lattice value for the specified SSA value, extended to the width
/// of the specified destType.  If allowTruncation is true, then this allows
/// truncating the lattice value to the specified type.
LatticeValue IMConstPropPass::getExtendedLatticeValue(ValueAndLeafIndex value,
                                                      FIRRTLType destType,
                                                      bool allowTruncation) {
  // If 'value' hasn't been computed yet, then it is unknown.
  auto it = find(value);
  if (it == latticeValues.end())
    return LatticeValue();

  auto result = it->second;
  // Unknown/overdefined stay whatever they are.
  if (result.isUnknown() || result.isOverdefined())
    return result;
  // InvalidValue gets wider.
  if (result.isInvalidValue())
    return InvalidValueAttr::get(destType);

  auto constant = result.getConstant();

  // If this is a BoolAttr then we are dealing with a special constant.
  if (auto boolAttr = constant.dyn_cast<BoolAttr>()) {
    // No extOrTrunc necessary for clock or reset types.
    return LatticeValue(boolAttr);
  }

  // If destType is wider than the source constant type, extend it.
  auto resultConstant = result.getConstant().getAPSInt();
  auto destWidth = destType.getBitWidthOrSentinel();
  if (destWidth == -1) // We don't support unknown width FIRRTL.
    return LatticeValue::getOverdefined();
  if (resultConstant.getBitWidth() == (unsigned)destWidth)
    return result; // Already the right width, we're done.

  // Otherwise, extend the constant using the signedness of the source.
  resultConstant = resultConstant.extOrTrunc(destWidth);
  return LatticeValue(IntegerAttr::get(destType.getContext(), resultConstant));
}

/// Mark a block executable if it isn't already.  This does an initial scan of
/// the block, processing nullary operations like wires, instances, and
/// constants that only get processed once.
void IMConstPropPass::markBlockExecutable(Block *block) {
  if (!executableBlocks.insert(block).second)
    return; // Already executable.

  for (auto &op : *block) {

    // Handle each of the special operations in the firrtl dialect.
    if (isa<WireOp>(op) || isa<RegOp>(op))
      markWireOrUnresetableRegOp(&op);
    else if (auto constant = dyn_cast<ConstantOp>(op))
      markConstantOp(constant);
    else if (auto specialConstant = dyn_cast<SpecialConstantOp>(op))
      markSpecialConstantOp(specialConstant);
    else if (auto invalid = dyn_cast<InvalidValueOp>(op))
      markInvalidValueOp(invalid);
    else if (auto instance = dyn_cast<InstanceOp>(op))
      markInstanceOp(instance);
    else if (auto regReset = dyn_cast<RegResetOp>(op))
      markRegResetOp(regReset);
    else if (auto mem = dyn_cast<MemOp>(op))
      markMemOp(mem);
    else if (auto subindex = dyn_cast<SubindexOp>(op))
      markSubindexOp(subindex);
    else if (auto subfield = dyn_cast<SubfieldOp>(op))
      markSubfieldOp(subfield);
  }
}

void IMConstPropPass::markWireOrUnresetableRegOp(Operation *wireOrReg) {
  // If the wire/reg has a non-ground type, then it is too complex for us
  // to handle, mark it as overdefined.
  auto resultValue = wireOrReg->getResult(0);
  if (!isTrackableType(resultValue.getType()))
    return markOverdefined(resultValue);

  // Otherwise, this starts out as InvalidValue and is upgraded by
  // connects.
  auto &destTypes = getLeafGroundTypes(resultValue.getType());
  for (unsigned i = 0, e = destTypes.size(); i < e; i++) {
    auto destType = destTypes[i].cast<FIRRTLType>();
    mergeLatticeValue({resultValue, i, /*isKnownRoot=*/true},
                      InvalidValueAttr::get(destType));
  }
}

void IMConstPropPass::markRegResetOp(RegResetOp regReset) {
  // If the reg has a passive type, then it is too complex for us to
  // handle, mark it as overdefined.
  if (!isTrackableType(regReset.getType()))
    return markOverdefined(regReset);

  // The reset value may be known - if so, merge it in.
  auto &destTypes = getLeafGroundTypes(regReset.getType());

  // Iterate over each ground type and merge lattice values separatetly.
  for (unsigned i = 0, e = destTypes.size(); i < e; i++) {
    auto destType = destTypes[i].cast<FIRRTLType>();
    auto srcValue =
        getExtendedLatticeValue({regReset.resetValue(), i}, destType,
                                /*allowTruncation=*/true);
    mergeLatticeValue({regReset, i, /*isKnownRoot=*/true}, srcValue);
  }
}

void IMConstPropPass::visitRegResetOp(RegResetOp regReset,
                                      ValueAndLeafIndex changedValue) {
  auto [srcRoot, srcStart, srcSize] =
      getRootValueWithCorrespondingChildIndexRange(regReset.resetValue());

  // If source is not changed value, we don't have to process this operation.
  if (srcRoot != changedValue.getValue())
    return;

  // If chandedValue is not included in the range of the source value, just
  // skip.
  if (srcStart > changedValue.getLeafId() ||
      srcStart + srcSize <= changedValue.getLeafId())
    return;

  unsigned index = changedValue.getLeafId() - srcStart;

  auto &destTypes = getLeafGroundTypes(regReset.getType());
  auto srcValue = getExtendedLatticeValue({regReset.resetValue(), index},
                                          destTypes[index].cast<FIRRTLType>(),
                                          /*allowTruncation=*/true);
  mergeLatticeValue({regReset, index, /*isKnownRoot=*/true}, srcValue);
}

void IMConstPropPass::markMemOp(MemOp mem) {
  for (auto result : mem.getResults())
    markOverdefined(result);
}

void IMConstPropPass::markSubindexOp(SubindexOp subindex) {
  auto offset =
      getChildIndexOffset(subindex.input().getType(), subindex.index());

  auto rootAndIndex = translateToRootIndex({subindex.input(), offset});

  // Register the current position to `valueToRootValueAndLeafIndex`.
  valueToRootValueAndLeafIndex[subindex] = rootAndIndex;

  for (unsigned i = 0; i < getNumberOfLeafs(subindex.getType()); i++)
    rootToChildrenAccess[{rootAndIndex.getValue(), rootAndIndex.getLeafId() + i,
                          /*isKnownRoot=*/
                          true}]
        .push_back(subindex);
}

void IMConstPropPass::markSubfieldOp(SubfieldOp subfield) {
  auto offset =
      getChildIndexOffset(subfield.input().getType(), subfield.fieldIndex());

  auto rootAndIndex = translateToRootIndex({subfield.input(), offset});

  // Register the current position to `valueToRootValueAndLeafIndex`.
  valueToRootValueAndLeafIndex[subfield] = rootAndIndex;

  for (unsigned i = 0; i < getNumberOfLeafs(subfield.getType()); i++)
    rootToChildrenAccess[{rootAndIndex.getValue(), rootAndIndex.getLeafId() + i,
                          /*isKnownRoot=*/
                          true}]
        .push_back(subfield);
}

void IMConstPropPass::markConstantOp(ConstantOp constant) {
  mergeLatticeValue({constant, 0, /*isKnownRoot=*/true},
                    LatticeValue(constant.valueAttr()));
}

void IMConstPropPass::markSpecialConstantOp(SpecialConstantOp specialConstant) {
  mergeLatticeValue({specialConstant, 0, /*isKnownRoot=*/true},
                    LatticeValue(specialConstant.valueAttr()));
}

void IMConstPropPass::markInvalidValueOp(InvalidValueOp invalid) {
  mergeLatticeValue({invalid, 0, /*isKnownRoot=*/true},
                    InvalidValueAttr::get(invalid.getType()));
}

/// Instances have no operands, so they are visited exactly once when their
/// enclosing block is marked live.  This sets up the def-use edges for ports.
void IMConstPropPass::markInstanceOp(InstanceOp instance) {
  // Get the module being reference or a null pointer if this is an extmodule.
  auto module = instanceGraph->getReferencedModule(instance);

  // If this is an extmodule, just remember that any results and inouts are
  // overdefined.
  if (auto extModule = dyn_cast<FExtModuleOp>(module)) {
    for (size_t resultNo = 0, e = instance.getNumResults(); resultNo != e;
         ++resultNo) {
      auto portVal = instance.getResult(resultNo);
      // If this is an input to the extmodule, we can ignore it.
      if (extModule.getPortDirection(resultNo) == Direction::In)
        continue;

      // Otherwise this is a result from it or an inout, mark it as overdefined.
      markOverdefined(portVal);
    }
    return;
  }

  // Otherwise this is a defined module.
  auto fModule = cast<FModuleOp>(module);
  markBlockExecutable(fModule.getBody());

  // Ok, it is a normal internal module reference.  Populate
  // resultPortToInstanceResultMapping, and forward any already-computed values.
  for (size_t resultNo = 0, e = instance.getNumResults(); resultNo != e;
       ++resultNo) {
    auto instancePortVal = instance.getResult(resultNo);
    // If this is an input to the instance, it will
    // get handled when any connects to it are processed.
    if (fModule.getPortDirection(resultNo) == Direction::In)
      continue;
    // We only support simple values so far.
    if (!instancePortVal.getType().cast<FIRRTLType>().isGround()) {
      // TODO: Add field sensitivity.
      markOverdefined(instancePortVal);
      continue;
    }

    // Otherwise we have a result from the instance.  We need to forward results
    // from the body to this instance result's SSA value, so remember it.
    BlockArgument modulePortVal = fModule.getArgument(resultNo);

    // Mark don't touch results as overdefined
    if (hasDontTouch(modulePortVal))
      markOverdefined(modulePortVal);

    resultPortToInstanceResultMapping[modulePortVal].push_back(instancePortVal);

    // If there is already a value known for modulePortVal make sure to forward
    // it here.
    mergeLatticeValue(instancePortVal, modulePortVal);
  }
}

// We merge the value from the RHS into the value of the LHS.
void IMConstPropPass::visitConnect(ConnectOp connect,
                                   ValueAndLeafIndex changedValue) {
  auto destType = connect.dest().getType().cast<FIRRTLType>().getPassiveType();

  // TODO: Generalize to subaccesses etc when we have a field sensitive
  // model.
  if (!isTrackableType(destType)) {
    connect.emitError("non-ground type connect unhandled by IMConstProp");
    return;
  }

  auto [srcRoot, srcStart, srcSize] =
      getRootValueWithCorrespondingChildIndexRange(connect.src());
  // If source is not changed value, we don't have to process this operation.
  if (srcRoot != changedValue.getValue())
    return;

  // If chandedValue is not included in the range of the source value, just
  // skip.
  if (srcStart > changedValue.getLeafId() ||
      srcStart + srcSize <= changedValue.getLeafId())
    return;

  unsigned index = changedValue.getLeafId() - srcStart;

  // Handle implicit extensions.
  auto srcValue = getExtendedLatticeValue(changedValue, destType);
  if (srcValue.isUnknown())
    return;

  // Driving result ports propagates the value to each instance using the
  // module.
  if (auto blockArg = connect.dest().dyn_cast<BlockArgument>()) {
    if (!AnnotationSet::get(blockArg).hasDontTouch())
      for (auto userOfResultPort : resultPortToInstanceResultMapping[blockArg])
        mergeLatticeValue({userOfResultPort, index, true}, srcValue);
    // Output ports are wire-like and may have users.
    mergeLatticeValue({connect.dest(), index}, srcValue);
    return;
  }

  auto dest = connect.dest().cast<mlir::OpResult>();

  // For wires and registers, we drive the value of the wire itself, which
  // automatically propagates to users.
  if (isWireOrReg(dest.getOwner()))
    return mergeLatticeValue({connect.dest(), index}, srcValue);

  // Driving an instance argument port drives the corresponding argument of the
  // referenced module.
  if (auto instance = dest.getDefiningOp<InstanceOp>()) {
    // Update the dest, when its an instance op.
    mergeLatticeValue({connect.dest(), index}, srcValue);
    auto module =
        dyn_cast<FModuleOp>(instanceGraph->getReferencedModule(instance));
    if (!module)
      return;

    BlockArgument modulePortVal = module.getArgument(dest.getResultNumber());
    return mergeLatticeValue({modulePortVal, index}, srcValue);
  }

  // Driving a memory result is ignored because these are always treated as
  // overdefined.
  if (auto subfield = dest.getDefiningOp<SubfieldOp>()) {
    if (subfield.getOperand().getDefiningOp<MemOp>())
      return;
    return mergeLatticeValue({subfield, index}, srcValue);
  }

  if (auto subindex = dest.getDefiningOp<SubindexOp>())
    return mergeLatticeValue({subindex, index}, srcValue);

  connect.emitError("connect unhandled by IMConstProp")
          .attachNote(connect.dest().getLoc())
      << "connect destination is here";
}

void IMConstPropPass::visitPartialConnect(PartialConnectOp partialConnect,
                                          ValueAndLeafIndex leafIndex) {
  partialConnect.emitError("IMConstProp cannot handle partial connect");
}

void IMConstPropPass::visitSubindex(SubindexOp subindex) {
  // We can just skip subindex op because lattice values are shared.
  return;
}

void IMConstPropPass::visitSubfield(SubfieldOp subindex) {
  // We can just skip subfield op because lattice values are shared.
  return;
}

/// This method is invoked when an operand of the specified op changes its
/// lattice value state and when the block containing the operation is first
/// noticed as being alive.
///
/// This should update the lattice value state for any result values.
///
void IMConstPropPass::visitOperation(Operation *op,
                                     ValueAndLeafIndex changedValue) {
  // If this is a operation with special handling, handle it specially.
  if (auto connectOp = dyn_cast<ConnectOp>(op))
    return visitConnect(connectOp, changedValue);
  if (auto partialConnectOp = dyn_cast<PartialConnectOp>(op))
    return visitPartialConnect(partialConnectOp, changedValue);
  if (auto regResetOp = dyn_cast<RegResetOp>(op))
    return visitRegResetOp(regResetOp, changedValue);
  if (auto subindexOp = dyn_cast<SubindexOp>(op))
    return visitSubindex(subindexOp);
  if (auto subfield = dyn_cast<SubfieldOp>(op))
    return visitSubfield(subfield);

  // The clock operand of regop changing doesn't change its result value.
  if (isa<RegOp>(op))
    return;
  // TODO: Handle 'when' operations.

  // If all of the results of this operation are already overdefined (or if
  // there are no results) then bail out early: we've converged.
  auto isOverdefinedFn = [&](Value value) { return isOverdefined(value); };
  if (llvm::all_of(op->getResults(), isOverdefinedFn))
    return;

  // Collect all of the constant operands feeding into this operation. If any
  // are not ready to be resolved, bail out and wait for them to resolve.
  SmallVector<Attribute, 8> operandConstants;
  operandConstants.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    // TODO: All operands might be guaranteed to have ground types
    // already.
    if (!operand.getType().cast<FIRRTLType>().isGround())
      return;
    auto &operandLattice = latticeValues[translateToRootIndex(operand)];

    // If the operand is an unknown value, then we generally don't want to
    // process it - we want to wait until the value is resolved to by the SCCP
    // algorithm.  However, some operations are defined on partial operations,
    // including firrtl.mux in particular.  We resolve these eagerly because the
    // fold hooks know how to deal with them, and they often form cycles.
    if (operandLattice.isUnknown()) {
      if (!isa<MuxPrimOp>(op))
        return;
      operandConstants.push_back(InvalidValueAttr::get(operand.getType()));
      continue;
    }

    // Otherwise, it must be constant, invalid, or overdefined.  Translate them
    // into attributes that the fold hook can look at.
    if (operandLattice.isConstant() || operandLattice.isInvalidValue())
      operandConstants.push_back(operandLattice.getValue());
    else
      operandConstants.push_back({});
  }

  // Simulate the result of folding this operation to a constant. If folding
  // fails or was an in-place fold, mark the results as overdefined.
  SmallVector<OpFoldResult, 8> foldResults;
  foldResults.reserve(op->getNumResults());
  if (failed(op->fold(operandConstants, foldResults))) {
    for (auto value : op->getResults())
      markOverdefined(value);
    return;
  }

  // Fold functions in general are allowed to do in-place updates, but FIRRTL
  // does not do this and supporting it costs more.
  assert(!foldResults.empty() &&
         "FIRRTL fold functions shouldn't do in-place updates!");

  // Merge the fold results into the lattice for this operation.
  assert(foldResults.size() == op->getNumResults() && "invalid result size");
  for (unsigned i = 0, e = foldResults.size(); i != e; ++i) {
    // Merge in the result of the fold, either a constant or a value.
    LatticeValue resultLattice;
    OpFoldResult foldResult = foldResults[i];
    if (Attribute foldAttr = foldResult.dyn_cast<Attribute>()) {
      if (auto intAttr = foldAttr.dyn_cast<IntegerAttr>())
        resultLattice = LatticeValue(intAttr);
      else if (auto invalidValueAttr = foldAttr.dyn_cast<InvalidValueAttr>())
        resultLattice = invalidValueAttr;
      else // Treat non integer constants as overdefined.
        resultLattice = LatticeValue::getOverdefined();
    } else { // Folding to an operand results in its value.
      resultLattice =
          latticeValues[{foldResult.get<Value>(), 0, /*isKnownRoot=*/true}];
    }

    // We do not "merge" the lattice value in, we set it.  This is because the
    // fold functions can produce different values over time, e.g. in the
    // presence of InvalidValue operands that get resolved to other constants.
    setLatticeValue({op->getResult(i), 0, /*isKnownRoot=*/true}, resultLattice);
  }
}

void IMConstPropPass::rewriteModuleBody(FModuleOp module) {
  auto *body = module.getBody();
  // If a module is unreachable, just ignore it.
  if (!executableBlocks.count(body))
    return;

  auto builder = OpBuilder::atBlockBegin(body);

  // If the lattice value for the specified value is a constant or
  // InvalidValue, update it and return true.  Otherwise return false.
  auto replaceValueIfPossible = [&](Value value) -> bool {
    // TODO: We don't allow to replace non-ground type values for now.
    if (!value.getType().cast<FIRRTLType>().isGround())
      return false;
    auto it = latticeValues.find({value, 0, /*isKnownRoot=*/true});
    if (it == latticeValues.end() || it->second.isOverdefined() ||
        it->second.isUnknown())
      return false;

    // TODO: Unique constants into the entry block of the module.
    Attribute constantValue = it->second.getValue();
    auto *cst = module->getDialect()->materializeConstant(
        builder, constantValue, value.getType(), value.getLoc());
    assert(cst && "all FIRRTL constants can be materialized");
    auto cstValue = cst->getResult(0);

    // Replace all uses of this value with the constant, unless this is the
    // destination of a connect.  We leave those alone to avoid upsetting flow.
    value.replaceUsesWithIf(cstValue, [](OpOperand &operand) {
      if (isa<ConnectOp>(operand.getOwner()) && operand.getOperandNumber() == 0)
        return false;
      return true;
    });
    return true;
  };

  // Constant propagate any ports that are always constant.
  for (auto &port : body->getArguments())
    replaceValueIfPossible(port);

  // TODO: Walk 'when's preorder with `walk`.

  // Walk the IR bottom-up when folding.  We often fold entire chains of
  // operations into constants, which make the intermediate nodes dead.  Going
  // bottom up eliminates the users of the intermediate ops, allowing us to
  // aggressively delete them.
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(*body))) {
    // Connects to values that we found to be constant can be dropped.
    if (auto connect = dyn_cast<ConnectOp>(op)) {
      if (auto *destOp = connect.dest().getDefiningOp()) {
        if (isDeletableWireOrReg(destOp) && !isOverdefined(connect.dest())) {
          connect.erase();
          ++numErasedOp;
        }
      }
      continue;
    }

    // We only fold single-result ops and instances in practice, because they
    // are the expressions.
    if (op.getNumResults() != 1 && !isa<InstanceOp>(op))
      continue;

    // If this operation is already dead, then go ahead and remove it.
    if (op.use_empty() &&
        (wouldOpBeTriviallyDead(&op) || isDeletableWireOrReg(&op))) {
      op.erase();
      continue;
    }

    // Don't "refold" constants.  TODO: Unique in the module entry block.
    if (isa<ConstantOp, SpecialConstantOp, InvalidValueOp>(op))
      continue;

    // If the op had any constants folded, replace them.
    builder.setInsertionPoint(&op);
    bool foldedAny = false;
    for (auto result : op.getResults())
      foldedAny |= replaceValueIfPossible(result);

    if (foldedAny)
      ++numFoldedOp;

    // If the operation folded to a constant then we can probably nuke it.
    if (foldedAny && op.use_empty() &&
        (wouldOpBeTriviallyDead(&op) || isDeletableWireOrReg(&op))) {
      op.erase();
      ++numErasedOp;
      continue;
    }
  }
}

std::unique_ptr<mlir::Pass> circt::firrtl::createIMConstPropPass() {
  return std::make_unique<IMConstPropPass>();
}
