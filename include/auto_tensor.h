#ifndef AUTO_TENSOR_H
#define AUTO_TENSOR_H

#include <memory>
#include <string>
#include <vector>
#include <cassert>

#include "taco/type.h"
#include "taco/format.h"

#include "taco/index_notation/index_notation.h"

#include "taco/storage/storage.h"
#include "taco/storage/index.h"
#include "taco/storage/array.h"
#include "taco/storage/typed_vector.h"
#include "taco/storage/typed_index.h"

#include "taco/util/name_generator.h"
#include "taco/error.h"
#include "taco/error/error_messages.h"


namespace taco {

/// AutoTensor is a tensor class which abstracts away compiler related
/// function calls such as pack, assemble, compile, compute...
/*
 * Provides all basic tensor interaction methods such as:
 * - insert
 * - remove
 * - get
 * - getSlice
 * - iterate
 * - print
 * - File I/O *
 */
class AutoTensor {
public:
  /// Create a tensor with the given data type, dimensions and format.
  AutoTensor(std::string name, Datatype ctype, std::vector<int> dimensions,
      Format format);

  // Getter and Setter methods

  /// Set the name of the tensor.
  void setName(std::string name) const;

  /// Get the name of the tensor.
  std::string getName() const;

  /// Get the order of the tensor (the number of modes).
  int getOrder() const;

  /// Get the dimension of a tensor mode.
  int getDimension(int mode) const;

  /// Get a vector with the dimension of each tensor mode.
  const std::vector<int>& getDimensions() const;

  /// Return the type of the tensor components).
  const Datatype& getComponentType() const;

  /// Get the format the tensor is packed into
  const Format& getFormat() const;

  /// Returns the tensor var for this tensor.
  const TensorVar& getTensorVar() const;

  /// Create an index expression that accesses (reads or writes) this tensor.
  Access operator()(const std::vector<IndexVar>& indices);

  /// Set the expression to be evaluated when calling compute or assemble.
  void setAssignment(Assignment assignment);

  /// Set the expression to be evaluated when calling compute or assemble.
  Assignment getAssignment() const;

  /// Get the source code of the kernel functions.
  std::string getSource() const;

  /// Get the taco_tensor_t representation of this tensor.
  taco_tensor_t* getTacoTensorT();

  // Tensor ops

  /// Insert a value into the tensor. The number of coordinates must match the
  /// tensor order.
  template <typename T>
  void insert(const std::vector<int>& coordinate, T value);

  /// Remove a value from the tensor. The number of coordinates must match the
  /// tensor order.
  void remove(const std::vector<int>& coordinate);

  /// Get a value from the tensor. The number of coordinates must match the
  /// tensor order.
  double get(const std::vector<int>& coordinate);

  /// Get a slice of the tensor.
  AutoTensor getSlice(const std::vector<int>& coordinates);

  /// Return an iterator over the values in this tensor.
  iterator iterate();

  /// Print this tensor.
  void print();

  // Overloads

  /// True iff two tensors have the same type and the same values.
  friend bool equals(const AutoTensor&, const AutoTensor&);

  /// Assign an expression to a scalar tensor.
  void operator=(const IndexExpr&);

  /// True iff two AutoTensor objects refer to the same tensor (AutoTensor
  /// and Tensor objects are references to tensors).
  friend bool operator==(const AutoTensor& a, const AutoTensor& b);
  friend bool operator!=(const AutoTensor& a, const AutoTensor& b);

  /// True iff the address of the tensor referenced by a is smaller than the
  /// address of b.  This is arbitrary and non-deterministic, but necessary for
  /// tensor to be placed in maps.
  friend bool operator<(const AutoTensor& a, const AutoTensor& b);
  friend bool operator>(const AutoTensor& a, const AutoTensor& b);
  friend bool operator<=(const AutoTensor& a, const AutoTensor& b);
  friend bool operator>=(const AutoTensor& a, const AutoTensor& b);

  /// Print a tensor to a stream.
  friend std::ostream& operator<<(std::ostream&, const AutoTensor&);

private:
  /// Set the tensor's storage
  void setStorage(TensorStorage storage);

  /// Returns the storage for this tensor. Tensor values are stored according
  /// to the format of the tensor.
  TensorStorage& getStorage();  /// Pack tensor into the given format

  void pack();

  /// Compile the tensor expression.
  void compile(bool assembleWhileCompute=false);

  /// Assemble the tensor storage, including index and value arrays.
  void assemble();

  /// Compute the given expression and put the values in the tensor storage.
  void compute();

  /// Compile, assemble and compute as needed.
  void evaluate();

  /// Set the size of the initial index allocations.  The default size is 1MB.
  void setAllocSize(size_t allocSize);

  /// Get the size of the initial index allocations.
  size_t getAllocSize() const;

  struct Content;
  std::shared_ptr<Content> content;
};
#endif