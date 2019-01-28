#ifndef TACO_TENSOR_API_H
#define TACO_TENSOR_API_H

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

class Tensor;

/// A small structure to represent a multidimensional coordinate tuple (i,j,k,...).
template<int order>
struct Coordinate {
public:
  Coordinate() : coordinate({}) {}

  template<typename... Ints>
  Coordinate(Ints... coordinates) : coordinate({coordinates...}) {}

  int get(int mode) {
    return coordinate[mode];
  }

  int[] get() {
    return coordinate;
  }

  int size() {
    return order;
  }

private:
  int coordinate[order];
};

/// A small structure to hold a non zero as a tuple (coordinate, value).
///
/// CType the type of the value stored.
/// order the number of dimensions of the component
template<typename CType, int order>
struct Component {
public:
  Component() : e_value(0) {}

  template<typename... Ints>
  Component(CType v, Ints... coordinates) : coordinate({coordinates...}), e_value(v) {
    taco_uassert(coordinate.size() == (size_t)order) <<
      "Wrong number of indices";
  }

  Component(CType v, Coordinate<order> coordinate) : coordinate(coordinate), e_value(v) {
    taco_uassert(coordinate.size() == (size_t)order) <<
      "Wrong number of indices";
  }

  int (int mode) const {
    taco_uassert(mode < order) << "Invalid mode";
    return coordinate[mode];
  }

  const Coordinate<order> dimensions() const {
    return coordinate;
  }

  const CType& value() const { return e_value; }

private:
  Coordinate<order> coordinate;
  T e_value;
};

/// The Tensor class represents all tensors in taco. The CType template typename
/// represents the type of the components of the tensor.
///
/// Tensor object copies copies the reference, and
/// subsequent method calls affect both tensor references. To deeply copy a
/// tensor (for instance to change the format) compute a copy index expression
/// e.g. `A(i,j) = B(i,j).
template <typename CType, int order>
class Tensor {
public:
  /* --- Constructor Methods --- */

  /// Create a scalar.
  ///
  /// Creates a scalar with the default name and value.
  Tensor();

  /// Create a scalar.
  ///
  /// @param name the internal name of the Tensor.
  explicit Tensor(std::string name);

  /// Create a scalar.
  ///
  /// @param value the value stored in this scalar.
  explicit Tensor(CType value);

  /// Create a tensor with the given dimensions. The format defaults to sparse 
  /// in every mode.
  ///
  /// @param dimensions the dimensions of each mode of the tensor.
  /// @param modeType
  Tensor(std::vector<int> dimensions, ModeFormat modeType = ModeFormat::compressed);

  /// Create a tensor with the given dimensions and format
  ///
  /// @param dimensions the dimensions of each mode of the tensor.
  /// @param format the format of the tensor.
  ///
  /// The format dimensions must match the number of modes provided in dimensions.
  Tensor(std::vector<int> dimensions, Format format);

  /// Create a tensor with the given name, dimensions and format. The format 
  /// defaults to sparse in every mode.
  ///
  /// @param name the internal name of the Tensor.
  /// @param dimensions the dimensions of each mode of the tensor.
  /// @param modeType
  Tensor(std::string name, std::vector<int> dimensions, 
         ModeFormat modeType = ModeFormat::compressed);

  /// Create a tensor with the given name, dimensions and format
  ///
  /// @param name the internal name of the Tensor.
  /// @param dimensions the dimensions of each mode of the tensor.
  /// @param format the format of the tensor.
  ///
  /// The format dimensions must match the number of modes provided in dimensions.
  Tensor(std::string name, std::vector<int> dimensions, Format format);

  /* --- Metadata Methods    --- */

  /// Get the name of the tensor.
  std::string name() const;

  /// Get the order of the tensor (the number of modes).
  int order() const;

  /// Get the dimension of a tensor mode.
  int getDimension(int mode) const;

  /// Get a vector with the dimension of each tensor modes.
  const std::vector<int>& dimensions() const;

  /// Return the type of the tensor components.
  const Datatype& componentType() const;

  /// Get the format the tensor is packed into.
  const Format& format() const;

  /// Returns the storage for this tensor.
  ///
  /// Tensor values are stored according to the format of the tensor.
  ///
  /// Note: The TensorStorage object is part of the internal representation of
  /// Tensor. Modifying this object breaks the Tensor layer of abstraction.
  /// Access to TensorStorage is provided to facilitate custom Tensor operations
  /// not supported by taco.
  const TensorStorage& storage() const;
  TensorStorage& storage();

  /* --- Write Methods       --- */

  /// Store a scalar value to a coordinate of the tensor.
  ///
  /// @param coordinate the location at which to insert the value
  /// @param value the scalar value to be inserted to the Tensor.
  void insert(const std::initializer_list<int>& coordinate, CType value);
  void insert(const std::vector<int>& coordinate, CType value);

  /// Fill the tensor with the list of components defined by the iterator range (begin, end).
  ///
  /// The input list of triplets does not have to be sorted, and can contains duplicated elements.
  /// The result is a Tensor where the duplicates have been summed up.
  /// The InputIterators value_type must provide the following interface:
  ///
  /// CType value() const;                    // the value
  /// Coordinate<order> dimensions() const;   // the coordinate
  /// 
  /// See for instance the taco::Component template class.
  template <typename InputIterators>
  void setFromComponents(const InputIterators& begin, const InputIterators& end);

  /// The same as setFromTriplets but when duplicates are met the functor dup_func is applied:
  ///
  /// value = dup_func(OldValue, NewValue)
  template <typename InputIterators, typename DupFunctor>
  void setFromComponents(const InputIterators& begin, const InputIterators& end, DupFunctor dup_func);

  /// Assign an expression to a scalar tensor.
  void operator=(const IndexExpr&);

  /* --- Read Methods        --- */

  CType getValue(const std::vector<size_t>& coordinate);

  template<typename T>
  class const_iterator {
  public:
    typedef const_iterator self_type;
    typedef std::pair<std::vector<T>,CType>  value_type;
    typedef std::pair<std::vector<T>,CType>& reference;
    typedef std::pair<std::vector<T>,CType>* pointer;
    typedef std::forward_iterator_tag iterator_category;

    const_iterator(const const_iterator&);

    const_iterator operator++();

    const_iterator operator++(int);

    const Component<CType, order>& operator*();

    const Component<CType, order>* operator->();

    bool operator==(const const_iterator& rhs);

    bool operator!=(const const_iterator& rhs);

  private:
    friend class Tensor;

    const_iterator(const Tensor<CType>* tensor, bool isEnd = false);

    void advanceIndex();

    bool advanceIndex(int lvl);

    const Tensor<CType,order>*       tensor;
    TypedIndexVector                 coord;
    TypedIndexVector                 ptrs;
    Component<CType, order>          curVal;
    size_t                           count;
    bool                             advance;
  };

  const_iterator<size_t> begin() const;

  const_iterator<size_t> end() const;

  template<typename T>
  const_iterator<T> beginTyped() const;

  template<typename T>
  const_iterator<T> endTyped() const;

  /* --- Access Methods      --- */

  /// ScalarAccess objects are defined to simplify the sintax used for inserting
  /// and getting scalar values stored in a tensor.
  class ScalarAccess {
  public:
    ScalarAccess(TensorBase * tensor, const std::vector<int>& indices);

    void operator=(CType scalar) {
      tensor->insert(indices, scalar);
    }

    operator CType() {
      return tensor->getValue<CType>(indices);
    }
  };

  /// Create a ScalarAccess object to read or write scalar values to the Tensor.
  ///
  /// @param indices the coordinate of the scalar to be accessed.
  ///
  /// Example usage:
  /// A(0,0) = 10;
  /// double n = A(1,1);
  template <typename... Ints>
  ScalarAccess operator()(const Ints&... indices);

  /// Create a ScalarAccess object to read or write scalar values to the Tensor.
  ///
  /// @param indices index variables used to iterate over the tensor content.
  ///
  /// Access objects are defined to enable Index expression notation to define
  /// tensor operations
  /// Example usage:
  /// a(i) = B(i,j) * c(j);
  template <typename... IndexVars>
  Access operator()(const IndexVars&... indices);

  /* --- Compiler Methods    --- */

  /// Pack all values in the tensor to its specified format.
  void pack();

  /// Compile the kernel for the assigned tensor expression.
  void compile();

  /// Assemble the tensor storage, including index and value arrays.
  void assemble();

  /// Compute the given expression and put the values in the tensor storage.
  void compute();

  /// Compile, assemble and compute as needed.
  void evaluate();

  /* --- Friend Functions    --- */

  /// True iff two tensors have the same type and the same values.
  friend bool equals(const TensorBase&, const TensorBase&);

  /// True iff two TensorBase objects refer to the same tensor (TensorBase
  /// and Tensor objects are references to tensors).
  friend bool operator==(const TensorBase& a, const TensorBase& b);
  friend bool operator!=(const TensorBase& a, const TensorBase& b);

  /// True iff the address of the tensor referenced by a is smaller than the
  /// address of b.  This is arbitrary and non-deterministic, but necessary for
  /// tensor to be placed in maps.
  friend bool operator<(const TensorBase& a, const TensorBase& b);
  friend bool operator>(const TensorBase& a, const TensorBase& b);
  friend bool operator<=(const TensorBase& a, const TensorBase& b);
  friend bool operator>=(const TensorBase& a, const TensorBase& b);

  /// Print a tensor to a stream.
  friend std::ostream& operator<<(std::ostream&, const TensorBase&);
  friend std::ostream& operator<<(std::ostream&, TensorBase&);  
};

}
#endif
