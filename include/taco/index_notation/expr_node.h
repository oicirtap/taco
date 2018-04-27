#ifndef TACO_EXPR_NODE_H
#define TACO_EXPR_NODE_H

#include <vector>
#include <memory>

#include "taco/type.h"
#include "taco/util/uncopyable.h"
#include "taco/util/intrusive_ptr.h"

namespace taco {

class IndexVar;
class IndexExprVisitorStrict;
class IndexNotationVisitorStrict;
class OperatorSplit;

/// A node of a scalar index expression tree.
struct ExprNode : public util::Manageable<ExprNode>, private util::Uncopyable {
public:
  ExprNode();
  ExprNode(DataType type);
  virtual ~ExprNode() = default;
  virtual void accept(IndexExprVisitorStrict*) const = 0;

  /// Split the expression.
  void splitOperator(IndexVar old, IndexVar left, IndexVar right);

  /// Returns the expression's operator splits.
  const std::vector<OperatorSplit>& getOperatorSplits() const;

  DataType getDataType() const;

private:
  std::shared_ptr<std::vector<OperatorSplit>> operatorSplits;
  DataType dataType;
};


/// A node in a tensor index expression tree
struct IndexStmtNode : public util::Manageable<IndexStmtNode>,
                        private util::Uncopyable {
public:
  IndexStmtNode();
  IndexStmtNode(Type type);
  virtual ~IndexStmtNode() = default;
  virtual void accept(IndexNotationVisitorStrict*) const = 0;

  Type getType() const;

private:
  Type type;
};

}
#endif
