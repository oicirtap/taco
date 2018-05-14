#ifndef TACO_IR_CUDA_REWRITE_H
#define TACO_IR_CUDA_REWRITE_H

namespace taco {
namespace ir {
class Expr;
class Stmt;

/// Simplifies an expression (e.g. by applying algebraic identities).
ir::Expr cuda_rewrite(const ir::Expr& expr);

/// Simplifies a statement (e.g. by applying constant copy propagation).
ir::Stmt cuda_rewrite(const ir::Stmt& stmt);

}}
#endif
