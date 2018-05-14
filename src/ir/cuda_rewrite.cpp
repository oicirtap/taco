#include "taco/ir/cuda_rewrite.h"

#include <map>
#include <queue>

#include "taco/ir/ir.h"
#include "taco/ir/ir_visitor.h"
#include "taco/ir/ir_rewriter.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"
#include "taco/util/scopedmap.h"

using namespace std;

namespace taco {
namespace ir {

struct ExpressionCudifier : IRRewriter {
  using IRRewriter::visit;  

  void visit(const Function* op) {
    Stmt body = rewrite(op->body);
    vector<Expr> inputs;
    vector<Expr> outputs;
    bool inputOutputsSame = true;
    for (auto& input : op->inputs) {
      Expr rewrittenInput = rewrite(input);
      inputs.push_back(rewrittenInput);
      if (rewrittenInput != input) {
	inputOutputsSame = false;
      }
    }
    for (auto& output : op->outputs) {
      Expr rewrittenOutput = rewrite(output);
      outputs.push_back(rewrittenOutput);
      if (rewrittenOutput != output) {
	inputOutputsSame = false;
      }
    }
    if (body == op->body && inputOutputsSame) {
      stmt = op;
    }
    else {
      stmt = Function::make(op->name, inputs, outputs, body);
    }

  }

};

ir::Expr cuda_rewrite(const ir::Expr& expr) {
  cout << "rewrite expr" << endl;
  return ExpressionCudifier().rewrite(expr);
}

ir::Stmt cuda_rewrite(const ir::Stmt& stmt) {
  cout << "rewrite stmt" << endl;
  return ExpressionCudifier().rewrite(stmt);
}
}}
