#include <iostream>
#include "taco.h"

using namespace taco;

int main(int argc, char* argv[]) {
  // Create formats
  Format csr({Dense,Sparse});
  Format csc({Dense,Sparse}, {1,0});
  Format csf({Sparse,Sparse,Sparse});
  Format  sv({Sparse});
  
  // Create tensors
  Tensor<double> A({2,3},   Format({Dense,Sparse}, {1,0})); // csr
  Tensor<double> B({2,3,3}, Format({Sparse,Sparse,Sparse}, {2,0,1})); // csf
  Tensor<double> c({3},     Format({Sparse}));  // sv

  // Insert data into B and c
  B.insert({0,0,0}, 1.0);
  B.insert({1,2,0}, 2.0);
  B.insert({1,2,1}, 3.0);
  c.insert({0}, 4.0);
  c.insert({1}, 5.0);

  // Pack data as described by the formats
  B.pack();
  c.pack();

  // Form a tensor-vector multiplication expression
  IndexVar i, j, k;
  A(i,j) = B(i,k,j) * c(k);

  // Compile the expression
  A.compile();

  // Assemble A's indices and numerically compute the result
  A.assemble();
  A.compute();

  //std::cout << A << std::endl;
  //std::cout << A.getSource() << std::endl;

  /*
  Tensor<double> X({3,2},   Format({Dense,Sparse}, {1,0}));
  Tensor<double> Z({3,2},   Format({Dense,Sparse}, {1,0}));

  Z.insert({0,0}, 1.0);
  Z.insert({0,1}, 2.0);
  Z.insert({1,0}, 3.0);
  Z.insert({1,1}, 4.0);
  Z.insert({2,1}, 5.0);
  Z.pack();
  
  IndexVar p, q;
  X(p,q) = Z(p,q);

  X.compile();
  //std::cout << Y.getSource() << std::endl;
  X.assemble();
  X.compute();

  std::cout << "z" << std::endl;
  for (const std::pair<std::vector<size_t>,double>& value : Z) {
    for (int coord : value.first) {
      std::cout << coord << " ";
    }
    std::cout << ":" << value.second << std::endl;
  }

  std::cout << "x" << std::endl;
  for (const std::pair<std::vector<size_t>,double>& value : X) {
    for (int coord : value.first) {
      std::cout << coord << " ";
    }
    std::cout << ":" << value.second << std::endl;
  }
  */
}
