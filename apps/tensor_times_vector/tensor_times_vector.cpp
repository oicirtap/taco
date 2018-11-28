#include <iostream>
#include "taco.h"

using namespace taco;

int main(int argc, char* argv[]) {
  // Create formats
  Format csr({Dense,Sparse});
  Format csf({Sparse,Sparse,Sparse});
  Format  sv({Sparse});

  // Create tensors
  Tensor<double> A({2,3},   csr);
  Tensor<double> B({2,3,4}, csf);
  Tensor<double> c({4},     sv);

  // Insert data into B and c
  B(0,0,0) = 1.0;
  B(1,2,0) = 2.0;
  B(1,3,1) = 3.0;
  c(0) = 4.0;
  c(1) = 5.0;
  //c.insert({1}, 5.0);

  //std::cout << B << std::endl;
  //std::cout << c << std::endl;

  // Form a tensor-vector multiplication expression
  IndexVar i, j, k;
  A(i,j) = B(i,j,k) * c(k);

  //A(i,j) = 1 + 3;

  //double i = B(0,0,0);

  //Tensor<double> t = B(i,j,k);

  //A(0, 0) = 10 + 1;

  //B.insert({0,0,0}, 2.0);
  //std::cout << B << std::endl;

  std::cout << A << std::endl;

  double n = A.getValue<double>({1,2});
  std::cout << n << std::endl;
}
