#include <iostream>
#include "taco.h"

using namespace taco;

int main(int argc, char* argv[]) {
  IndexVar i, j, i1, i2, j1, j2;
  std::cout << "format:" << std::endl;

  std::vector<std::vector<ModeFormat>> test = {{Sparse},{Dense(3)}};
  for (auto i : test) {
    for (auto j : i) {
      std::cout << j << std::endl;
      std::cout << j.hasDimensionSize() << std::endl;
      std::cout << j.getDimensionSize() << std::endl;
    }
  }

  Format  bv2({{Sparse},{Dense(3)}});
  //Format  bv3({{Sparse},{Dense(2)}});
  //Format  bsm2({{Dense,Sparse},{Dense(3),Dense(2)}});

  // Create tensors
  Tensor<double> x({2},      bv2);
  //Tensor<double> Y({2,4},    bsm2);
  Tensor<double> z({2},      bv2);

  // Insert data into B and c
  //Y.insert({0,0,0,0}, 1.0);
  //Y.insert({1,2,0,0}, 2.0);
  //Y.insert({1,2,1,0}, 3.0);
  z.insert({0,0}, 4.0);
  z.insert({1,0}, 5.0);

  // Pack data as described by the formats
  //Y.pack();
  z.pack();

  // Form a tensor-vector multiplication expression
  //a(i) = B(i,j) * c(j); 
  //x(i1,i2) = Y(i1,j1,i2,j2) * z(j1,j2); 
  x(i1,i2) = z(i1,i2);

  // Compile the expression
  x.compile();

  // Assemble A's indices and numerically compute the result
  x.assemble();
  x.compute();

  std::cout << x << std::endl;

  // Create formats
  Format csr({Dense,Sparse});
  Format csf({Sparse,Sparse,Sparse});
  Format  sv({Sparse});

  Format  bv({Sparse,Dense});
  Format  bsm({Dense,Sparse,Dense,Dense});

  // Create tensors
  Tensor<double> a({2,3},      bv);
  Tensor<double> B({2,4,3,2},  bsm);
  Tensor<double> c({4,2},      bv);



  // Insert data into B and c
  B.insert({0,0,0,0}, 1.0);
  B.insert({1,2,0,0}, 2.0);
  B.insert({1,2,1,0}, 3.0);
  c.insert({0,0}, 4.0);
  c.insert({1,0}, 5.0);

  // Pack data as described by the formats
  B.pack();
  c.pack();

  // Form a tensor-vector multiplication expression
  //a(i) = B(i,j) * c(j); 
  a(i1,i2) = B(i1,j1,i2,j2) * c(j1,j2); 

  // Compile the expression
  a.compile();

  // Assemble A's indices and numerically compute the result
  a.assemble();
  a.compute();

  std::cout << a << std::endl;
}
