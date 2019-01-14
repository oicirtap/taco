#include <iostream>
#include "taco.h"

using namespace taco;

typedef Element<double, 1> T1;
typedef Element<double, 3> T3;

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
  //std::vector<T3> b_element_list;
  //b_element_list.push_back(T3(1.0,0,0,0));
  //b_element_list.push_back(T3(2.0,1,2,0));
  //b_element_list.push_back(T3(3.0,1,3,1));
  //B.setFromElements(b_element_list.begin(), b_element_list.end());

  B(0,0,0) = 1.0;
  B(1,2,0) = 2.0;
  B(1,3,1) = 3.0;
  
  //std::vector<T1> c_element_list;
  //c_element_list.push_back(T1(4.0,0));
  //c_element_list.push_back(T1(5.0,1));
  //c.setFromElements(c_element_list.begin(), c_element_list.end());

  c(0) = 4.0;
  c(1) = 5.0;

  //c(0) = 4.0;
  //c(1) = 5.0;
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

  double n2 = A(0,0);
  std::cout << n2 << std::endl;
}
