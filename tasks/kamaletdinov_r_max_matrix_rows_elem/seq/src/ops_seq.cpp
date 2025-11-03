#include "kamaletdinov_r_max_matrix_rows_elem/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "kamaletdinov_r_max_matrix_rows_elem/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kamaletdinov_r_max_matrix_rows_elem {
KamaletdinovRMaxMatrixRowsElemSEQ::KamaletdinovRMaxMatrixRowsElemSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool KamaletdinovRMaxMatrixRowsElemSEQ::ValidationImpl() {
  std::size_t n = std::get<0>(GetInput());
  std::size_t m = std::get<1>(GetInput());
  std::vector<int> val = std::get<2>(GetInput());

  return (n > 0) && (m > 0) && (val.size() == (n * m));
}

bool KamaletdinovRMaxMatrixRowsElemSEQ::PreProcessingImpl() {
  return true;
}

bool KamaletdinovRMaxMatrixRowsElemSEQ::RunImpl() {
  // if (GetInput() == 0) {
  //   return false;
  // }
  std::size_t n = std::get<0>(GetInput());
  std::size_t m = std::get<1>(GetInput());
  std::vector<int> val = std::get<2>(GetInput());
  if(((n > 0) && (m > 0) && (val.size() == (n * m))) == false) {
    return false;
  }

  std::vector<int> max_rows_elem(n);
  for(std::size_t i = 0; i < n; i++) {
    max_rows_elem[i] = val[i * m];
    for(std::size_t j = 1; j < m; j++) {
      if(max_rows_elem[i] < val[i * m + j]){
        max_rows_elem[i] = val[i * m + j];
      }
    }
  }
  GetOutput() = max_rows_elem;
  return true;
}

bool KamaletdinovRMaxMatrixRowsElemSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kamaletdinov_r_max_matrix_rows_elem
