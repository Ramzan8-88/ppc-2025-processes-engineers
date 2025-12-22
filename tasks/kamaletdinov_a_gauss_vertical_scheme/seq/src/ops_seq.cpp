#include "kamaletdinov_a_gauss_vertical_scheme/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "kamaletdinov_a_gauss_vertical_scheme/common/include/common.hpp"

namespace kamaletdinov_a_gauss_vertical_scheme {

KamaletdinovAGaussVerticalSchemeSEQ::KamaletdinovAGaussVerticalSchemeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KamaletdinovAGaussVerticalSchemeSEQ::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }
  int n = static_cast<int>(GetInput()[0]);
  if (n <= 0) {
    return false;
  }
  std::size_t expected_size = 1 + (static_cast<std::size_t>(n) * static_cast<std::size_t>(n + 1));
  return GetInput().size() == expected_size;
}

bool KamaletdinovAGaussVerticalSchemeSEQ::PreProcessingImpl() {
  n_ = static_cast<int>(GetInput()[0]);
  extended_matrix_.resize(static_cast<std::size_t>(n_) * (n_ + 1));
  std::copy(GetInput().begin() + 1, GetInput().end(), extended_matrix_.begin());
  solution_.resize(n_, 0.0);
  return true;
}

int KamaletdinovAGaussVerticalSchemeSEQ::FindPivotRow(int k, int cols) {
  int max_row = k;
  double max_val = std::abs(extended_matrix_[(k * cols) + k]);
  for (int i = k + 1; i < n_; i++) {
    double val = std::abs(extended_matrix_[(i * cols) + k]);
    if (val > max_val) {
      max_val = val;
      max_row = i;
    }
  }
  return max_row;
}

void KamaletdinovAGaussVerticalSchemeSEQ::SwapRows(int row1, int row2, int cols) {
  for (int j = 0; j < cols; j++) {
    std::swap(extended_matrix_[(row1 * cols) + j], extended_matrix_[(row2 * cols) + j]);
  }
}

void KamaletdinovAGaussVerticalSchemeSEQ::EliminateColumn(int k, int cols) {
  double pivot = extended_matrix_[(k * cols) + k];
  if (std::abs(pivot) < 1e-10) {
    return;
  }

  for (int j = k; j < cols; j++) {
    extended_matrix_[(k * cols) + j] /= pivot;
  }

  for (int i = k + 1; i < n_; i++) {
    double factor = extended_matrix_[(i * cols) + k];
    for (int j = k; j < cols; j++) {
      extended_matrix_[(i * cols) + j] -= factor * extended_matrix_[(k * cols) + j];
    }
  }
}

void KamaletdinovAGaussVerticalSchemeSEQ::BackSubstitution() {
  int cols = n_ + 1;
  for (int i = n_ - 1; i >= 0; i--) {
    solution_[i] = extended_matrix_[(i * cols) + n_];
    for (int j = i + 1; j < n_; j++) {
      solution_[i] -= extended_matrix_[(i * cols) + j] * solution_[j];
    }
  }
}

bool KamaletdinovAGaussVerticalSchemeSEQ::RunImpl() {
  int cols = n_ + 1;
  for (int k = 0; k < n_; k++) {
    int max_row = FindPivotRow(k, cols);
    if (max_row != k) {
      SwapRows(k, max_row, cols);
    }
    EliminateColumn(k, cols);
  }
  BackSubstitution();
  return true;
}

bool KamaletdinovAGaussVerticalSchemeSEQ::PostProcessingImpl() {
  GetOutput() = solution_;
  return true;
}

}  // namespace kamaletdinov_a_gauss_vertical_scheme
