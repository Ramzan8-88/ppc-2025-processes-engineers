#include "kamaletdinov_a_gauss_vertical_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "kamaletdinov_a_gauss_vertical_scheme/common/include/common.hpp"

namespace kamaletdinov_a_gauss_vertical_scheme {

KamaletdinovAGaussVerticalSchemeMPI::KamaletdinovAGaussVerticalSchemeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KamaletdinovAGaussVerticalSchemeMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  if (rank_ != 0) {
    return true;
  }
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

bool KamaletdinovAGaussVerticalSchemeMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

  if (rank_ == 0) {
    n_ = static_cast<int>(GetInput()[0]);
  }
  MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  extended_matrix_.resize(static_cast<std::size_t>(n_) * (n_ + 1));

  if (rank_ == 0) {
    std::copy(GetInput().begin() + 1, GetInput().end(), extended_matrix_.begin());
    DistributeMatrixByStripes();
  } else {
    ReceiveMatrixStripe();
  }

  SynchronizeMatrixByStripes();

  solution_.resize(n_, 0.0);
  return true;
}

void KamaletdinovAGaussVerticalSchemeMPI::DistributeMatrixByStripes() {
  int cols = n_ + 1;
  for (int proc = 1; proc < size_; proc++) {
    for (int i = 0; i < n_; i++) {
      std::vector<double> stripe;
      for (int j = proc; j < cols; j += size_) {
        stripe.push_back(extended_matrix_[(i * cols) + j]);
      }
      MPI_Send(stripe.data(), static_cast<int>(stripe.size()), MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
    }
  }
}

void KamaletdinovAGaussVerticalSchemeMPI::ReceiveMatrixStripe() {
  int cols = n_ + 1;
  for (int i = 0; i < n_; i++) {
    int stripe_size = 0;
    for (int j = rank_; j < cols; j += size_) {
      stripe_size++;
    }
    std::vector<double> stripe(stripe_size);
    MPI_Recv(stripe.data(), stripe_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int idx = 0;
    for (int j = rank_; j < cols; j += size_) {
      extended_matrix_[(i * cols) + j] = stripe[idx++];
    }
  }
}

void KamaletdinovAGaussVerticalSchemeMPI::SynchronizeMatrixByStripes() {
  for (int i = 0; i < n_; i++) {
    ExchangeStripesForRow(i);
  }
}

void KamaletdinovAGaussVerticalSchemeMPI::ExchangeStripesForRow(int row) {
  for (int proc = 0; proc < size_; proc++) {
    if (proc != rank_) {
      ExchangeStripeWithProcess(row, proc);
    }
  }
}

void KamaletdinovAGaussVerticalSchemeMPI::ExchangeStripeWithProcess(int row, int proc) {
  int cols = n_ + 1;
  std::vector<double> my_stripe;
  for (int j = rank_; j < cols; j += size_) {
    my_stripe.push_back(extended_matrix_[(row * cols) + j]);
  }

  int recv_size = 0;
  for (int j = proc; j < cols; j += size_) {
    recv_size++;
  }
  std::vector<double> recv_stripe(recv_size);

  if (rank_ < proc) {
    MPI_Send(my_stripe.data(), static_cast<int>(my_stripe.size()), MPI_DOUBLE, proc, row, MPI_COMM_WORLD);
    MPI_Recv(recv_stripe.data(), recv_size, MPI_DOUBLE, proc, row, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
    MPI_Recv(recv_stripe.data(), recv_size, MPI_DOUBLE, proc, row, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(my_stripe.data(), static_cast<int>(my_stripe.size()), MPI_DOUBLE, proc, row, MPI_COMM_WORLD);
  }

  int idx = 0;
  for (int j = proc; j < cols; j += size_) {
    extended_matrix_[(row * cols) + j] = recv_stripe[idx++];
  }
}

int KamaletdinovAGaussVerticalSchemeMPI::FindPivotRow(int k, int cols) {
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

void KamaletdinovAGaussVerticalSchemeMPI::SwapRows(int row1, int row2, int cols) {
  for (int j = 0; j < cols; j++) {
    std::swap(extended_matrix_[(row1 * cols) + j], extended_matrix_[(row2 * cols) + j]);
  }
}

void KamaletdinovAGaussVerticalSchemeMPI::SynchronizeRow(int k, int row, int cols) {
  std::vector<double> row_data(cols - k);
  for (int j = k; j < cols; j++) {
    row_data[j - k] = extended_matrix_[(row * cols) + j];
  }

  if (rank_ == 0) {
    for (int proc = 1; proc < size_; proc++) {
      std::vector<double> recv_data(cols - k);
      MPI_Recv(recv_data.data(), cols - k, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (int j = k + proc; j < cols; j += size_) {
        row_data[j - k] = recv_data[j - k];
      }
    }
    for (int j = k; j < cols; j++) {
      extended_matrix_[(row * cols) + j] = row_data[j - k];
    }
    for (int proc = 1; proc < size_; proc++) {
      MPI_Send(row_data.data(), cols - k, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
    }
  } else {
    MPI_Send(row_data.data(), cols - k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    MPI_Recv(row_data.data(), cols - k, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int j = k; j < cols; j++) {
      extended_matrix_[(row * cols) + j] = row_data[j - k];
    }
  }
}

void KamaletdinovAGaussVerticalSchemeMPI::EliminateColumn(int k, int cols) {
  double pivot = extended_matrix_[(k * cols) + k];
  if (std::abs(pivot) < 1e-10) {
    return;
  }

  for (int j = k; j < cols; j++) {
    extended_matrix_[(k * cols) + j] /= pivot;
  }

  for (int i = k + 1; i < n_; i++) {
    double factor = extended_matrix_[(i * cols) + k];
    int start_col = k + rank_;
    for (int j = start_col; j < cols; j += size_) {
      extended_matrix_[(i * cols) + j] -= factor * extended_matrix_[(k * cols) + j];
    }
  }

  if (size_ > 1) {
    for (int i = k + 1; i < n_; i++) {
      SynchronizeRow(k, i, cols);
    }
  }
}

void KamaletdinovAGaussVerticalSchemeMPI::BackSubstitution() {
  int cols = n_ + 1;
  if (rank_ == 0) {
    for (int i = n_ - 1; i >= 0; i--) {
      solution_[i] = extended_matrix_[(i * cols) + n_];
      for (int j = i + 1; j < n_; j++) {
        solution_[i] -= extended_matrix_[(i * cols) + j] * solution_[j];
      }
    }
  }
}

bool KamaletdinovAGaussVerticalSchemeMPI::RunImpl() {
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

bool KamaletdinovAGaussVerticalSchemeMPI::PostProcessingImpl() {
  MPI_Bcast(solution_.data(), n_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = solution_;
  return true;
}

}  // namespace kamaletdinov_a_gauss_vertical_scheme
