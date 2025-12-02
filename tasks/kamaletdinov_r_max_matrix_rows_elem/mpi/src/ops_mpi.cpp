#include "kamaletdinov_r_max_matrix_rows_elem/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "kamaletdinov_r_max_matrix_rows_elem/common/include/common.hpp"

namespace kamaletdinov_r_max_matrix_rows_elem {

KamaletdinovRMaxMatrixRowsElemMPI::KamaletdinovRMaxMatrixRowsElemMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool KamaletdinovRMaxMatrixRowsElemMPI::ValidationImpl() {
  std::size_t m = std::get<0>(GetInput());
  std::size_t n = std::get<1>(GetInput());
  std::vector<int> &val = std::get<2>(GetInput());
  valid_ = (n > 0) && (m > 0) && (val.size() == (n * m));
  return valid_;
}

bool KamaletdinovRMaxMatrixRowsElemMPI::PreProcessingImpl() {
  if (valid_) {
    std::size_t m = std::get<0>(GetInput());
    std::size_t n = std::get<1>(GetInput());
    std::vector<int> &val = std::get<2>(GetInput());
    t_matrix_ = std::vector<int>(n * m);
    for (std::size_t i = 0; i < m; i++) {
      for (std::size_t j = 0; j < n; j++) {
        t_matrix_[(j * m) + i] = val[(i * n) + j];
      }
    }
    return true;
  }
  return false;
}

bool KamaletdinovRMaxMatrixRowsElemMPI::RunImpl() {
  if (!valid_) {
    return false;
  }

  std::size_t m = std::get<0>(GetInput());
  std::size_t n = std::get<1>(GetInput());

  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  std::size_t total = t_matrix_.size();
  std::size_t process_step = total / static_cast<std::size_t>(mpi_size);
  std::size_t start = process_step * static_cast<std::size_t>(rank);
  std::size_t end = process_step * static_cast<std::size_t>(rank + 1);
  if (rank == mpi_size - 1) {
    end = total;
  }

  // compute local max per column
  std::vector<int> local_max(n, std::numeric_limits<int>::min());
  for (std::size_t i = start; i < end; ++i) {
    std::size_t col = i / m;
    local_max[col] = std::max(local_max[col], t_matrix_[i]);
  }

  // root receives all local maxima in recvbuf
  std::vector<int> recvbuf;
  if (rank == 0) {
    recvbuf.resize(static_cast<std::size_t>(mpi_size) * n, std::numeric_limits<int>::min());
  }

  MPI_Gather(local_max.data(), static_cast<int>(n), MPI_INT, rank == 0 ? recvbuf.data() : nullptr, static_cast<int>(n),
             MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> final_max(n, std::numeric_limits<int>::min());
  if (rank == 0) {
    for (int proc = 0; proc < mpi_size; ++proc) {
      const std::size_t offset = static_cast<std::size_t>(proc) * n;
      for (std::size_t i = 0; i < n; ++i) {
        final_max[i] = std::max(final_max[i], recvbuf[offset + i]);
      }
    }
  }

  MPI_Bcast(final_max.data(), static_cast<int>(n), MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = final_max;
  return true;
}

bool KamaletdinovRMaxMatrixRowsElemMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kamaletdinov_r_max_matrix_rows_elem
