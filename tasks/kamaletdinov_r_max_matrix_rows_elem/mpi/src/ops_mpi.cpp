#include "kamaletdinov_r_max_matrix_rows_elem/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "kamaletdinov_r_max_matrix_rows_elem/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kamaletdinov_r_max_matrix_rows_elem {

KamaletdinovRMaxMatrixRowsElemMPI::KamaletdinovRMaxMatrixRowsElemMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KamaletdinovRMaxMatrixRowsElemMPI::ValidationImpl() {
  std::size_t n = std::get<0>(GetInput());
  std::size_t m = std::get<1>(GetInput());
  std::vector<int> val = std::get<2>(GetInput());

  return (n > 0) && (m > 0) && (val.size() == (n * m));
}

bool KamaletdinovRMaxMatrixRowsElemMPI::PreProcessingImpl() {
  return true;
}

bool KamaletdinovRMaxMatrixRowsElemMPI::RunImpl() {
  std::size_t n = std::get<0>(GetInput());
  std::size_t m = std::get<1>(GetInput());
  std::vector<int> val = std::get<2>(GetInput());
  if (((n > 0) && (m > 0) && (val.size() == (n * m))) == false) {
    return false;
  }
  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  std::size_t procesess_step = val.size() / mpi_size;
  std::size_t start = procesess_step * rank;
  std::size_t end = procesess_step * (rank + 1);

  if (rank == mpi_size - 1) {
    end = val.size();
  }

  std::vector<int> max_rows_elem(n);
  std::size_t row = start / m;

  for(std::size_t i = start; i < end; i++) {
    if(i == (row + 1) * m) {
      row++;
      max_rows_elem[row] = val[i];
    }
    if(max_rows_elem[row] < val[i]) {
      max_rows_elem[row] = val[i];
    }
  }

  {
    int* recv_buf = new int[n * mpi_size];
    const int* send_buf = max_rows_elem.data();
    MPI_Gather(send_buf, n, MPI_INT, &recv_buf, n * mpi_size, MPI_INT, 0, MPI_COMM_WORLD);
  }

  if(rank == 0) {
    for(size_t i = 0; i < n; i++) {
      for(int j = 0; j < mpi_size; j++) {
        if(max_rows_elem[i] < recv_buf[i*m + n*j]) {
          max_rows_elem[i] = recv_buf[i*m + n*j];
        }
      }
    }
  }
  int* answer = max_rows_elem.data();
  MPI_Bcast(&answer, n, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = max_rows_elem;

  return true;
}

bool KamaletdinovRMaxMatrixRowsElemMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kamaletdinov_r_max_matrix_rows_elem
