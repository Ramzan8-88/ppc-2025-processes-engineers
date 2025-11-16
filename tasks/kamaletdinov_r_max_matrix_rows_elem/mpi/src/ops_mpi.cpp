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
  if(valid_) {
    std::size_t m = std::get<0>(GetInput());
    std::size_t n = std::get<1>(GetInput());
    std::vector<int> &val = std::get<2>(GetInput());
    t_matrix_ = std::vector<int>(n * m);
    for(std::size_t i = 0; i < m; i++) {
      for(std::size_t j = 0; j < n; j++) {
        t_matrix_[(j * m) + i] = val[(i * n) + j];
      }
    }
    return true;
  }
  return false;
}

bool KamaletdinovRMaxMatrixRowsElemMPI::RunImpl() {
  //проверка корректности данных
  if(!valid_) {
    return false;
  }
  //получение размера матрицы
  std::size_t m = std::get<0>(GetInput());
  std::size_t n = std::get<1>(GetInput());
  
  //debug
  // std::string deb = "\n\n-----------\n";
  // for(std::size_t i = 0; i < n; i++) {
  //   for(std::size_t j = 0; j < m; j++) {
  //     deb += std::to_string(t_matrix_[i*m + j]) + " ";
  //   }
  //   deb += "\n";
  // }
  // std::cout << deb;

  //данные о процессе
  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  //рассчет части матрицы для обработки процесса
  std::size_t procesess_step = t_matrix_.size() / mpi_size;
  std::size_t start = procesess_step * rank;
  std::size_t end = procesess_step * (rank + 1);

  if (rank == mpi_size - 1) {
    end = t_matrix_.size();
  }

  //выделение памяти для сохранения максимального элемента
  std::vector<int> max_rows_elem;
  if(rank == 0) {
    max_rows_elem.resize(n * mpi_size, 0);
  } else {
    max_rows_elem.resize(n, 0);
  }
  std::size_t row = start / m;
  max_rows_elem[row] = t_matrix_[start];

  for(std::size_t i = start; i < end; i++) {
    if(i == (row + 1) * m) {
      row++;
      max_rows_elem[row] = t_matrix_[i];
    }
    if(max_rows_elem[row] < t_matrix_[i]) {
      max_rows_elem[row] = t_matrix_[i];
    }
  }

  //debug output
  // for (std::size_t i = 0; i < max_rows_elem.size(); i++) {
  //   std::cout << max_rows_elem[i] << " ";
  // }
  // std::cout << "\n";
  
  MPI_Gather(max_rows_elem.data(), n, MPI_INT, max_rows_elem.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  if(rank == 0) {
    for(std::size_t i = 0; i < n; i++) {
      for(int j = 0; j < mpi_size; j++) {
        if(max_rows_elem[i] < max_rows_elem[j * n + i]) {
          max_rows_elem[i] = max_rows_elem[j * n + i];
        }
      }
    }
  }
  
  MPI_Bcast(max_rows_elem.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

  //debug output
  // std::cout << rank << ":";
  // for(std::size_t i = 0; i < n; i++) {
  //   std::cout << max_rows_elem[i] << " ";
  // }
  // std::cout << std::endl;

  GetOutput() = max_rows_elem;

  return true;
}

bool KamaletdinovRMaxMatrixRowsElemMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kamaletdinov_r_max_matrix_rows_elem
