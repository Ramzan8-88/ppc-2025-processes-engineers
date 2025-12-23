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

  // Распределяем строки транспонированной матрицы между процессами
  // Транспонированная матрица имеет n строк по m элементов каждая
  std::size_t rows_per_process = n / static_cast<std::size_t>(mpi_size);
  std::size_t remainder = n % static_cast<std::size_t>(mpi_size);

  // Подготовка sendcounts и displacements для Scatterv
  std::vector<int> sendcounts(mpi_size);
  std::vector<int> displacements(mpi_size);
  std::size_t current_displacement = 0;

  for (int i = 0; i < mpi_size; ++i) {
    std::size_t rows_for_process = rows_per_process;
    if (static_cast<std::size_t>(i) < remainder) {
      rows_for_process += 1;
    }
    sendcounts[i] = static_cast<int>(rows_for_process * m);
    displacements[i] = static_cast<int>(current_displacement);
    current_displacement += rows_for_process * m;
  }

  // Локальный буфер для приема данных
  std::size_t local_elements = sendcounts[rank];
  std::size_t local_rows = local_elements / m;
  std::vector<int> local_data(local_elements);

  // Распределяем данные с использованием Scatterv
  MPI_Scatterv(t_matrix_.data(),      // sendbuf
               sendcounts.data(),     // sendcounts
               displacements.data(),  // displacements
               MPI_INT,               // datatype
               local_data.data(),     // recvbuf
               sendcounts[rank],      // recvcount
               MPI_INT,               // datatype
               0,                     // root
               MPI_COMM_WORLD         // comm
  );

  // Вычисляем локальный максимум для строк, которые получил этот процесс
  // Каждая строка транспонированной матрицы соответствует столбцу исходной матрицы
  std::vector<int> local_max(n, std::numeric_limits<int>::min());

  // Определяем, какие глобальные строки (столбцы исходной матрицы) обрабатывает этот процесс
  std::size_t start_row = 0;
  for (int i = 0; i < rank; ++i) {
    std::size_t rows_for_i = rows_per_process;
    if (static_cast<std::size_t>(i) < remainder) {
      rows_for_i += 1;
    }
    start_row += rows_for_i;
  }

  // Обрабатываем локальные строки
  for (std::size_t local_row = 0; local_row < local_rows; ++local_row) {
    std::size_t global_row = start_row + local_row;
    if (global_row >= n) {
      break;
    }
    // Инициализируем максимум первым элементом строки
    local_max[global_row] = local_data[local_row * m];
    // Находим максимум в строке
    for (std::size_t col = 1; col < m; ++col) {
      local_max[global_row] = std::max(local_max[global_row], local_data[local_row * m + col]);
    }
  }

  // Корневой процесс получает все локальные максимумы
  std::vector<int> recvbuf;
  if (rank == 0) {
    recvbuf.resize(static_cast<std::size_t>(mpi_size) * n, std::numeric_limits<int>::min());
  }

  MPI_Gather(local_max.data(), static_cast<int>(n), MPI_INT, rank == 0 ? recvbuf.data() : nullptr, static_cast<int>(n),
             MPI_INT, 0, MPI_COMM_WORLD);

  // На корневом процессе вычисляем финальные максимумы
  if (rank == 0) {
    std::vector<int> result(n);
    for (std::size_t col = 0; col < n; ++col) {
      result[col] = std::numeric_limits<int>::min();
      for (int proc = 0; proc < mpi_size; ++proc) {
        result[col] = std::max(result[col], recvbuf[proc * n + col]);
      }
    }
    GetOutput() = result;
  }

  return true;
}

bool KamaletdinovRMaxMatrixRowsElemMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kamaletdinov_r_max_matrix_rows_elem
