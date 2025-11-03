#pragma once

#include "kamaletdinov_r_max_matrix_rows_elem/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kamaletdinov_r_max_matrix_rows_elem{

class KamaletdinovRMaxMatrixRowsElemMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KamaletdinovRMaxMatrixRowsElemMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kamaletdinov_r_max_matrix_rows_elem
