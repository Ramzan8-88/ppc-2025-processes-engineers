#pragma once

#include <vector>

#include "kamaletdinov_a_gauss_vertical_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kamaletdinov_a_gauss_vertical_scheme {

class KamaletdinovAGaussVerticalSchemeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KamaletdinovAGaussVerticalSchemeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int FindPivotRow(int k, int cols);
  void SwapRows(int row1, int row2, int cols);
  void EliminateColumn(int k, int cols);
  void BackSubstitution();

  int n_{0};
  std::vector<double> extended_matrix_;
  std::vector<double> solution_;
};

}  // namespace kamaletdinov_a_gauss_vertical_scheme
