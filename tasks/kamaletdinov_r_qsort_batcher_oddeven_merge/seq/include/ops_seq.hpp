#pragma once
#include <utility>
#include <vector>

#include "kamaletdinov_r_qsort_batcher_oddeven_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kamaletdinov_quicksort_with_batcher_even_odd_merge {

class KamaletdinovQuicksortWithBatcherEvenOddMergeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KamaletdinovQuicksortWithBatcherEvenOddMergeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static std::pair<int, int> PartitionRange(std::vector<int> &array, int left, int right);
  static void PushPartitionsToStack(std::vector<std::pair<int, int>> &stack, int left, int right,
                                    const std::pair<int, int> &borders);
};

}  // namespace kamaletdinov_quicksort_with_batcher_even_odd_merge
