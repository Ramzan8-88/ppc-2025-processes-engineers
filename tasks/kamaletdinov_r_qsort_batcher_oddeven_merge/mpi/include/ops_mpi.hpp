#pragma once

#include <vector>

#include "kamaletdinov_r_qsort_batcher_oddeven_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kamaletdinov_quicksort_with_batcher_even_odd_merge {

class KamaletdinovQuicksortWithBatcherEvenOddMergeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KamaletdinovQuicksortWithBatcherEvenOddMergeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void NeighborExchange(std::vector<int> &local, int partner_rank, bool keep_lower);
  void BroadcastOutputToAllRanks();
  void BatcherPhases(std::vector<int> &local, int rank, int size, int global_size);
};

}  // namespace kamaletdinov_quicksort_with_batcher_even_odd_merge
