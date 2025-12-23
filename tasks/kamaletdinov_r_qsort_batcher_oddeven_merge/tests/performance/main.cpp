#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "kamaletdinov_r_qsort_batcher_oddeven_merge/common/include/common.hpp"
#include "kamaletdinov_r_qsort_batcher_oddeven_merge/mpi/include/ops_mpi.hpp"
#include "kamaletdinov_r_qsort_batcher_oddeven_merge/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kamaletdinov_quicksort_with_batcher_even_odd_merge {

class KamaletdinovQuicksortWithBatcherEvenOddMergePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 40000000;
  InType input_data_;
  OutType res_;

  void SetUp() override {
    std::vector<int> vec(kCount_);
    for (int i = 0; i < kCount_; i++) {
      vec[i] = kCount_ - i;
    }
    input_data_ = vec;
    std::ranges::sort(vec);
    res_ = vec;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return res_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KamaletdinovQuicksortWithBatcherEvenOddMergePerfTests, QuicksortWithBatcherEvenOddMergePerf) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KamaletdinovQuicksortWithBatcherEvenOddMergeMPI,
                                                       KamaletdinovQuicksortWithBatcherEvenOddMergeSEQ>(
    PPC_SETTINGS_kamaletdinov_r_qsort_batcher_oddeven_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KamaletdinovQuicksortWithBatcherEvenOddMergePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(QuicksortWithBatcherEvenOddMergePerf, KamaletdinovQuicksortWithBatcherEvenOddMergePerfTests,
                         kGtestValues, kPerfTestName);

}  // namespace kamaletdinov_quicksort_with_batcher_even_odd_merge
