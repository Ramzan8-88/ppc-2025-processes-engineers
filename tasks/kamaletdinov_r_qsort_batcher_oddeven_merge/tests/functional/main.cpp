#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "kamaletdinov_r_qsort_batcher_oddeven_merge/common/include/common.hpp"
#include "kamaletdinov_r_qsort_batcher_oddeven_merge/mpi/include/ops_mpi.hpp"
#include "kamaletdinov_r_qsort_batcher_oddeven_merge/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kamaletdinov_quicksort_with_batcher_even_odd_merge {

class KamaletdinovQuicksortWithBatcherEvenOddMergeFuncTests
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<1>(params);
    res_ = std::get<2>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (res_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType res_;
};

namespace {

TEST_P(KamaletdinovQuicksortWithBatcherEvenOddMergeFuncTests, QuicksortWithBatcherEvenOddMerge) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 12> kTestParam = {
    std::make_tuple("a", std::vector<int>{2, 1}, std::vector<int>{1, 2}),
    std::make_tuple("b", std::vector<int>{0, -1, 1}, std::vector<int>{-1, 0, 1}),
    std::make_tuple("c", std::vector<int>{5, 5, 5}, std::vector<int>{5, 5, 5}),
    std::make_tuple("d", std::vector<int>{10, 9, 8, 7, 6}, std::vector<int>{6, 7, 8, 9, 10}),
    std::make_tuple("e", std::vector<int>{1, 3, 2, 4}, std::vector<int>{1, 2, 3, 4}),
    std::make_tuple("f", std::vector<int>{-5, -10, -3, -1}, std::vector<int>{-10, -5, -3, -1}),
    std::make_tuple("g", std::vector<int>{100}, std::vector<int>{100}),
    std::make_tuple("h", std::vector<int>{4, 2, 4, 1, 3}, std::vector<int>{1, 2, 3, 4, 4}),
    std::make_tuple("i", std::vector<int>{50, 23, 9, 87, 34, 12, 77, 5, 1, 100, 56, 78, 33, 45, 66, 89, 3, 22, 11, 90},
                    std::vector<int>{1, 3, 5, 9, 11, 12, 22, 23, 33, 34, 45, 50, 56, 66, 77, 78, 87, 89, 90, 100}),
    std::make_tuple("j", std::vector<int>{1000, -500, 230, 0, 999, -1, 450, -300, 750, 20, -700, 33, 18, 600, 55},
                    std::vector<int>{-700, -500, -300, -1, 0, 18, 20, 33, 55, 230, 450, 600, 750, 999, 1000}),
    std::make_tuple("k", std::vector<int>{3, -10, 5, 99, -3, -1, 0, 42, 18, -7},
                    std::vector<int>{-10, -7, -3, -1, 0, 3, 5, 18, 42, 99}),
    std::make_tuple("l", std::vector<int>{}, std::vector<int>{}),
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<KamaletdinovQuicksortWithBatcherEvenOddMergeMPI, InType>(
                       kTestParam, PPC_SETTINGS_kamaletdinov_r_qsort_batcher_oddeven_merge),
                   ppc::util::AddFuncTask<KamaletdinovQuicksortWithBatcherEvenOddMergeSEQ, InType>(
                       kTestParam, PPC_SETTINGS_kamaletdinov_r_qsort_batcher_oddeven_merge));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KamaletdinovQuicksortWithBatcherEvenOddMergeFuncTests::PrintFuncTestName<
    KamaletdinovQuicksortWithBatcherEvenOddMergeFuncTests>;

INSTANTIATE_TEST_SUITE_P(QuicksortWithBatcherEvenOddMerge, KamaletdinovQuicksortWithBatcherEvenOddMergeFuncTests,
                         kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kamaletdinov_quicksort_with_batcher_even_odd_merge
