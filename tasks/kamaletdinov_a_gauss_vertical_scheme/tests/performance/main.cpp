#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "kamaletdinov_a_gauss_vertical_scheme/common/include/common.hpp"
#include "kamaletdinov_a_gauss_vertical_scheme/mpi/include/ops_mpi.hpp"
#include "kamaletdinov_a_gauss_vertical_scheme/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kamaletdinov_a_gauss_vertical_scheme {

namespace {

std::vector<double> GenerateDiagonallyDominantMatrix(int n) {
  std::vector<double> input;
  input.push_back(static_cast<double>(n));

  for (int i = 0; i < n; i++) {
    double row_sum = 0.0;
    for (int j = 0; j < n; j++) {
      double val = (i == j) ? 0.0 : (static_cast<double>((i + j) % 5) + 1.0);
      input.push_back(val);
      row_sum += std::abs(val);
    }
    input[1 + (i * (n + 1)) + i] = row_sum + static_cast<double>(n);
    double b_val = 0.0;
    for (int j = 0; j < n; j++) {
      b_val += input[1 + (i * (n + 1)) + j];
    }
    input.push_back(b_val);
  }
  return input;
}

std::vector<double> GetExpectedSolution(int n) {
  std::vector<double> expected(n, 1.0);
  return expected;
}

bool CompareVectors(const std::vector<double> &a, const std::vector<double> &b, double eps = 1e-4) {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); i++) {
    if (std::abs(a[i] - b[i]) > eps) {
      return false;
    }
  }
  return true;
}

}  // namespace

class KamaletdinovAGaussVerticalSchemePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixSize_ = 300;
  InType input_data_;
  OutType expected_output_;
  int rank_ = 0;

  void SetUp() override {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    input_data_ = GenerateDiagonallyDominantMatrix(kMatrixSize_);
    expected_output_ = GetExpectedSolution(kMatrixSize_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (rank_ != 0) {
      return true;
    }
    return CompareVectors(expected_output_, output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KamaletdinovAGaussVerticalSchemePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KamaletdinovAGaussVerticalSchemeMPI, KamaletdinovAGaussVerticalSchemeSEQ>(
        PPC_SETTINGS_kamaletdinov_a_gauss_vertical_scheme);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KamaletdinovAGaussVerticalSchemePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KamaletdinovAGaussVerticalSchemePerfTests, kGtestValues, kPerfTestName);

}  // namespace kamaletdinov_a_gauss_vertical_scheme
