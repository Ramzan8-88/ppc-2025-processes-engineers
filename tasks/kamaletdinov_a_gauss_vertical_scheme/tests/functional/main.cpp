#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "kamaletdinov_a_gauss_vertical_scheme/common/include/common.hpp"
#include "kamaletdinov_a_gauss_vertical_scheme/mpi/include/ops_mpi.hpp"
#include "kamaletdinov_a_gauss_vertical_scheme/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kamaletdinov_a_gauss_vertical_scheme {

namespace {

std::vector<double> CreateInput(int n, const std::vector<double> &matrix) {
  std::vector<double> input;
  input.push_back(static_cast<double>(n));
  input.insert(input.end(), matrix.begin(), matrix.end());
  return input;
}

bool CompareVectors(const std::vector<double> &a, const std::vector<double> &b, double eps = 1e-6) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); i++) {
    if (std::abs(a[i] - b[i]) > eps) {
      return false;
    }
  }
  return true;
}

}  // namespace

class KamaletdinovAGaussVerticalSchemeFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_id = std::get<0>(params);

    switch (test_id) {
      case 1: {
        input_data_ = CreateInput(1, {2.0, 4.0});
        expected_output_ = {2.0};
        break;
      }
      case 2: {
        input_data_ = CreateInput(2, {2.0, 1.0, 5.0, 1.0, 3.0, 5.0});
        expected_output_ = {2.0, 1.0};
        break;
      }
      case 3: {
        input_data_ = CreateInput(3, {2.0, 1.0, -1.0, 8.0, -3.0, -1.0, 2.0, -11.0, -2.0, 1.0, 2.0, -3.0});
        expected_output_ = {2.0, 3.0, -1.0};
        break;
      }
      case 4: {
        input_data_ = CreateInput(3, {1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0});
        expected_output_ = {1.0, 2.0, 3.0};
        break;
      }
      case 5: {
        input_data_ = CreateInput(2, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
        expected_output_ = {-1.0, 2.0};
        break;
      }
      case 6: {
        input_data_ = CreateInput(
            4, {4.0, 1.0, 0.0, 0.0, 5.0, 1.0, 4.0, 1.0, 0.0, 6.0, 0.0, 1.0, 4.0, 1.0, 6.0, 0.0, 0.0, 1.0, 4.0, 5.0});
        expected_output_ = {1.0, 1.0, 1.0, 1.0};
        break;
      }
      case 7: {
        input_data_ = CreateInput(3, {3.0, 0.0, 0.0, 9.0, 0.0, 2.0, 0.0, 8.0, 0.0, 0.0, 4.0, 12.0});
        expected_output_ = {3.0, 4.0, 3.0};
        break;
      }
      case 8: {
        input_data_ = CreateInput(3, {1.0, 2.0, 3.0, 14.0, 0.0, 4.0, 5.0, 23.0, 0.0, 0.0, 6.0, 18.0});
        expected_output_ = {1.0, 2.0, 3.0};
        break;
      }
      case 9: {
        input_data_ = CreateInput(2, {1.0, 1.0, 1.0, 2.0, 1.0, 1.5});
        expected_output_ = {0.5, 0.5};
        break;
      }
      case 10: {
        input_data_ = CreateInput(2, {1.0, 2.0, 0.0, 3.0, 4.0, 0.0});
        expected_output_ = {0.0, 0.0};
        break;
      }
      default: {
        input_data_ = CreateInput(2, {1.0, 0.0, 1.0, 0.0, 1.0, 1.0});
        expected_output_ = {1.0, 1.0};
        break;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return CompareVectors(expected_output_, output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

TEST_P(KamaletdinovAGaussVerticalSchemeFuncTests, GaussSolveTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    std::make_tuple(1, "System1x1"),         std::make_tuple(2, "System2x2_A"),
    std::make_tuple(3, "System3x3_Classic"), std::make_tuple(4, "System3x3_Identity"),
    std::make_tuple(5, "System2x2_B"),       std::make_tuple(6, "System4x4_Tridiag"),
    std::make_tuple(7, "DiagonalMatrix"),    std::make_tuple(8, "UpperTriangular"),
    std::make_tuple(9, "FractionalSol"),     std::make_tuple(10, "ZeroRHS")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KamaletdinovAGaussVerticalSchemeMPI, InType>(
                                               kTestParam, PPC_SETTINGS_kamaletdinov_a_gauss_vertical_scheme),
                                           ppc::util::AddFuncTask<KamaletdinovAGaussVerticalSchemeSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_kamaletdinov_a_gauss_vertical_scheme));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    KamaletdinovAGaussVerticalSchemeFuncTests::PrintFuncTestName<KamaletdinovAGaussVerticalSchemeFuncTests>;

INSTANTIATE_TEST_SUITE_P(GaussSolverTests, KamaletdinovAGaussVerticalSchemeFuncTests, kGtestValues, kPerfTestName);

}  // namespace kamaletdinov_a_gauss_vertical_scheme
