#include "kamaletdinov_r_qsort_batcher_oddeven_merge/seq/include/ops_seq.hpp"

#include <utility>
#include <vector>

#include "kamaletdinov_r_qsort_batcher_oddeven_merge/common/include/common.hpp"

namespace kamaletdinov_quicksort_with_batcher_even_odd_merge {

KamaletdinovQuicksortWithBatcherEvenOddMergeSEQ::KamaletdinovQuicksortWithBatcherEvenOddMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KamaletdinovQuicksortWithBatcherEvenOddMergeSEQ::ValidationImpl() {
  return GetOutput().empty();
}

bool KamaletdinovQuicksortWithBatcherEvenOddMergeSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

std::pair<int, int> KamaletdinovQuicksortWithBatcherEvenOddMergeSEQ::PartitionRange(std::vector<int> &array, int left,
                                                                                    int right) {
  int i = left;
  int j = right;

  const int distance = right - left;
  const int mid = left + (distance / 2);
  const int pivot = array[mid];

  while (i <= j) {
    while (array[i] < pivot) {
      ++i;
    }
    while (array[j] > pivot) {
      --j;
    }
    if (i <= j) {
      std::swap(array[i], array[j]);
      ++i;
      --j;
    }
  }

  return std::make_pair(i, j);
}

void KamaletdinovQuicksortWithBatcherEvenOddMergeSEQ::PushPartitionsToStack(std::vector<std::pair<int, int>> &stack,
                                                                            int left, int right,
                                                                            const std::pair<int, int> &borders) {
  const int left_size = borders.second - left;
  const int right_size = right - borders.first;

  if (left_size > right_size) {
    if (left < borders.second) {
      stack.emplace_back(left, borders.second);
    }
    if (borders.first < right) {
      stack.emplace_back(borders.first, right);
    }
  } else {
    if (borders.first < right) {
      stack.emplace_back(borders.first, right);
    }
    if (left < borders.second) {
      stack.emplace_back(left, borders.second);
    }
  }
}

bool KamaletdinovQuicksortWithBatcherEvenOddMergeSEQ::RunImpl() {
  std::vector<int> array = GetInput();
  if (array.size() < 2) {
    GetOutput() = array;
    return true;
  }

  std::vector<std::pair<int, int>> stack;
  stack.emplace_back(0, static_cast<int>(array.size()) - 1);

  while (!stack.empty()) {
    const auto range = stack.back();
    stack.pop_back();

    const int left = range.first;
    const int right = range.second;

    if (left >= right) {
      continue;
    }

    const auto borders = PartitionRange(array, left, right);
    PushPartitionsToStack(stack, left, right, borders);
  }

  GetOutput().swap(array);
  return true;
}

bool KamaletdinovQuicksortWithBatcherEvenOddMergeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kamaletdinov_quicksort_with_batcher_even_odd_merge
