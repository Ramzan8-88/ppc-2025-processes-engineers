#include "kamaletdinov_r_qsort_batcher_oddeven_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include "kamaletdinov_r_qsort_batcher_oddeven_merge/common/include/common.hpp"

namespace kamaletdinov_quicksort_with_batcher_even_odd_merge {

KamaletdinovQuicksortWithBatcherEvenOddMergeMPI::KamaletdinovQuicksortWithBatcherEvenOddMergeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KamaletdinovQuicksortWithBatcherEvenOddMergeMPI::ValidationImpl() {
  return GetOutput().empty();
}

bool KamaletdinovQuicksortWithBatcherEvenOddMergeMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

namespace {

int ChoosePivotIndex(int left, int right) {
  return left + ((right - left) / 2);
}

std::pair<int, int> PartitionBlock(std::vector<int> &data, int left, int right) {
  int i = left;
  int j = right;
  const int pivot_value = data[ChoosePivotIndex(left, right)];

  while (i <= j) {
    while (data[i] < pivot_value) {
      ++i;
    }
    while (data[j] > pivot_value) {
      --j;
    }
    if (i <= j) {
      std::swap(data[i++], data[j--]);
    }
  }
  return {i, j};
}

void IterativeQuickSort(std::vector<int> &data) {
  if (data.size() < 2) {
    return;
  }
  std::vector<std::pair<int, int>> stack{{0, static_cast<int>(data.size() - 1)}};

  while (!stack.empty()) {
    auto [left, right] = stack.back();
    stack.pop_back();
    if (left >= right) {
      continue;
    }

    auto [i, j] = PartitionBlock(data, left, right);
    if (left < j) {
      stack.emplace_back(left, j);
    }
    if (i < right) {
      stack.emplace_back(i, right);
    }
  }
}

void MergeKeepPart(std::vector<int> &local, const std::vector<int> &received, bool keep_low_part) {
  std::vector<int> merged(local.size() + received.size());
  std::ranges::merge(local, received, merged.begin());
  if (keep_low_part) {
    std::copy_n(merged.begin(), local.size(), local.begin());
  } else {
    std::copy_n(merged.end() - static_cast<std::ptrdiff_t>(local.size()), local.size(), local.begin());
  }
}

// Unused helpers kept for possible future use; silenced to satisfy -Wunused-function
[[maybe_unused]] void PerformEvenPhaseExchange(std::vector<int> & /*local*/, int /*rank*/, int /*size*/,
                                               bool is_even_rank, bool has_next, bool has_prev) {
  if (is_even_rank && has_next) {
    // NeighborExchange call will be made in caller
  }
  if (!is_even_rank && has_prev) {
    // NeighborExchange call will be made in caller
  }
}

[[maybe_unused]] void PerformOddPhaseExchange(std::vector<int> & /*local*/, int /*rank*/, int /*size*/,
                                              bool is_even_rank, bool has_next, bool has_prev) {
  if (!is_even_rank && has_next) {
    // NeighborExchange call will be made in caller
  }
  if (is_even_rank && has_prev) {
    // NeighborExchange call will be made in caller
  }
}

void NeighborExchangeStatic(std::vector<int> &local, int partner_rank, bool keep_lower) {
  const int send_size = static_cast<int>(local.size());
  int recv_size = 0;
  MPI_Sendrecv(&send_size, 1, MPI_INT, partner_rank, 0, &recv_size, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  std::vector<int> recv_buffer(recv_size);
  const int *send_ptr = (send_size != 0) ? local.data() : nullptr;
  int *recv_ptr = (recv_size != 0) ? recv_buffer.data() : nullptr;

  MPI_Sendrecv(send_ptr, send_size, MPI_INT, partner_rank, 1, recv_ptr, recv_size, MPI_INT, partner_rank, 1,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (recv_size > 0) {
    MergeKeepPart(local, recv_buffer, keep_lower);
  }
}

}  // namespace

void KamaletdinovQuicksortWithBatcherEvenOddMergeMPI::
    NeighborExchange(  // NOLINT(readability-convert-member-functions-to-static)
        std::vector<int> &local, int partner_rank, bool keep_lower) {
  NeighborExchangeStatic(local, partner_rank, keep_lower);
}

void KamaletdinovQuicksortWithBatcherEvenOddMergeMPI::BatcherPhases(std::vector<int> &local, int rank, int size,
                                                                    int global_size) {
  int min_block = std::max(1, global_size / size);
  const int phase_count = (global_size + min_block - 1) / min_block;
  const bool is_even_rank = (rank % 2 == 0);
  const bool has_next = (rank + 1 < size);
  const bool has_prev = (rank - 1 >= 0);

  for (int phase = 0; phase < phase_count; ++phase) {
    const bool even_phase = (phase % 2 == 0);

    // Direction: +1 -> talk to rank+1 (keep lower part), -1 -> talk to rank-1 (keep upper part)
    const int direction = (even_phase == is_even_rank) ? 1 : -1;
    const bool can_communicate = (direction > 0 && has_next) || (direction < 0 && has_prev);
    if (!can_communicate) {
      continue;
    }

    const int partner = rank + direction;
    const bool keep_lower = direction > 0;
    NeighborExchange(local, partner, keep_lower);
  }
}

void KamaletdinovQuicksortWithBatcherEvenOddMergeMPI::BroadcastOutputToAllRanks() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int total_size = rank == 0 ? static_cast<int>(GetOutput().size()) : 0;
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput().resize(static_cast<std::size_t>(total_size));
  }

  MPI_Bcast(total_size > 0 ? GetOutput().data() : nullptr, total_size, MPI_INT, 0, MPI_COMM_WORLD);
}

bool KamaletdinovQuicksortWithBatcherEvenOddMergeMPI::RunImpl() {
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int global_size = rank == 0 ? static_cast<int>(GetInput().size()) : 0;
  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (global_size == 0) {
    if (rank == 0) {
      GetOutput().clear();
    }
    return true;
  }

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  for (int i = 0; i < size; ++i) {
    counts[i] = (global_size / size) + (i < global_size % size ? 1 : 0);
  }
  std::partial_sum(counts.begin(), counts.end() - 1, displs.begin() + 1);

  const int local_count = counts[rank];
  std::vector<int> local_data(local_count);

  const int *send_ptr = (rank == 0 && !GetInput().empty()) ? GetInput().data() : nullptr;
  MPI_Scatterv(send_ptr, counts.data(), displs.data(), MPI_INT, (local_count != 0) ? local_data.data() : nullptr,
               local_count, MPI_INT, 0, MPI_COMM_WORLD);

  IterativeQuickSort(local_data);
  BatcherPhases(local_data, rank, size, global_size);

  if (rank == 0) {
    GetOutput().resize(static_cast<std::size_t>(global_size));
  }
  MPI_Gatherv((local_count != 0) ? local_data.data() : nullptr, local_count, MPI_INT,
              rank == 0 ? GetOutput().data() : nullptr, counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  BroadcastOutputToAllRanks();
  return true;
}

bool KamaletdinovQuicksortWithBatcherEvenOddMergeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kamaletdinov_quicksort_with_batcher_even_odd_merge
