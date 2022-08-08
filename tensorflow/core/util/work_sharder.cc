/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/util/work_sharder.h"

#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

/* ABSL_CONST_INIT */ thread_local int per_thread_max_parallism = 1000000;

void SetPerThreadMaxParallelism(int max_parallelism) {
  CHECK_LE(0, max_parallelism);
  per_thread_max_parallism = max_parallelism;
}

int GetPerThreadMaxParallelism() { return per_thread_max_parallism; }

inline int GetIntEnvVarOrZero(const char* name) {
  const char* val = getenv(name);
  if (!val) {
    return 0;
  }
  int res = std::stoi(val);
  LOG(INFO) << res;
  return res;
}

void Shard(int max_parallelism, thread::ThreadPool* workers, int64 total,
           int64 cost_per_unit, std::function<void(int64, int64)> work, CostRecorder* recorder) {
  // if recorder is nullptr, we use cost_per_unit as cost
  // if recorder is not nullptr, we use recorder as cost
  if (recorder) cost_per_unit = recorder->GetCost();
  CHECK_GE(total, 0);
  if (total == 0) {
    return;
  }
  max_parallelism = std::min(max_parallelism, GetPerThreadMaxParallelism());
  if (max_parallelism <= 1) {
    // Just inline the whole work since we only have 1 thread (core).
    work(0, total);
    return;
  }
  if (max_parallelism >= workers->NumThreads()) {
    if (recorder) cost_per_unit = recorder->GetEigenPoolCost();
    workers->ParallelFor(total, cost_per_unit, work);
    return;
  }
  Sharder::Do(total, cost_per_unit, work,
              [&workers](Sharder::Closure c) { workers->Schedule(c); },
              max_parallelism);
}

// DEPRECATED: Prefer threadpool->TransformRangeConcurrently, which allows you
// to directly specify the shard size.
void Sharder::Do(int64 total, int64 cost_per_unit, const Work& work,
                 const Runner& runner, int max_parallelism) {
  cost_per_unit = std::max(int64{1}, cost_per_unit);
  // We shard [0, total) into "num_shards" shards.
  //   1 <= num_shards <= num worker threads
  //
  // If total * cost_per_unit is small, it is not worth shard too
  // much. Let us assume each cost unit is 1ns, kMinCostPerShard=10000
  // is 10us.
  static const int64 kMinCostPerShard = GetIntEnvVarOrZero("MIN_COST_PER");
  const int num_shards =
      std::max<int>(1, std::min(static_cast<int64>(max_parallelism),
                                total * cost_per_unit / kMinCostPerShard));

  // Each shard contains up to "block_size" units. [0, total) is sharded
  // into:
  //   [0, block_size), [block_size, 2*block_size), ...
  // The 1st shard is done by the caller thread and the other shards
  // are dispatched to the worker threads. The last shard may be smaller than
  // block_size.
  const int64 block_size = (total + num_shards - 1) / num_shards;
  CHECK_GT(block_size, 0);  // total > 0 guarantees this.
  if (block_size >= total) {
    work(0, total);
    return;
  }
  const int num_shards_used = (total + block_size - 1) / block_size;
  BlockingCounter counter(num_shards_used - 1);
  for (int64 start = block_size; start < total; start += block_size) {
    auto limit = std::min(start + block_size, total);
    runner([&work, &counter, start, limit]() {
      work(start, limit);        // Compute the shard.
      counter.DecrementCount();  // The shard is done.
    });
  }

  // Inline execute the 1st shard.
  work(0, std::min(block_size, total));
  counter.Wait();
}

void TestShard(int64& times, int64& all_cost, int max_parallelism, thread::ThreadPool* workers, int64 total,
           int64 cost_per_unit, std::function<void(int64, int64)> work, CostRecorder* recorder) {
  static int STATIC_COST = GetIntEnvVarOrZero("STATIC_COST");
  auto start_time = std::chrono::high_resolution_clock::now();
  if (STATIC_COST != 0 || recorder == nullptr)
    Shard(max_parallelism,
          workers, total,
          cost_per_unit, work);
  else
    Shard(max_parallelism,
            workers, total,
            cost_per_unit, work, recorder);
  auto cost = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::high_resolution_clock::now() - start_time).count();
  times++;
  all_cost += cost;
  if(times % 100 == 0) {
    LOG(INFO) << recorder << " " << all_cost << " " << times << " " << recorder->GetCost();
    all_cost = 0;
  }
}

CostRecorder::CostRecorder(int64 default_cost)
  : record_cost_(default_cost), eigen_pool_param_(1.0) {
    stage_ = kStageWait;
    cost_vec_.resize(stage_times_);
}

void CostRecorder::StageWait(const int64& cost) {
  // avoid unstable operation during initialization
  if(record_times_ < stage_times_) {
    record_times_++;
    return;
  }
  record_times_ = 0;
  stage_ = kStageInit;
}

void CostRecorder::StageInit(const int64& cost) {
  // wait for a while and initialize record_cost_
  if(record_times_ < stage_times_) {
    cost_vec_[record_times_] = cost;
    record_times_++;
    return;
  }
  std::sort(cost_vec_.begin(), cost_vec_.end());
  int64 sum_cost = 0, num_cost = 0;
  for(auto pre_cost: cost_vec_) {
    if(pre_cost < 10 * cost_vec_[0]){
      sum_cost += pre_cost;
      num_cost++;
    }
  }
  record_cost_ = sum_cost / num_cost;
	cost_vec_.clear();
	cost_vec_.shrink_to_fit();
  stage_ = kStageWork;
}

void CostRecorder::StageWork(const int64& cost) {
  // dynamic recording unit cost
  if (cost > 10 * record_cost_) return;
  record_cost_ = 0.9 * record_cost_ + 0.1 * cost;
}

void CostRecorder::Stage(const int64& cost) {
  switch (stage_) {
    case StageType::kStageWork: {
      StageWork(cost);
      break;
    }
    case StageType::kStageInit: {
      StageInit(cost);
      break;
    }
    case StageType::kStageWait: {
      StageWait(cost);
      break;
    }
    default: break;
  }
}

void CostRecorder::SetStart() {
  start_time_ = clock::now();
}

int64 CostRecorder::GetCost() {
  return record_cost_;
}

int64 CostRecorder::GetEigenPoolCost() {
  return record_cost_ * eigen_pool_param_;
}

void CostRecorder::UnitUpdate(const int& unit_num) {
  // updata strategy
  auto real_cost = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            clock::now() - start_time_).count() / std::max(unit_num, 1);
  Stage(real_cost);
}

void CostRecorder::SetPoolParam(const double& param) {
  eigen_pool_param_ = param;
}

}  // end namespace tensorflow
