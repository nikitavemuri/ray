// Copyright 2022 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <boost/circular_buffer.hpp>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "ray/common/asio/instrumented_io_context.h"
#include "ray/common/asio/periodical_runner.h"
#include "ray/common/id.h"
#include "ray/common/task/task_spec.h"
#include "ray/gcs/gcs_client/gcs_client.h"
#include "ray/util/counter_map.h"
#include "src/ray/protobuf/gcs.pb.h"
#include "src/ray/protobuf/export_api/export_task_event.pb.h"

namespace ray {
namespace core {

namespace worker {

using TaskAttempt = std::pair<TaskID, int32_t>;

/// A  wrapper class that will be converted to rpc::TaskEvents
///
/// This will be created by CoreWorker and stored in TaskEventBuffer, and
/// when it is being flushed periodically to GCS, it will be converted to
/// rpc::TaskEvents.
/// This is an optimization so that converting to protobuf (which is costly)
/// will not happen in the critical path of task execution/submission.
class TaskEvent {
 public:
  /// Constructor for Profile events
  explicit TaskEvent(TaskID task_id, JobID job_id, int32_t attempt_number);

  virtual ~TaskEvent() = default;

  /// Convert itself a rpc::TaskEvents
  ///
  /// NOTE: this method will modify internal states by moving fields to the
  /// rpc::TaskEvents.
  /// \param[out] rpc_task_events The rpc task event to be filled.
  virtual void ToRpcTaskEvents(rpc::TaskEvents *rpc_task_events) = 0;

  /// Convert itself a rpc::ExportTaskEventData
  ///
  /// NOTE: this method will modify internal states by moving fields to the
  /// rpc::ExportTaskEventData.
  /// \param[out] rpc_task_export_event_data The rpc export task event data to be filled.
  virtual void ToRpcTaskExportEvents(std::shared_ptr<rpc::ExportTaskEventData> rpc_task_export_event_data) = 0;

  /// If it is a profile event.
  virtual bool IsProfileEvent() const = 0;

  virtual TaskAttempt GetTaskAttempt() const {
    return std::make_pair(task_id_, attempt_number_);
  }

 protected:
  /// Task Id.
  const TaskID task_id_ = TaskID::Nil();
  /// Job id.
  const JobID job_id_ = JobID::Nil();
  /// Attempt number
  const int32_t attempt_number_ = -1;
};

/// TaskStatusEvent is generated when a task changes its status.
class TaskStatusEvent : public TaskEvent {
 public:
  /// A class that contain data that will be converted to rpc::TaskStateUpdate
  struct TaskStateUpdate {
    TaskStateUpdate() {}

    TaskStateUpdate(const absl::optional<const rpc::RayErrorInfo> &error_info)
        : error_info_(error_info) {}

    TaskStateUpdate(const NodeID &node_id, const WorkerID &worker_id)
        : node_id_(node_id), worker_id_(worker_id) {}

    TaskStateUpdate(const rpc::TaskLogInfo &task_log_info)
        : task_log_info_(task_log_info) {}

    TaskStateUpdate(const std::string &actor_repr_name, uint32_t pid)
        : actor_repr_name_(actor_repr_name), pid_(pid) {}

    TaskStateUpdate(uint32_t pid) : pid_(pid) {}

    TaskStateUpdate(bool is_debugger_paused) : is_debugger_paused_(is_debugger_paused) {}

   private:
    friend class TaskStatusEvent;

    /// Node id if it's a SUBMITTED_TO_WORKER status change.
    const absl::optional<NodeID> node_id_ = absl::nullopt;
    /// Worker id if it's a SUBMITTED_TO_WORKER status change.
    const absl::optional<WorkerID> worker_id_ = absl::nullopt;
    /// Task error info.
    const absl::optional<rpc::RayErrorInfo> error_info_ = absl::nullopt;
    /// Task log info.
    const absl::optional<rpc::TaskLogInfo> task_log_info_ = absl::nullopt;
    /// Actor task repr name.
    const std::string actor_repr_name_ = "";
    /// Worker's pid if it's a RUNNING status change.
    const absl::optional<uint32_t> pid_ = absl::nullopt;
    /// Is the task is paused by the debugger.
    const absl::optional<bool> is_debugger_paused_ = absl::nullopt;
  };

  explicit TaskStatusEvent(
      TaskID task_id,
      JobID job_id,
      int32_t attempt_number,
      const rpc::TaskStatus &task_status,
      int64_t timestamp,
      const std::shared_ptr<const TaskSpecification> &task_spec = nullptr,
      absl::optional<const TaskStateUpdate> state_update = absl::nullopt);

  void ToRpcTaskEvents(rpc::TaskEvents *rpc_task_events) override;

  void ToRpcTaskExportEvents(std::shared_ptr<rpc::ExportTaskEventData> rpc_task_export_event_data) override;

  bool IsProfileEvent() const override { return false; }

 private:
  /// The task status change if it's a status change event.
  const rpc::TaskStatus task_status_ = rpc::TaskStatus::NIL;
  /// The time when the task status change happens.
  const int64_t timestamp_ = -1;
  /// Pointer to the task spec.
  const std::shared_ptr<const TaskSpecification> task_spec_ = nullptr;
  /// Optional task state update
  const absl::optional<const TaskStateUpdate> state_update_ = absl::nullopt;
};

/// TaskProfileEvent is generated when `RAY_enable_timeline` is on.
class TaskProfileEvent : public TaskEvent {
 public:
  explicit TaskProfileEvent(TaskID task_id,
                            JobID job_id,
                            int32_t attempt_number,
                            const std::string &component_type,
                            const std::string &component_id,
                            const std::string &node_ip_address,
                            const std::string &event_name,
                            int64_t start_time);

  void ToRpcTaskEvents(rpc::TaskEvents *rpc_task_events) override;

  void ToRpcTaskExportEvents(std::shared_ptr<rpc::ExportTaskEventData> rpc_task_export_event_data) override;

  bool IsProfileEvent() const override { return true; }

  void SetEndTime(int64_t end_time) { end_time_ = end_time; }

  void SetExtraData(const std::string &extra_data) { extra_data_ = extra_data; }

 private:
  /// The below fields mirror rpc::ProfileEvent
  const std::string component_type_;
  const std::string component_id_;
  const std::string node_ip_address_;
  const std::string event_name_;
  const int64_t start_time_;
  int64_t end_time_;
  std::string extra_data_;
};

/// @brief An enum class defining counters to be used in TaskEventBufferImpl.
enum TaskEventBufferCounter {
  kNumTaskProfileEventDroppedSinceLastFlush,
  kNumTaskStatusEventDroppedSinceLastFlush,
  kNumTaskProfileEventsStored,
  kNumTaskStatusEventsStored,
  kNumDroppedTaskAttemptsStored,
  kTotalNumTaskProfileEventDropped,
  kTotalNumTaskStatusEventDropped,
  kTotalNumTaskAttemptsReported,
  kTotalNumLostTaskAttemptsReported,
  kTotalTaskEventsBytesReported,
  kTotalNumFailedToReport,
};

/// An interface for a buffer that stores task status changes and profiling events,
/// and reporting these events to the GCS periodically.
///
/// Dropping of task events
/// ========================
/// Task events will be lost in the below cases for now:
///   1. If any of the gRPC call failed, the task events will be dropped and warnings
///   logged. This is probably fine since this usually indicated a much worse issue.
///
///   2. More than `RAY_task_events_max_num_status_events_buffer_on_worker` tasks have
///   been stored in the buffer, any new task events will be dropped. In this case, the
///   number of dropped task events will also be included in the next flush to surface
///   this.
///
/// No overloading of GCS
/// =====================
/// If GCS failed to respond quickly enough to the previous report, reporting of events to
/// GCS will be delayed until GCS replies the gRPC in future intervals.
class TaskEventBuffer {
 public:
  virtual ~TaskEventBuffer() = default;

  /// Add a task event to be reported.
  ///
  /// \param task_events Task events.
  virtual void AddTaskEvent(std::unique_ptr<TaskEvent> task_event) = 0;

  /// Flush all task events stored in the buffer to GCS.
  ///
  /// This function will be called periodically configured by
  /// `RAY_task_events_report_interval_ms`, and send task events stored in a buffer to
  /// GCS. If GCS has not responded to a previous flush, it will defer the flushing to
  /// the next interval (if not forced.)
  ///
  /// Before flushing to GCS, events from a single task attempt will also be coalesced
  /// into one rpc::TaskEvents as an optimization.
  ///
  /// \param forced When set to true, buffered events will be sent to GCS even if GCS has
  ///       not responded to the previous flush. A forced flush will be called before
  ///       CoreWorker disconnects to ensure all task events in the buffer are sent.
  virtual void FlushEvents(bool forced) = 0;

  /// Start the TaskEventBuffer.
  ///
  /// Connects the GCS client, starts its io_thread, and sets up periodical runner for
  /// flushing events to GCS.
  /// When it returns non ok status, the TaskEventBuffer will be disabled, and call to
  /// Enabled() will return false.
  ///
  /// \param auto_flush Test only flag to disable periodical flushing events if false.
  /// \return Status code. When the status is not ok, events will not be recorded nor
  /// reported.
  virtual Status Start(bool auto_flush = true) = 0;

  /// Stop the TaskEventBuffer and it's underlying IO, disconnecting GCS clients.
  virtual void Stop() = 0;

  /// Return true if recording and reporting of task events is enabled.
  ///
  /// The TaskEventBuffer will be disabled if Start() returns not ok.
  virtual bool Enabled() const = 0;

  /// Return a string that describes the task event buffer stats.
  virtual const std::string DebugString() = 0;
};

/// Implementation of TaskEventBuffer.
///
/// The buffer has its own io_context and io_thread, that's isolated from other
/// components.
///
/// This class is thread-safe.
class TaskEventBufferImpl : public TaskEventBuffer {
 public:
  /// Constructor
  ///
  /// \param gcs_client GCS client
  TaskEventBufferImpl(std::shared_ptr<gcs::GcsClient> gcs_client);

  ~TaskEventBufferImpl() override;

  void AddTaskEvent(std::unique_ptr<TaskEvent> task_event)
      ABSL_LOCKS_EXCLUDED(mutex_) override;

  void FlushEvents(bool forced) ABSL_LOCKS_EXCLUDED(mutex_) override;

  Status Start(bool auto_flush = true) ABSL_LOCKS_EXCLUDED(mutex_) override;

  void Stop() ABSL_LOCKS_EXCLUDED(mutex_) override;

  bool Enabled() const override;

  const std::string DebugString() override;

 private:
  /// Add a task status event to be reported.
  ///
  /// \param status_event Task status event.
  void AddTaskStatusEvent(std::unique_ptr<TaskEvent> status_event)
      ABSL_LOCKS_EXCLUDED(mutex_);

  /// Add a task profile event to be reported.
  ///
  /// \param profile_event Task profile event.
  void AddTaskProfileEvent(std::unique_ptr<TaskEvent> profile_event)
      ABSL_LOCKS_EXCLUDED(profile_mutex_);

  /// Get data related to task status events to be send to GCS.
  ///
  /// \param[out] status_events_to_send Task status events to be sent.
  /// \param[out] dropped_task_attempts_to_send Task attempts that were dropped due to
  ///             status events being dropped.
  void GetTaskStatusEventsToSend(
      std::vector<std::unique_ptr<TaskEvent>> *status_events_to_send,
      absl::flat_hash_set<TaskAttempt> *dropped_task_attempts_to_send)
      ABSL_LOCKS_EXCLUDED(mutex_);

  /// Get data related to task profile events to be send to GCS.
  ///
  /// \param[out] profile_events_to_send Task profile events to be sent.
  void GetTaskProfileEventsToSend(
      std::vector<std::unique_ptr<TaskEvent>> *profile_events_to_send)
      ABSL_LOCKS_EXCLUDED(profile_mutex_);

  /// Get the task events to GCS.
  ///
  /// \param status_events_to_send Task status events to be sent.
  /// \param profile_events_to_send Task profile events to be sent.
  /// \param dropped_task_attempts_to_send Task attempts that were dropped due to
  ///        status events being dropped.
  /// \return A unique_ptr to rpc::TaskEvents to be sent to GCS.
  std::unique_ptr<rpc::TaskEventData> CreateDataToSend(
      std::vector<std::unique_ptr<TaskEvent>> &&status_events_to_send,
      std::vector<std::unique_ptr<TaskEvent>> &&profile_events_to_send,
      absl::flat_hash_set<TaskAttempt> &&dropped_task_attempts_to_send);
  
  /// Write export task events.
  ///
  /// \param status_events_to_send Task status events to be written.
  /// \param profile_events_to_send Task profile events to be written.
  /// \param dropped_task_attempts_to_send Task attempts that were dropped due to
  ///        status events being dropped.
  void WriteExportData(
      std::vector<std::unique_ptr<TaskEvent>> &&status_events_to_send,
      std::vector<std::unique_ptr<TaskEvent>> &&profile_events_to_send,
      absl::flat_hash_set<TaskAttempt> &&dropped_task_attempts_to_send);

  /// Reset the counters during flushing data to GCS.
  void ResetCountersForFlush();

  /// Test only functions.
  size_t GetNumTaskEventsStored() {
    return stats_counter_.Get(TaskEventBufferCounter::kNumTaskStatusEventsStored) +
           stats_counter_.Get(TaskEventBufferCounter::kNumTaskProfileEventsStored);
  }

  /// Test only functions.
  size_t GetTotalNumStatusTaskEventsDropped() {
    return stats_counter_.Get(TaskEventBufferCounter::kTotalNumTaskStatusEventDropped);
  }

  /// Test only functions.
  size_t GetNumStatusTaskEventsDroppedSinceLastFlush() {
    return stats_counter_.Get(
        TaskEventBufferCounter::kNumTaskStatusEventDroppedSinceLastFlush);
  }

  /// Test only functions.
  size_t GetTotalNumProfileTaskEventsDropped() {
    return stats_counter_.Get(TaskEventBufferCounter::kTotalNumTaskProfileEventDropped);
  }

  /// Test only functions.
  size_t GetNumProfileTaskEventsDroppedSinceLastFlush() {
    return stats_counter_.Get(
        TaskEventBufferCounter::kNumTaskProfileEventDroppedSinceLastFlush);
  }

  /// Test only functions.
  size_t GetNumFailedToReport() {
    return stats_counter_.Get(TaskEventBufferCounter::kTotalNumFailedToReport);
  }

  /// Test only functions.
  gcs::GcsClient *GetGcsClient() {
    absl::MutexLock lock(&mutex_);
    return gcs_client_.get();
  }

  /// Mutex guarding task_events_data_.
  absl::Mutex mutex_;

  absl::Mutex profile_mutex_;

  /// IO service event loop owned by TaskEventBuffer.
  instrumented_io_context io_service_;

  /// Work guard to prevent the io_context from exiting when no work.
  boost::asio::executor_work_guard<boost::asio::io_context::executor_type> work_guard_;

  /// Dedicated io thread for running the periodical runner and the GCS client.
  std::thread io_thread_;

  /// The runner to run function periodically.
  PeriodicalRunner periodical_runner_;

  /// Client to the GCS used to push profile events to it.
  std::shared_ptr<gcs::GcsClient> gcs_client_ ABSL_GUARDED_BY(mutex_);

  /// True if the TaskEventBuffer is enabled.
  std::atomic<bool> enabled_ = false;

  /// Circular buffered task status events.
  boost::circular_buffer<std::unique_ptr<TaskEvent>> status_events_
      ABSL_GUARDED_BY(mutex_);

  /// Buffered task attempts that were dropped due to status events being dropped.
  /// This will be sent to GCS to surface the dropped task attempts.
  absl::flat_hash_set<TaskAttempt> dropped_task_attempts_unreported_
      ABSL_GUARDED_BY(mutex_);

  /// Buffered task profile events. A FIFO queue to be sent to GCS.
  absl::flat_hash_map<TaskAttempt, std::vector<std::unique_ptr<TaskEvent>>>
      profile_events_ ABSL_GUARDED_BY(profile_mutex_);

  /// Stats counter map.
  CounterMapThreadSafe<TaskEventBufferCounter> stats_counter_;

  /// True if there's a pending gRPC call. It's a simple way to prevent overloading
  /// GCS with too many calls. There is no point sending more events if GCS could not
  /// process them quick enough.
  std::atomic<bool> grpc_in_progress_ = false;

  FRIEND_TEST(TaskEventBufferTestManualStart, TestGcsClientFail);
  FRIEND_TEST(TaskEventBufferTestBatchSend, TestBatchedSend);
  FRIEND_TEST(TaskEventBufferTest, TestAddEvent);
  FRIEND_TEST(TaskEventBufferTest, TestFlushEvents);
  FRIEND_TEST(TaskEventBufferTest, TestFailedFlush);
  FRIEND_TEST(TaskEventBufferTest, TestBackPressure);
  FRIEND_TEST(TaskEventBufferTest, TestForcedFlush);
  FRIEND_TEST(TaskEventBufferTestLimitBuffer, TestBufferSizeLimitStatusEvents);
  FRIEND_TEST(TaskEventBufferTestLimitProfileEvents, TestBufferSizeLimitProfileEvents);
  FRIEND_TEST(TaskEventBufferTestLimitProfileEvents, TestLimitProfileEventsPerTask);
};

inline void FillExportTaskInfo(rpc::ExportTaskEventData::TaskInfoEntry *task_info,
                         const TaskSpecification &task_spec) {
  rpc::TaskType type;
  if (task_spec.IsNormalTask()) {
    type = rpc::TaskType::NORMAL_TASK;
  } else if (task_spec.IsDriverTask()) {
    type = rpc::TaskType::DRIVER_TASK;
  } else if (task_spec.IsActorCreationTask()) {
    type = rpc::TaskType::ACTOR_CREATION_TASK;
    task_info->set_actor_id(task_spec.ActorCreationId().Binary());
  } else {
    RAY_CHECK(task_spec.IsActorTask());
    type = rpc::TaskType::ACTOR_TASK;
    task_info->set_actor_id(task_spec.ActorId().Binary());
  }
  task_info->set_type(type);
  task_info->set_language(task_spec.GetLanguage());
  task_info->set_func_or_class_name(task_spec.FunctionDescriptor()->CallString());

  task_info->set_task_id(task_spec.TaskId().Binary());
  // NOTE: we set the parent task id of a task to be submitter's task id, where
  // the submitter depends on the owner coreworker's:
  // - if the owner coreworker runs a normal task, the submitter's task id is the task id.
  // - if the owner coreworker runs an actor, the submitter's task id will be the actor's
  // creation task id.
  task_info->set_parent_task_id(task_spec.SubmitterTaskId().Binary());
  const auto &resources_map = task_spec.GetRequiredResources().GetResourceMap();
  task_info->mutable_required_resources()->insert(resources_map.begin(),
                                                  resources_map.end());

  auto export_runtime_env_info = task_info->mutable_runtime_env_info();
  export_runtime_env_info->set_serialized_runtime_env(task_spec.RuntimeEnvInfo().serialized_runtime_env());
  auto export_runtime_env_uris = export_runtime_env_info->mutable_uris();
  export_runtime_env_uris->set_working_dir_uri(task_spec.RuntimeEnvInfo().uris().working_dir_uri());
  export_runtime_env_uris->mutable_py_modules_uris()->CopyFrom(
    task_spec.RuntimeEnvInfo().uris().py_modules_uris()
  );
  auto export_runtime_env_config = export_runtime_env_info->mutable_runtime_env_config();
  export_runtime_env_config->set_setup_timeout_seconds(task_spec.RuntimeEnvInfo().runtime_env_config().setup_timeout_seconds());
  export_runtime_env_config->set_eager_install(task_spec.RuntimeEnvInfo().runtime_env_config().eager_install());
  export_runtime_env_config->mutable_log_files()->CopyFrom(task_spec.RuntimeEnvInfo().runtime_env_config().log_files());
  
  const auto &pg_id = task_spec.PlacementGroupBundleId().first;
  if (!pg_id.IsNil()) {
    task_info->set_placement_group_id(pg_id.Binary());
  }
}

/// Fill the rpc::ExportTaskEventData::TaskStateUpdate with the timestamps according to the status change.
///
/// \param task_status The task status.
/// \param timestamp The timestamp.
/// \param[out] state_updates The state updates with timestamp to be updated.
inline void FillExportTaskStatusUpdateTime(const ray::rpc::TaskStatus &task_status,
                                     int64_t timestamp,
                                     rpc::ExportTaskEventData::TaskStateUpdate *state_updates) {
  if (task_status == rpc::TaskStatus::NIL) {
    // Not status change.
    return;
  }
  (*state_updates->mutable_state_ts_ns())[task_status] = timestamp;
}

inline void TaskLogInfoToExport(const rpc::TaskLogInfo& src, rpc::ExportTaskEventData::TaskLogInfo* dest) {
      dest->set_stdout_file(src.stdout_file());
      dest->set_stderr_file(src.stderr_file());
      dest->set_stdout_start(src.stdout_start());
      dest->set_stdout_end(src.stdout_end());
      dest->set_stderr_start(src.stderr_start());
      dest->set_stderr_end(src.stderr_end());
  }

}  // namespace worker

}  // namespace core
}  // namespace ray
