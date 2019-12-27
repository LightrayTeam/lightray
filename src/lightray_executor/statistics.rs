use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

#[derive(Serialize, Deserialize, Debug)]
pub struct LightrayModelStatistic {
    /// Execution time of TorchScript model
    pub elapsed_execution_time: Duration,
    /// Total time spent in queue
    pub elapsed_queue_time: Duration,
    /// SystemTime of when object was first put into queue
    pub start_queue_time: SystemTime,
    /// SystemTime of when object began to execute
    pub start_execution_time: SystemTime,
    /// SystemTime of when object execution ended
    pub end_execution_time: SystemTime,
    /// Binary flag that indicates whether or not sample was run in a batched context
    pub batched_mode: bool,
}
