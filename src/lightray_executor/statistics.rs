use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

#[derive(Serialize, Deserialize, Debug)]
pub struct LightrayModelExecutionStatistic {
    /// Execution time of TorchScript model
    pub elapsed_execution_time: Duration,
    /// SystemTime of when object was first put into queue
    pub start_execution_time: SystemTime,
    /// SystemTime of when object execution ended
    pub end_execution_time: SystemTime,
}
