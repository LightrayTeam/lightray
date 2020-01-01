use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

#[derive(Serialize, Deserialize)]
pub struct SchedulerStatistics {
    pub time_spent_in_queue: Duration,
    pub start_time_in_queue: SystemTime,
    pub end_time_in_queue: SystemTime,
    pub number_of_elements_in_queue: usize,
}
