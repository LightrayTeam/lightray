use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Serialize, Deserialize, Debug)]
pub struct LightrayModelStatistic {
    elapsed_execution_time: Duration,
    start_execution_time: Instant,
    end_execution_time: Instant,
}
