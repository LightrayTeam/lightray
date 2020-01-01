use crate::lightray_executor::executor::LightrayExecutorResult;
use crate::lightray_scheduler::errors::LightraySchedulerError;
use crate::lightray_scheduler::queue::{LightrayScheduledExecutionResult, LightrayWorkQueue};
use crate::lightray_scheduler::statistics::SchedulerStatistics;
use crate::lightray_torch::core::TorchScriptInput;
use async_trait::async_trait;
use crossbeam_queue::SegQueue;
use std::time::{Instant, SystemTime};
use tokio::sync::oneshot::{channel, Receiver, Sender};
pub struct ChannelBasedWork {
    payload: TorchScriptInput,
    sender: Sender<LightrayExecutorResult>,
}
pub struct LightrayFIFOWorkQueue {
    worker_queue: SegQueue<ChannelBasedWork>,
    worker_function: fn(&TorchScriptInput) -> LightrayExecutorResult,
}

impl LightrayFIFOWorkQueue {
    pub fn new(
        worker_function: fn(&TorchScriptInput) -> LightrayExecutorResult,
    ) -> LightrayFIFOWorkQueue {
        LightrayFIFOWorkQueue {
            worker_queue: SegQueue::new(),
            worker_function,
        }
    }
}

#[async_trait(?Send)]
impl LightrayWorkQueue for LightrayFIFOWorkQueue {
    async fn enqueue(&mut self, payload: TorchScriptInput) -> LightrayScheduledExecutionResult {
        let (tx, rx): (
            Sender<LightrayExecutorResult>,
            Receiver<LightrayExecutorResult>,
        ) = channel();
        let work: ChannelBasedWork = ChannelBasedWork {
            payload,
            sender: tx,
        };
        let queue_instance_start_time = Instant::now();
        let queue_start_time = SystemTime::now();

        self.worker_queue.push(work);
        let execution_result = rx.await;

        let queue_instance_end_time = Instant::now();
        let queue_end_time = SystemTime::now();

        let metrics = SchedulerStatistics {
            time_spent_in_queue: queue_instance_end_time - queue_instance_start_time,
            start_time_in_queue: queue_start_time,
            end_time_in_queue: queue_end_time,
            number_of_elements_in_queue: self.worker_queue.len(),
        };

        match execution_result {
            Ok(executor_result) => LightrayScheduledExecutionResult {
                execution_result: Some(executor_result),
                scheduler_error: None,
                scheduler_metrics: metrics,
            },
            Err(_) => LightrayScheduledExecutionResult {
                execution_result: None,
                scheduler_error: Some(LightraySchedulerError::SchedulerError),
                scheduler_metrics: metrics,
            },
        }
    }
    fn worker_loop(&mut self) {
        loop {
            if let Ok(value) = self.worker_queue.pop() {
                let executed_value = (self.worker_function)(&value.payload);
                let _x = value.sender.send(executed_value);
            }
        }
    }
}
