use crate::lightray_executor::executor::{LightrayExecutor, LightrayExecutorResult};
use crate::lightray_executor::model::LightrayModelId;
use crate::lightray_scheduler::errors::LightraySchedulerError;
use crate::lightray_scheduler::queue::{LightrayScheduledExecutionResult, LightrayWorkQueue};
use crate::lightray_scheduler::statistics::SchedulerStatistics;
use crate::lightray_torch::core::TorchScriptInput;
use async_trait::async_trait;
use crossbeam_queue::SegQueue;
use std::thread;
use std::time::{Instant, SystemTime};
use tokio::sync::oneshot::{channel, Receiver, Sender};
pub struct ChannelBasedWork {
    payload: TorchScriptInput,
    model_id: LightrayModelId,
    sender: Sender<LightrayExecutorResult>,
}
pub struct LightrayFIFOWorkQueue<T: LightrayExecutor> {
    worker_queue: SegQueue<ChannelBasedWork>,
    worker_executor: T,
    verify_model_input: bool,
}

impl<T: LightrayExecutor> LightrayFIFOWorkQueue<T> {
    pub fn new(worker_executor: T, verify_model_input: bool) -> LightrayFIFOWorkQueue<T> {
        LightrayFIFOWorkQueue::<T> {
            worker_queue: SegQueue::new(),
            worker_executor,
            verify_model_input,
        }
    }
}

#[async_trait(?Send)]
impl<T: LightrayExecutor> LightrayWorkQueue<T> for LightrayFIFOWorkQueue<T> {
    async fn enqueue(
        &mut self,
        payload: TorchScriptInput,
        model_id: LightrayModelId,
    ) -> LightrayScheduledExecutionResult {
        let (tx, rx): (
            Sender<LightrayExecutorResult>,
            Receiver<LightrayExecutorResult>,
        ) = channel();
        let work: ChannelBasedWork = ChannelBasedWork {
            payload,
            model_id,
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
                let executed_value = self.worker_executor.execute(
                    &value.model_id,
                    &value.payload,
                    self.verify_model_input,
                );
                let _x = value.sender.send(executed_value);
            } else {
                // Otherwise, this will be an incredibly tight loop which might end up taking a whole core.
                // To stop this from happening we yield the thread when the queue is empty.
                thread::yield_now();
            }
        }
    }
    fn get_executor(&self) -> &T {
        &self.worker_executor
    }
}
