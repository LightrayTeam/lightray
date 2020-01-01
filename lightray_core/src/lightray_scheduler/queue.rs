use crate::lightray_executor::executor::LightrayExecutorResult;
use crate::lightray_scheduler::errors::LightraySchedulerError;
use crate::lightray_scheduler::statistics::SchedulerStatistics;
use crate::lightray_torch::core::TorchScriptInput;

use async_trait::async_trait;

pub struct LightrayScheduledExecutionResult {
    pub execution_result: Option<LightrayExecutorResult>,
    pub scheduler_error: Option<LightraySchedulerError>,
    pub scheduler_metrics: SchedulerStatistics,
}
#[async_trait(?Send)]
pub trait LightrayWorkQueue {
    async fn enqueue(&mut self, payload: TorchScriptInput) -> LightrayScheduledExecutionResult;
    fn worker_loop(&mut self);
}
