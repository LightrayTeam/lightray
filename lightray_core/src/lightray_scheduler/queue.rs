use crate::lightray_executor::executor::{LightrayExecutor, LightrayExecutorResult};
use crate::lightray_executor::model::LightrayModelId;
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
pub trait LightrayWorkQueue<T: LightrayExecutor> {
    async fn enqueue(
        &mut self,
        payload: TorchScriptInput,
        model_id: LightrayModelId,
    ) -> LightrayScheduledExecutionResult;
    fn worker_loop(&mut self);
    fn get_executor(&self) -> &T;
}
