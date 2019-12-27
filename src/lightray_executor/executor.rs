use crate::lightray_executor::errors::LightrayModelVerificationError;
use crate::lightray_executor::model::{LightrayModel, LightrayModelId};
use crate::lightray_executor::statistics::LightrayModelExecutionStatistic;
use crate::lightray_torch::core::{SerializableIValue, TorchScriptInput};

pub struct LightrayExecutedExample {
    pub execution_statistic: LightrayModelExecutionStatistic,
    pub execution_result: SerializableIValue,
}

pub trait LightRayExecutor {
    fn execute(
        &self,
        model_id: LightrayModelId,
        example: TorchScriptInput,
    ) -> Result<LightrayExecutedExample, LightrayModelVerificationError>;

    fn register_model(&mut self, model: LightrayModel);
    fn delete_model(&mut self, model_id: LightrayModelId);
}
