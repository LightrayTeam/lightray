use crate::lightray_executor::executor::LightrayExecutorResult;
use crate::lightray_executor::model::LightrayModelId;
use crate::lightray_torch::core::SerializableIValue;
use crate::lightray_torch::core::TorchScriptInput;
pub trait LightrayCaching {
    fn try_read(
        &mut self,
        model_id: &LightrayModelId,
        model_input: &TorchScriptInput,
    ) -> Option<LightrayExecutorResult>;
    fn hint_cache(
        &mut self,
        model_id: &LightrayModelId,
        model_input: &TorchScriptInput,
        model_output: &SerializableIValue,
    ) -> bool;
}
