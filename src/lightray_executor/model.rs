use crate::lightray_torch::core::{TorchScriptGraph, TorchScriptInput};
use crate::lightray_torch::errors::InternalTorchError;
use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
struct LightrayModelId {
    model_id: u64,
    model_version: u16,
}
struct LightrayModel {
    id: LightrayModelId,
    executor: TorchScriptGraph,
    samples: Option<Vec<TorchScriptInput>>,
}
impl LightrayModel {
    fn verify(&self) -> Option<InternalTorchError> {
        if let Some(unwrapped_samples) = &self.samples {
            for sample in unwrapped_samples {
                let model_output = self.executor.forward(&sample);
                match model_output {
                    Err(e) => {
                        return Some(e);
                    }
                    _ => continue,
                }
            }
            return None;
        }
        None
    }
}
