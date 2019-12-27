use crate::lightray_executor::errors::{LightrayMissingSamples, LightrayModelVerificationError};
use crate::lightray_torch::core::{TorchScriptGraph, TorchScriptInput};
use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub struct LightrayModelId {
    pub model_id: u64,
    pub model_version: u16,
}
pub struct LightrayModel {
    pub id: LightrayModelId,
    pub executor: TorchScriptGraph,
    pub samples: Vec<TorchScriptInput>,
}
impl LightrayModel {
    pub fn new(
        id: LightrayModelId,
        executor: TorchScriptGraph,
        samples: Vec<TorchScriptInput>,
    ) -> LightrayModel {
        return LightrayModel {
            id: id,
            samples: samples,
            executor: executor,
        };
    }
    pub fn verify(&self) -> Result<(), LightrayModelVerificationError> {
        if self.samples.len() == 0 {
            return Err(LightrayModelVerificationError::LightrayMissingSamples(
                LightrayMissingSamples {},
            ));
        }
        for sample in &self.samples {
            if let Err(err) = self.executor.forward(&sample) {
                return Err(LightrayModelVerificationError::InternalTorchError(err));
            }
        }
        Ok(())
    }
    pub fn warmup_jit(&self, warmup_count: u16) -> Result<(), LightrayModelVerificationError> {
        let _out = self.verify()?;
        let mut counter = 0;
        loop {
            for sample in &self.samples {
                if counter >= warmup_count {
                    return Ok(());
                }
                if let Err(err) = self.executor.forward(&sample) {
                    return Err(LightrayModelVerificationError::InternalTorchError(err));
                }
                counter += 1;
            }
        }
    }
}
