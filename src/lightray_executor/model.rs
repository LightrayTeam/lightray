use crate::lightray_executor::errors::{
    LightrayMissingSamples, LightrayModelExecutionError, LightrayModelVerificationError,
};
use crate::lightray_executor::semantics::LightrayModelSemantics;
use crate::lightray_torch::core::{SerializableIValue, TorchScriptGraph, TorchScriptInput};

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
    pub semantics: LightrayModelSemantics,
}
impl LightrayModel {
    pub fn new(
        id: LightrayModelId,
        executor: TorchScriptGraph,
        samples: Vec<TorchScriptInput>,
        semantics: LightrayModelSemantics,
    ) -> Result<LightrayModel, LightrayModelVerificationError> {
        let model = LightrayModel {
            id,
            samples,
            executor,
            semantics,
        };
        model.verify()?;
        Ok(model)
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

    pub fn execute(
        &self,
        input: &TorchScriptInput,
        do_semantic_verification: bool,
    ) -> Result<SerializableIValue, LightrayModelExecutionError> {
        if do_semantic_verification {
            if let Err(x) = self.semantics.verify_semantics(input, &self.samples[0]) {
                return Err(LightrayModelExecutionError::LightrayModelInputSemanticError(x));
            }
        }
        let result = self.executor.forward(input);
        match result {
            Result::Ok(x) => Ok(x),
            Result::Err(y) => Err(LightrayModelExecutionError::InternalTorchScriptError(y)),
        }
    }
}
