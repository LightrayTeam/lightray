use crate::lightray_executor::errors::{LightrayModelExecutionError, LightrayRegistrationError};
use crate::lightray_executor::model::{LightrayModel, LightrayModelId};
use crate::lightray_executor::statistics::LightrayModelExecutionStatistic;
use crate::lightray_torch::core::{SerializableIValue, TorchScriptInput};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime};

use serde::{Deserialize, Serialize};

pub type LightrayExecutorResult = Result<LightrayExecutedExample, LightrayModelExecutionError>;

#[derive(Serialize, Deserialize, Debug)]
pub struct LightrayExecutedExample {
    pub execution_statistic: LightrayModelExecutionStatistic,
    pub execution_result: SerializableIValue,
}

pub trait LightrayExecutor {
    fn execute(
        &self,
        model_id: &LightrayModelId,
        example: &TorchScriptInput,
        do_verification: bool,
    ) -> LightrayExecutorResult;

    fn register_model(
        &self,
        model: LightrayModel,
    ) -> Result<LightrayModelId, LightrayRegistrationError>;

    fn delete_model(&self, model_id: LightrayModelId) -> Result<(), LightrayRegistrationError>;
}

#[derive(Default)]
pub struct InMemorySimpleLightrayExecutor {
    in_memory_mapping: Arc<Mutex<HashMap<LightrayModelId, Arc<LightrayModel>>>>,
}

impl InMemorySimpleLightrayExecutor {
    pub fn new() -> Self {
        Self {
            in_memory_mapping: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl LightrayExecutor for InMemorySimpleLightrayExecutor {
    fn execute(
        &self,
        model_id: &LightrayModelId,
        example: &TorchScriptInput,
        do_semantic_verification: bool,
    ) -> LightrayExecutorResult {
        if let Some(x) = self.in_memory_mapping.lock().unwrap().get(&model_id) {
            let system_start_time = SystemTime::now();
            let instant_start_time = Instant::now();
            let model_output = x.execute(&example, do_semantic_verification);
            let instant_end_time = Instant::now();
            let system_end_time = SystemTime::now();

            match model_output {
                Ok(output_value) => {
                    return Ok(LightrayExecutedExample {
                        execution_statistic: LightrayModelExecutionStatistic {
                            elapsed_execution_time: instant_end_time - instant_start_time,
                            start_execution_time: system_start_time,
                            end_execution_time: system_end_time,
                        },
                        execution_result: output_value,
                    })
                }
                Err(error) => {
                    return Err(error);
                }
            }
        }
        Err(LightrayModelExecutionError::MissingModel)
    }

    fn register_model(
        &self,
        model: LightrayModel,
    ) -> Result<LightrayModelId, LightrayRegistrationError> {
        if let Err(x) = model.verify() {
            return Err(LightrayRegistrationError::LightrayModelVerificationError(x));
        }
        let model_id_clone = model.id;
        self.in_memory_mapping
            .lock()?
            .insert(model.id, Arc::new(model));
        Ok(model_id_clone)
    }

    fn delete_model(&self, model_id: LightrayModelId) -> Result<(), LightrayRegistrationError> {
        match self.in_memory_mapping.lock()?.remove(&model_id) {
            None => Err(LightrayRegistrationError::MissingModel),
            _ => Ok(()),
        }
    }
}
