use crate::lightray_executor::errors::{LightrayModelExecutionError, LightrayRegistrationError};
use crate::lightray_executor::model::{LightrayModel, LightrayModelId};
use crate::lightray_executor::statistics::LightrayModelExecutionStatistic;
use crate::lightray_torch::core::{SerializableIValue, TorchScriptInput};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::mem::drop;
use std::sync::{Arc, RwLock};
use std::time::{Instant, SystemTime};

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
    in_memory_mapping: Arc<RwLock<HashMap<LightrayModelId, Arc<LightrayModel>>>>,
}

impl InMemorySimpleLightrayExecutor {
    pub fn new() -> Self {
        Self {
            in_memory_mapping: Arc::new(RwLock::new(HashMap::new())),
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
        let read_guard = self.in_memory_mapping.read()?;
        let model = read_guard
            .get(&model_id)
            .ok_or(LightrayModelExecutionError::MissingModel)?
            .clone();
        drop(read_guard);

        let system_start_time = SystemTime::now();
        let instant_start_time = Instant::now();
        let model_output = model.execute(&example, do_semantic_verification);
        let instant_end_time = Instant::now();
        let system_end_time = SystemTime::now();

        match model_output {
            Ok(output_value) => Ok(LightrayExecutedExample {
                execution_statistic: LightrayModelExecutionStatistic {
                    elapsed_execution_time: instant_end_time - instant_start_time,
                    start_execution_time: system_start_time,
                    end_execution_time: system_end_time,
                },
                execution_result: output_value,
            }),
            Err(error) => Err(error),
        }
    }

    fn register_model(
        &self,
        model: LightrayModel,
    ) -> Result<LightrayModelId, LightrayRegistrationError> {
        let model_id_clone = model.id;
        self.in_memory_mapping
            .write()?
            .insert(model.id, Arc::new(model));
        Ok(model_id_clone)
    }

    fn delete_model(&self, model_id: LightrayModelId) -> Result<(), LightrayRegistrationError> {
        match self.in_memory_mapping.write()?.remove(&model_id) {
            None => Err(LightrayRegistrationError::MissingModel),
            _ => Ok(()),
        }
    }
}
