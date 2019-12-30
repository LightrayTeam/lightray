use crate::lightray_executor::errors::{LightrayModelExecutionError, LightrayRegistrationError};
use crate::lightray_executor::model::{LightrayModel, LightrayModelId};
use crate::lightray_executor::statistics::LightrayModelExecutionStatistic;
use crate::lightray_torch::core::{SerializableIValue, TorchScriptInput};

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use std::time::{Instant, SystemTime};

pub struct LightrayExecutedExample {
    pub execution_statistic: LightrayModelExecutionStatistic,
    pub execution_result: SerializableIValue,
}

pub trait LightrayExecutor {
    fn execute(
        &self,
        model_id: &LightrayModelId,
        example: &TorchScriptInput,
    ) -> Result<LightrayExecutedExample, LightrayModelExecutionError>;

    fn register_model(&mut self, model: LightrayModel) -> Result<LightrayModelId, LightrayRegistrationError>;
    fn delete_model(&mut self, model_id: LightrayModelId) -> Result<(), LightrayRegistrationError>;
}

pub struct InMemorySimpleLightrayExecutor {
    in_memory_mapping: Rc<RefCell<HashMap<LightrayModelId, Rc<LightrayModel>>>>,
}

impl InMemorySimpleLightrayExecutor {
    pub fn new() -> Self {
        Self {
            in_memory_mapping: Rc::new(RefCell::new(HashMap::new())),
        }
    }
}
impl LightrayExecutor for InMemorySimpleLightrayExecutor {
    fn execute(
        &self,
        model_id: &LightrayModelId,
        example: &TorchScriptInput,
    ) -> Result<LightrayExecutedExample, LightrayModelExecutionError> {
        if let Some(x) = self.in_memory_mapping.borrow().get(&model_id) {
            let system_start_time = SystemTime::now();
            let instant_start_time = Instant::now();
            let model_output = x.execute(&example);
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
                    return Err(LightrayModelExecutionError::InternalTorchScriptError(error))
                }
            }
        }
        Err(LightrayModelExecutionError::MissingModel)
    }
    fn register_model(&mut self, model: LightrayModel) -> Result<LightrayModelId, LightrayRegistrationError> {
        if let Err(x) = model.verify() {
            return Err(LightrayRegistrationError::LightrayModelVerificationError(x));
        }
        let model_id_clone = model.id.clone();
        self.in_memory_mapping
            .borrow_mut()
            .insert(model.id, Rc::new(model).clone());
        Ok(model_id_clone)
    }
    fn delete_model(&mut self, model_id: LightrayModelId) -> Result<(), LightrayRegistrationError> {
        match self.in_memory_mapping.borrow_mut().remove(&model_id) {
            None => return Err(LightrayRegistrationError::MissingModel),
            _ => Ok(()),
        }
    }
}
