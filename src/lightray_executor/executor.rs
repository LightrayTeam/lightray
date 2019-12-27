use crate::lightray_executor::errors::{LightrayModelExecutionError, LightrayRegistrationError};
use crate::lightray_executor::model::{LightrayModel, LightrayModelId};
use crate::lightray_executor::statistics::LightrayModelExecutionStatistic;
use crate::lightray_torch::core::{SerializableIValue, TorchScriptInput};

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
pub struct LightrayExecutedExample {
    pub execution_statistic: LightrayModelExecutionStatistic,
    pub execution_result: SerializableIValue,
}

pub trait LightrayExecutor {
    fn execute(
        &self,
        model_id: LightrayModelId,
        example: TorchScriptInput,
    ) -> Result<LightrayExecutedExample, LightrayModelExecutionError>;

    fn register_model(&mut self, model: LightrayModel) -> Result<(), LightrayRegistrationError>;
    fn delete_model(&mut self, model_id: LightrayModelId) -> Result<(), LightrayRegistrationError>;
}

pub struct InMemorySimpleLightrayExecutor {
    in_memory_mapping: Rc<RefCell<HashMap<LightrayModelId, Rc<LightrayModel>>>>,
}

impl InMemorySimpleLightrayExecutor {
    pub fn new(in_memory_mapping: Rc<HashMap<LightrayModelId, Rc<LightrayModel>>>) -> Self {
        Self {
            in_memory_mapping: Rc::new(RefCell::new(HashMap::new())),
        }
    }
}
impl LightrayExecutor for InMemorySimpleLightrayExecutor {
    fn execute(
        &self,
        model_id: LightrayModelId,
        example: TorchScriptInput,
    ) -> Result<LightrayExecutedExample, LightrayModelExecutionError> {
        unimplemented!()
    }
    fn register_model(&mut self, model: LightrayModel) -> Result<(), LightrayRegistrationError> {
        if let Err(x) = model.verify() {
            return Err(LightrayRegistrationError::LightrayModelVerificationError(x));
        }
        self.in_memory_mapping
            .borrow_mut()
            .insert(model.id, Rc::new(model));
        Ok(())
    }
    fn delete_model(&mut self, model_id: LightrayModelId) -> Result<(), LightrayRegistrationError> {
        match self.in_memory_mapping.borrow_mut().remove(&model_id) {
            None => return Err(LightrayRegistrationError::MissingModel),
            _ => Ok(())
        }
    }
}
