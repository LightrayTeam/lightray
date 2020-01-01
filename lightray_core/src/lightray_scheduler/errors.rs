use crate::lightray_executor::errors::LightrayModelExecutionError;
#[derive(Debug)]
pub enum LightraySchedulerError {
    LightrayModelExecutionError(LightrayModelExecutionError),
    SchedulerError,
}

impl From<LightrayModelExecutionError> for LightraySchedulerError {
    fn from(x: LightrayModelExecutionError) -> Self {
        LightraySchedulerError::LightrayModelExecutionError(x)
    }
}
