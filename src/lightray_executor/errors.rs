use crate::lightray_torch::errors::InternalTorchError;
use std::error::Error;
use std::fmt;
#[derive(Debug)]
pub struct LightrayMissingSamples {}

impl fmt::Display for LightrayMissingSamples {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LightrayModel is missing samples to verify/warmup TorchScript models"
        )
    }
}

impl Error for LightrayMissingSamples {}

pub enum LightrayModelVerificationError {
    InternalTorchError(InternalTorchError),
    LightrayMissingSamples(LightrayMissingSamples),
}
