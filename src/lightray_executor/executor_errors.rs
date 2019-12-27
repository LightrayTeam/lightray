use std::error::Error;
use std::fmt;

use crate::lightray_torch::errors::InternalTorchError;

pub enum LightrayModelExecutionError {
    InternalTorchScriptError(InternalTorchError)
}