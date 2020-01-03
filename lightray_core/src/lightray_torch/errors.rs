use std::error::Error;
use std::fmt;
#[derive(Debug)]
pub struct InternalTorchError {
    pub internal_error: String,
}

impl fmt::Display for InternalTorchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Internal TorchScript Error: {0}", self.internal_error)
    }
}

impl Error for InternalTorchError {}

impl From<String> for InternalTorchError {
    fn from(internal_error: String) -> Self {
        InternalTorchError { internal_error }
    }
}
