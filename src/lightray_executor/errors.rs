use crate::lightray_torch::core::SerializableIValue;
use crate::lightray_torch::errors::InternalTorchError;
use serde_json;
use std::error::Error;
use std::fmt;
#[derive(Debug)]
pub struct LightrayMissingSamples {}

#[derive(Debug)]
pub struct LightrayVerificationInputSize {
    pub input_length: u16,
    pub output_length: u16,
}
#[derive(Debug)]
pub struct LightrayVerificationInputTypes {
    pub argument_position: u16,
}
#[derive(Debug)]
pub struct LightrayVerificationInputDoesNotEqual<'a> {
    pub argument_position: u16,
    pub value: &'a SerializableIValue,
}
#[derive(Debug)]
pub struct LightrayVerificationInputSizeDoesNotEqual {
    pub argument_position: u16,
    pub expected_input_size: u16,
}
impl fmt::Display for LightrayMissingSamples {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LightrayModel is missing samples to verify/warmup TorchScript models"
        )
    }
}

impl Error for LightrayMissingSamples {}

impl fmt::Display for LightrayVerificationInputSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LightrayModel expected {0} inputs but instead got {1}",
            self.input_length, self.output_length
        )
    }
}

impl Error for LightrayVerificationInputSize {}

impl fmt::Display for LightrayVerificationInputTypes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LightrayModel input does not match the correct type at position {0}",
            self.argument_position
        )
    }
}

impl Error for LightrayVerificationInputTypes {}

impl<'a> fmt::Display for LightrayVerificationInputDoesNotEqual<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LightrayModel input at position {0} does not equal it's exact value {1}",
            self.argument_position,
            // This should not panic, our tests cover all variants of SerializableIValue
            serde_json::to_string(&self.value).unwrap()
        )
    }
}

impl<'a> Error for LightrayVerificationInputDoesNotEqual<'a> {}

impl fmt::Display for LightrayVerificationInputSizeDoesNotEqual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LightrayModel input at position {0} requires exactly {1} arguments",
            self.argument_position, self.expected_input_size
        )
    }
}

impl Error for LightrayVerificationInputSizeDoesNotEqual {}
#[derive(Debug)]
pub enum LightrayModelVerificationError {
    InternalTorchError(InternalTorchError),
    LightrayMissingSamples(LightrayMissingSamples),
}

#[derive(Debug)]
pub enum LightrayModelExecutionError {
    InternalTorchScriptError(InternalTorchError),
    IncorrectTypeSignature,
    MissingModel,
}

#[derive(Debug)]
pub enum LightrayRegistrationError {
    LightrayModelVerificationError(LightrayModelVerificationError),
    MissingModel,
}

#[derive(Debug)]
pub enum LightrayModelInputSemanticError<'a> {
    LightrayVerificationInputSize(LightrayVerificationInputSize),
    LightrayVerificationInputTypes(LightrayVerificationInputTypes),
    LightrayVerificationInputDoesNotEqual(LightrayVerificationInputDoesNotEqual<'a>),
    LightrayVerificationInputSizeDoesNotEqual(LightrayVerificationInputSizeDoesNotEqual)
}
