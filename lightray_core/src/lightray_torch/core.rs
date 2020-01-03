use crate::lightray_torch::errors::InternalTorchError;
use crate::lightray_torch::tensor::read_npy;
use base64;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use tch::IValue;
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub enum SerializableIValue {
    None,
    Bool(bool),
    Int(i64),
    Double(f64),
    Str(String),
    Tuple(Vec<SerializableIValue>),
    List(Vec<SerializableIValue>),
    Optional(Option<Box<SerializableIValue>>),
    TensorNPYBase64(String),
}

impl TryFrom<&IValue> for SerializableIValue {
    type Error = String;
    fn try_from(value: &IValue) -> Result<Self, Self::Error> {
        match value {
            IValue::None => Ok(SerializableIValue::None),
            IValue::Bool(bool_value) => Ok(SerializableIValue::Bool(*bool_value)),
            IValue::Int(int_value) => Ok(SerializableIValue::Int(*int_value)),
            IValue::Double(double_value) => Ok(SerializableIValue::Double(*double_value)),
            IValue::String(string_value) => Ok(SerializableIValue::Str(string_value.clone())),
            IValue::Tuple(tuple_value) => Ok(SerializableIValue::Tuple(
                tuple_value
                    .iter()
                    .map(SerializableIValue::try_from)
                    .collect::<Result<Vec<SerializableIValue>, String>>()?,
            )),
            IValue::GenericList(tuple_value) => Ok(SerializableIValue::List(
                tuple_value
                    .iter()
                    .map(SerializableIValue::try_from)
                    .collect::<Result<Vec<SerializableIValue>, String>>()?,
            )),
            IValue::DoubleList(doubles_value) => Ok(SerializableIValue::List(
                doubles_value
                    .iter()
                    .map(|x| SerializableIValue::Double(*x))
                    .collect(),
            )),
            IValue::IntList(ints_value) => Ok(SerializableIValue::List(
                ints_value
                    .iter()
                    .map(|x| SerializableIValue::Int(*x))
                    .collect(),
            )),
            IValue::BoolList(ints_value) => Ok(SerializableIValue::List(
                ints_value
                    .iter()
                    .map(|x| SerializableIValue::Bool(*x))
                    .collect(),
            )),
            _ => unimplemented!(),
        }
    }
}
impl TryFrom<&SerializableIValue> for IValue {
    type Error = String;
    fn try_from(value: &SerializableIValue) -> Result<Self, Self::Error> {
        match value {
            SerializableIValue::None => Ok(IValue::None),
            SerializableIValue::Bool(bool_value) => Ok(IValue::Bool(*bool_value)),
            SerializableIValue::Int(int_value) => Ok(IValue::Int(*int_value)),
            SerializableIValue::Double(double_value) => Ok(IValue::Double(*double_value)),
            SerializableIValue::Str(string_value) => Ok(IValue::String(string_value.clone())),
            SerializableIValue::Tuple(tuple_value) => Ok(IValue::Tuple(
                tuple_value
                    .iter()
                    .map(IValue::try_from)
                    .collect::<Result<Vec<IValue>, String>>()?,
            )),
            SerializableIValue::List(list_value) => Ok(IValue::GenericList(
                list_value
                    .iter()
                    .map(IValue::try_from)
                    .collect::<Result<Vec<IValue>, String>>()?,
            )),
            SerializableIValue::Optional(optional) => match &optional {
                Option::None => Ok(IValue::None),
                Option::Some(x) => Ok(IValue::try_from(&**x)?),
            },
            SerializableIValue::TensorNPYBase64(x) => match &base64::decode(&x) {
                Result::Ok(byte_array) => Ok(IValue::Tensor(read_npy(byte_array)?)),
                Result::Err(y) => Err(y.to_string()),
            },
        }
    }
}
#[derive(Serialize, Deserialize, Debug)]
pub struct TorchScriptInput {
    pub positional_arguments: Vec<SerializableIValue>,
}

impl PartialEq for TorchScriptInput {
    fn eq(&self, other: &TorchScriptInput) -> bool {
        if self.positional_arguments.len() != other.positional_arguments.len() {
            return false;
        }
        for i in 0..self.positional_arguments.len() {
            if self.positional_arguments[i] != other.positional_arguments[i] {
                return false;
            }
        }
        true
    }
}
pub struct TorchScriptGraph {
    pub batchable: bool,
    pub module: tch::CModule,
}

impl TorchScriptGraph {
    pub fn forward(
        &self,
        inputs: &TorchScriptInput,
    ) -> Result<SerializableIValue, InternalTorchError> {
        let model_inputs: Vec<IValue> = inputs
            .positional_arguments
            .iter()
            .map(IValue::try_from)
            .collect::<Result<Vec<IValue>, String>>()?;

        let model_output = self.module.forward_is(&model_inputs);
        match model_output {
            Result::Ok(true_model_output) => Ok(SerializableIValue::try_from(&true_model_output)?),
            Result::Err(error) => Err(InternalTorchError {
                internal_error: error.to_string(),
            }),
        }
    }
    pub fn forward_batched(&self) {
        assert!(
            self.batchable,
            r#"forward_batched can only be called on batchable TorchScriptGraph's"#
        );
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization() {
        let torchscript_input = TorchScriptInput {
            positional_arguments: vec![
                SerializableIValue::List(vec![
                    SerializableIValue::Str("<bos>".to_string()),
                    SerializableIValue::Str("call".to_string()),
                    SerializableIValue::Str("mom".to_string()),
                    SerializableIValue::Str("<eos>".to_string()),
                ]),
                SerializableIValue::Bool(true),
                SerializableIValue::Int(3),
                SerializableIValue::Int(3),
            ],
        };
        let serialized = serde_json::to_string(&torchscript_input).unwrap();
        let unserialized: TorchScriptInput = serde_json::from_str(&serialized).unwrap();
        assert_eq!(torchscript_input, unserialized)
    }
}
