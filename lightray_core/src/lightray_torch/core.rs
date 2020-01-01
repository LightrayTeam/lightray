use crate::lightray_torch::errors::InternalTorchError;
use serde::{Deserialize, Serialize};
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
}

impl From<&IValue> for SerializableIValue {
    fn from(value: &IValue) -> Self {
        match value {
            IValue::None => SerializableIValue::None,
            IValue::Bool(bool_value) => SerializableIValue::Bool(*bool_value),
            IValue::Int(int_value) => SerializableIValue::Int(*int_value),
            IValue::Double(double_value) => SerializableIValue::Double(*double_value),
            IValue::String(string_value) => SerializableIValue::Str(string_value.clone()),
            IValue::Tuple(tuple_value) => SerializableIValue::Tuple(
                tuple_value.iter().map(SerializableIValue::from).collect(),
            ),
            IValue::GenericList(tuple_value) => {
                SerializableIValue::List(tuple_value.iter().map(SerializableIValue::from).collect())
            }
            IValue::DoubleList(doubles_value) => SerializableIValue::List(
                doubles_value
                    .iter()
                    .map(|x| SerializableIValue::Double(*x))
                    .collect(),
            ),
            IValue::IntList(ints_value) => SerializableIValue::List(
                ints_value
                    .iter()
                    .map(|x| SerializableIValue::Int(*x))
                    .collect(),
            ),
            IValue::BoolList(ints_value) => SerializableIValue::List(
                ints_value
                    .iter()
                    .map(|x| SerializableIValue::Bool(*x))
                    .collect(),
            ),
            _ => unimplemented!(),
        }
    }
}
impl From<&SerializableIValue> for IValue {
    fn from(value: &SerializableIValue) -> Self {
        match value {
            SerializableIValue::None => IValue::None,
            SerializableIValue::Bool(bool_value) => IValue::Bool(*bool_value),
            SerializableIValue::Int(int_value) => IValue::Int(*int_value),
            SerializableIValue::Double(double_value) => IValue::Double(*double_value),
            SerializableIValue::Str(string_value) => IValue::String(string_value.clone()),
            SerializableIValue::Tuple(tuple_value) => {
                IValue::Tuple(tuple_value.iter().map(IValue::from).collect())
            }
            SerializableIValue::List(list_value) => {
                IValue::GenericList(list_value.iter().map(IValue::from).collect())
            }
            SerializableIValue::Optional(optional) => match &optional {
                Option::None => IValue::None,
                Option::Some(x) => IValue::from(&**x),
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
            .map(IValue::from)
            .collect();

        let model_output = self.module.forward_is(&model_inputs);
        match model_output {
            Result::Ok(true_model_output) => Ok(SerializableIValue::from(&true_model_output)),
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
