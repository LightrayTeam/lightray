use crate::lightray_executor::errors::{
    LightrayModelInputSemanticError, LightrayVerificationInputDoesNotEqual,
    LightrayVerificationInputSize, LightrayVerificationInputSizeDoesNotEqual,
    LightrayVerificationInputTypes,
};
use crate::lightray_torch::core::{SerializableIValue, TorchScriptInput};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum LightrayIValueSemantic {
    ExactValueMatch,
    SizeMatch,
    TypeMatch,
}
#[derive(Serialize, Deserialize)]
pub struct LightrayModelSemantics {
    pub positional_semantics: Vec<LightrayIValueSemantic>,
}

impl LightrayModelSemantics {
    pub fn verify_semantics(
        &self,
        model_input: &TorchScriptInput,
        model_baseline: &TorchScriptInput,
    ) -> Result<(), LightrayModelInputSemanticError> {
        if model_input.positional_arguments.len() != self.positional_semantics.len() {
            return Err(
                LightrayModelInputSemanticError::LightrayVerificationInputSize(
                    LightrayVerificationInputSize {
                        input_length: model_input.positional_arguments.len() as u16,
                        output_length: self.positional_semantics.len() as u16,
                    },
                ),
            );
        }
        for i in 0..model_input.positional_arguments.len() {
            if std::mem::discriminant(&model_baseline.positional_arguments[i])
                != std::mem::discriminant(&model_input.positional_arguments[i])
            {
                return Err(
                    LightrayModelInputSemanticError::LightrayVerificationInputTypes(
                        LightrayVerificationInputTypes {
                            argument_position: i as u16,
                        },
                    ),
                );
            }
            match &self.positional_semantics[i] {
                LightrayIValueSemantic::ExactValueMatch => {
                    if model_baseline.positional_arguments[i] != model_input.positional_arguments[i]
                    {
                        return Err(
                            LightrayModelInputSemanticError::LightrayVerificationInputDoesNotEqual(
                                LightrayVerificationInputDoesNotEqual {
                                    argument_position: i as u16,
                                },
                            ),
                        );
                    }
                }
                LightrayIValueSemantic::SizeMatch => {
                    match (
                        &model_baseline.positional_arguments[i],
                        &model_input.positional_arguments[i],
                    ) {
                        (SerializableIValue::Tuple(x), SerializableIValue::Tuple(y)) => {
                            if x.len() != y.len() {
                                return Err(LightrayModelInputSemanticError::LightrayVerificationInputSizeDoesNotEqual(
                                    LightrayVerificationInputSizeDoesNotEqual{ argument_position: i as u16, expected_input_size: x.len() as u16}));
                            }
                        }
                        (SerializableIValue::List(x), SerializableIValue::List(y)) => {
                            if x.len() != y.len() {
                                return Err(LightrayModelInputSemanticError::LightrayVerificationInputSizeDoesNotEqual(
                                    LightrayVerificationInputSizeDoesNotEqual{ argument_position: i as u16, expected_input_size: x.len() as u16}));
                            }
                        }
                        (_, _) => {}
                    }
                }
                LightrayIValueSemantic::TypeMatch => { //already satisfied
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_different_length_verification() {
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
        let _torchscript_semantic = LightrayModelSemantics {
            positional_semantics: vec![],
        };
        let serialized = serde_json::to_string(&torchscript_input).unwrap();
        let unserialized: TorchScriptInput = serde_json::from_str(&serialized).unwrap();
        assert_eq!(torchscript_input, unserialized)
    }
}
