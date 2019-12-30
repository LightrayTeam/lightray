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
    positional_semantics: Vec<LightrayIValueSemantic>,
    sample_input: TorchScriptInput,
}

impl LightrayModelSemantics {
    pub fn verify_semantics(
        &self,
        model_input: &TorchScriptInput,
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
            if std::mem::discriminant(&self.sample_input.positional_arguments[i])
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
                    if &self.sample_input.positional_arguments[i]
                        != &model_input.positional_arguments[i]
                    {
                        return Err(
                            LightrayModelInputSemanticError::LightrayVerificationInputDoesNotEqual(
                                LightrayVerificationInputDoesNotEqual {
                                    argument_position: i as u16,
                                    value: &self.sample_input.positional_arguments[i],
                                },
                            ),
                        );
                    }
                }
                LightrayIValueSemantic::SizeMatch => {
                    match (
                        &self.sample_input.positional_arguments[i],
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
