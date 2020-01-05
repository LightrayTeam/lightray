use lightray_core::lightray_executor::{
    LightrayIValueSemantic, LightrayModel, LightrayModelId, LightrayModelSemantics,
};
use lightray_core::lightray_torch::{SerializableIValue, TorchScriptGraph, TorchScriptInput};
use tch::CModule;
use uuid::Uuid;
pub static GENERIC_TEXT_BASED_MODEL: &'static str =
    "tests/torchscript_models/generic_text_based_model.pt";

pub static GENERIC_TEXT_BASED_MODEL_INPUT: &'static str = r#"{"positional_arguments":
 [{"List":[{"Str":"<bos>"},{"Str":"call"},{"Str":"mom"},{"Str":"<eos>"}]},
 {"Int":3},
 {"Int":3}]}"#;

pub fn generic_text_based_model_input() -> TorchScriptInput {
    TorchScriptInput {
        positional_arguments: vec![
            SerializableIValue::List(vec![
                SerializableIValue::Str("<bos>".to_string()),
                SerializableIValue::Str("call".to_string()),
                SerializableIValue::Str("mom".to_string()),
                SerializableIValue::Str("<eos>".to_string()),
            ]),
            SerializableIValue::Int(3),
            SerializableIValue::Int(3),
        ],
    }
}
pub fn generic_text_based_model_semantics() -> LightrayModelSemantics {
    LightrayModelSemantics {
        positional_semantics: vec![
            LightrayIValueSemantic::TypeMatch,
            LightrayIValueSemantic::ExactValueMatch,
            LightrayIValueSemantic::ExactValueMatch,
        ],
    }
}
pub fn generic_text_based_model() -> LightrayModel {
    let graph = TorchScriptGraph {
        batchable: false,
        module: CModule::load(GENERIC_TEXT_BASED_MODEL).unwrap(),
    };
    let lightray_id = LightrayModelId {
        model_id: Uuid::new_v4(),
        model_version: 0,
    };

    LightrayModel::new(
        lightray_id,
        graph,
        vec![generic_text_based_model_input()],
        generic_text_based_model_semantics(),
    )
    .unwrap()
}
