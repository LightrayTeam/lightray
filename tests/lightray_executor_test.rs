use lightray::lightray_executor::{LightrayModel, LightrayModelId};
use lightray::lightray_torch::{SerializableIValue, TorchScriptGraph, TorchScriptInput};
use tch::CModule;

static GENERIC_TEXT_BASED_MODEL: &'static str =
    "tests/torchscript_models/generic_text_based_model.pt";

fn generic_text_based_model_input() -> TorchScriptInput {
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
#[test]
fn lightray_generic_text_based_model() {
    let graph = TorchScriptGraph {
        batchable: false,
        module: CModule::load(GENERIC_TEXT_BASED_MODEL).unwrap(),
    };
    let lightray_id = LightrayModelId {
        model_id: 1234,
        model_version: 0,
    };
    let lightray_model =
        LightrayModel::new(lightray_id, graph, vec![generic_text_based_model_input()]);

    let expected_output = SerializableIValue::List(vec![
        SerializableIValue::Str("<bos>".to_string()),
        SerializableIValue::Str("call".to_string()),
        SerializableIValue::Str("mom".to_string()),
        SerializableIValue::Str("<eos>".to_string()),
    ]);
    assert!(!lightray_model.verify().is_err());
    assert!(!lightray_model.warmup_jit(100).is_err());
}
