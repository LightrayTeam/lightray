use lightray::lightray_torch::{SerializableIValue, TorchScriptGraph, TorchScriptInput};
use tch::CModule;

static GENERIC_TEXT_BASED_MODEL: &'static str =
    "tests/torchscript_models/generic_text_based_model.pt";

#[test]
fn torchscript_generic_text_based_model() {
    let graph = TorchScriptGraph {
        batchable: false,
        module: CModule::load(GENERIC_TEXT_BASED_MODEL).unwrap(),
    };

    let valid_input = TorchScriptInput {
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
    };
    let expected_output = SerializableIValue::List(vec![
        SerializableIValue::Str("<bos>".to_string()),
        SerializableIValue::Str("call".to_string()),
        SerializableIValue::Str("mom".to_string()),
        SerializableIValue::Str("<eos>".to_string()),
    ]);
    let model_output = graph.forward(&valid_input).unwrap();
    assert_eq!(model_output, expected_output);
}
