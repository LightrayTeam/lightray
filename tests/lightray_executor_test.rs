use lightray::lightray_executor::executor::{
    InMemorySimpleLightrayExecutor, LightrayExecutedExample, LightrayExecutor,
};
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
fn generic_text_based_model() -> LightrayModel {
    let graph = TorchScriptGraph {
        batchable: false,
        module: CModule::load(GENERIC_TEXT_BASED_MODEL).unwrap(),
    };
    let lightray_id = LightrayModelId {
        model_id: 1234,
        model_version: 0,
    };

    LightrayModel::new(lightray_id, graph, vec![generic_text_based_model_input()])
}

#[test]
fn lightray_generic_text_based_model() {
    let lightray_model = generic_text_based_model();
    let expected_output = SerializableIValue::List(vec![
        SerializableIValue::Str("<bos>".to_string()),
        SerializableIValue::Str("call".to_string()),
        SerializableIValue::Str("mom".to_string()),
        SerializableIValue::Str("<eos>".to_string()),
    ]);
    assert!(!lightray_model.verify().is_err());
    assert!(!lightray_model.warmup_jit(100).is_err());
    assert_eq!(
        lightray_model
            .execute(&generic_text_based_model_input())
            .unwrap(),
        expected_output
    )
}
#[test]
fn simple_executor_generic_text_based_model() {
    let mut executor = InMemorySimpleLightrayExecutor::new();
    let lightray_model = generic_text_based_model();

    let register_result = executor.register_model(lightray_model);

    assert!(register_result.is_ok());
    let model_id = register_result.unwrap();

    let output_exec = executor.execute(&model_id, &generic_text_based_model_input());
    assert!(output_exec.is_ok());

    let raw_output: LightrayExecutedExample = output_exec.unwrap();

    assert!(
        raw_output
            .execution_statistic
            .elapsed_execution_time
            .subsec_millis()
            < 5
    );
    assert_eq!(
        raw_output.execution_result,
        SerializableIValue::List(vec![
            SerializableIValue::Str("<bos>".to_string()),
            SerializableIValue::Str("call".to_string()),
            SerializableIValue::Str("mom".to_string()),
            SerializableIValue::Str("<eos>".to_string()),
        ])
    );
}
