use lightray::lightray_executor::executor::{
    InMemorySimpleLightrayExecutor, LightrayExecutedExample, LightrayExecutor,
};
use lightray::lightray_executor::{
    LightrayIValueSemantic, LightrayModel, LightrayModelId, LightrayModelSemantics,
};
use lightray::lightray_torch::{SerializableIValue, TorchScriptGraph, TorchScriptInput};
use tch::CModule;

static GENERIC_TEXT_BASED_MODEL: &'static str =
    "tests/torchscript_models/generic_text_based_model.pt";
static GENERIC_TEXT_BASED_MODEL_INPUT: &'static str = r#"{"positional_arguments":
 [{"List":[{"Str":"<bos>"},{"Str":"call"},{"Str":"mom"},{"Str":"<eos>"}]},
 {"Int":3},
 {"Int":3}]}"#;

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
fn generic_text_based_model_semantics() -> LightrayModelSemantics {
    LightrayModelSemantics {
        positional_semantics: vec![
            LightrayIValueSemantic::TypeMatch,
            LightrayIValueSemantic::ExactValueMatch,
            LightrayIValueSemantic::ExactValueMatch,
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

    LightrayModel::new(
        lightray_id,
        graph,
        vec![generic_text_based_model_input()],
        generic_text_based_model_semantics(),
    )
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

#[test]
fn text_simple_executor_generic_text_based_model() {
    let mut executor = InMemorySimpleLightrayExecutor::new();
    let register_result = executor.register_model(generic_text_based_model());
    let model_input =
        serde_json::from_str::<TorchScriptInput>(&GENERIC_TEXT_BASED_MODEL_INPUT).unwrap();
    let raw_output: LightrayExecutedExample = executor
        .execute(&register_result.unwrap(), &model_input)
        .unwrap();
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
