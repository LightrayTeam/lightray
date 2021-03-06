use lightray_core::lightray_executor::errors::{
    LightrayModelExecutionError, LightrayModelInputSemanticError,
};
use lightray_core::lightray_executor::executor::{
    InMemorySimpleLightrayExecutor, LightrayExecutedExample, LightrayExecutor,
};
use lightray_core::lightray_torch::{SerializableIValue, TorchScriptInput};

static GENERIC_TEXT_BASED_MODEL: &'static str =
    "tests/torchscript_models/generic_text_based_model.pt";
static GENERIC_TEXT_BASED_MODEL_INPUT: &'static str = r#"{"positional_arguments":
 [{"List":[{"Str":"<bos>"},{"Str":"call"},{"Str":"mom"},{"Str":"<eos>"}]},
 {"Int":3},
 {"Int":3}]}"#;

mod common;

#[test]
fn lightray_generic_text_based_model() {
    let lightray_model = common::generic_text_based_model();
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
            .execute(&common::generic_text_based_model_input(), false)
            .unwrap(),
        expected_output
    )
}
#[test]
fn simple_executor_generic_text_based_model() {
    let executor = InMemorySimpleLightrayExecutor::new();
    let lightray_model = common::generic_text_based_model();

    let register_result = executor.register_model(lightray_model);

    assert!(register_result.is_ok());
    let model_id = register_result.unwrap();

    let output_exec = executor.execute(&model_id, &common::generic_text_based_model_input(), false);
    assert!(output_exec.is_ok());

    let raw_output: LightrayExecutedExample = output_exec.unwrap();

    assert!(
        raw_output
            .execution_statistic
            .elapsed_execution_time
            .subsec_millis()
            < 1
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
    let executor = InMemorySimpleLightrayExecutor::new();
    let register_result = executor.register_model(common::generic_text_based_model());
    let model_input =
        serde_json::from_str::<TorchScriptInput>(&common::GENERIC_TEXT_BASED_MODEL_INPUT).unwrap();
    let raw_output: LightrayExecutedExample = executor
        .execute(&register_result.unwrap(), &model_input, false)
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

#[test]
fn test_lightray_model_semantics() {
    let model = common::generic_text_based_model();

    let wrong_first_type_input = TorchScriptInput {
        positional_arguments: vec![
            SerializableIValue::Tuple(vec![
                SerializableIValue::Str("<bos>".to_string()),
                SerializableIValue::Str("call".to_string()),
                SerializableIValue::Str("mom".to_string()),
                SerializableIValue::Str("<eos>".to_string()),
            ]),
            SerializableIValue::Int(3),
            SerializableIValue::Int(3),
        ],
    };
    let wrong_sized_input = TorchScriptInput {
        positional_arguments: vec![
            SerializableIValue::Tuple(vec![
                SerializableIValue::Str("<bos>".to_string()),
                SerializableIValue::Str("call".to_string()),
                SerializableIValue::Str("mom".to_string()),
                SerializableIValue::Str("<eos>".to_string()),
            ]),
            SerializableIValue::Int(3),
            SerializableIValue::Int(3),
            SerializableIValue::Int(3),
        ],
    };

    match model.execute(&wrong_first_type_input, true) {
        Result::Ok(_x) => assert!(false, "failed for LightrayVerificationInputTypes"),
        Result::Err(y) => match y {
            LightrayModelExecutionError::LightrayModelInputSemanticError(z) => match z {
                LightrayModelInputSemanticError::LightrayVerificationInputTypes(_) => (),
                _ => assert!(false, "failed for LightrayVerificationInputTypes"),
            },
            _ => assert!(false, "failed for LightrayVerificationInputTypes"),
        },
    }
    match model.execute(&wrong_sized_input, true) {
        Result::Ok(_x) => assert!(false, "failed for LightrayVerificationInputSize"),
        Result::Err(y) => match y {
            LightrayModelExecutionError::LightrayModelInputSemanticError(z) => match z {
                LightrayModelInputSemanticError::LightrayVerificationInputSize(_) => (),
                _ => assert!(false, "failed for LightrayVerificationInputSize"),
            },
            _ => assert!(false, "failed for LightrayVerificationInputSize"),
        },
    }
}
