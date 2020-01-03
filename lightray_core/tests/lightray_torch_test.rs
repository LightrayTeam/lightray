use lightray_core::lightray_torch::{SerializableIValue, TorchScriptGraph, TorchScriptInput};
use std::convert::TryFrom;
use std::fs::read_to_string;
use tch::{CModule, IValue};
static GENERIC_TEXT_BASED_MODEL: &'static str =
    "tests/torchscript_models/generic_text_based_model.pt";
static NPY_VECTOR_3: &'static str = "tests/torchscript_models/single_vector_numpy.npy";
static NPY_MATRIX_3X5: &'static str = "tests/torchscript_models/single_matrix_numpy.npy";
static NPY_TENSOR_3X5X7: &'static str = "tests/torchscript_models/single_tensor3_numpy.npy";
static NPY_TENSOR_3X5X7X9: &'static str = "tests/torchscript_models/single_tensor5_numpy.npy";

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

#[test]
fn tensor_serialization_vector() {
    let value: String = read_to_string(NPY_VECTOR_3).unwrap();
    let s_tensor = SerializableIValue::TensorNPYBase64(value);
    let tensor_ivalue: IValue = IValue::try_from(&s_tensor).unwrap();
    match tensor_ivalue {
        IValue::Tensor(tensor_value) => {
            assert_eq!(tensor_value.size(), &[3]);
            assert_eq!(tensor_value.kind(), tch::Kind::Float);
        }
        _ => panic!("unpacking should be to Tensor"),
    }
}
#[test]
fn tensor_serialization_matrix() {
    let value: String = read_to_string(NPY_MATRIX_3X5).unwrap();
    let s_tensor = SerializableIValue::TensorNPYBase64(value);
    let tensor_ivalue: IValue = IValue::try_from(&s_tensor).unwrap();
    match tensor_ivalue {
        IValue::Tensor(tensor_value) => {
            assert_eq!(tensor_value.size(), &[3, 5]);
            assert_eq!(tensor_value.kind(), tch::Kind::Float);
        }
        _ => panic!("unpacking should be to Tensor"),
    }
}
#[test]
fn tensor_serialization_tensor3() {
    let value: String = read_to_string(NPY_TENSOR_3X5X7).unwrap();
    let s_tensor = SerializableIValue::TensorNPYBase64(value);
    let tensor_ivalue: IValue = IValue::try_from(&s_tensor).unwrap();
    match tensor_ivalue {
        IValue::Tensor(tensor_value) => {
            assert_eq!(tensor_value.size(), &[3, 5, 7]);
            assert_eq!(tensor_value.kind(), tch::Kind::Float);
        }
        _ => panic!("unpacking should be to Tensor"),
    }
}
#[test]
fn tensor_serialization_tensor4() {
    let value: String = read_to_string(NPY_TENSOR_3X5X7X9).unwrap();
    let s_tensor = SerializableIValue::TensorNPYBase64(value);
    let tensor_ivalue: IValue = IValue::try_from(&s_tensor).unwrap();
    match tensor_ivalue {
        IValue::Tensor(tensor_value) => {
            assert_eq!(tensor_value.size(), &[3, 5, 7, 9]);
            assert_eq!(tensor_value.kind(), tch::Kind::Float);
        }
        _ => panic!("unpacking should be to Tensor"),
    }
}
