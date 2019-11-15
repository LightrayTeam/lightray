use serde::{Deserialize, Serialize};
use std::collections::HashMap;
#[derive(Serialize, Deserialize, Debug)]
struct Tensor {
    //TODO: Figure out tensor backend
}

#[derive(Serialize, Deserialize, Debug)]
enum IValue {
    Bool(bool),
    Int(i64),
    Double(f64),
    Str(String),
    Tensor(Tensor),
    Tuple(Vec<IValue>),
    List(Vec<IValue>),
}

#[derive(Serialize, Deserialize, Debug)]
struct TorchScriptInput {
    named_arguments: HashMap<String, IValue>,
}

#[derive(Serialize, Deserialize, Debug)]
struct TorchScriptOutput {
    named_arguments: HashMap<String, IValue>,
    positional_arguments: Vec<IValue>,
}

struct TorchScriptGraph {}

impl TorchScriptGraph {
    fn forward(&self, inputs: &TorchScriptInput) -> Option<TorchScriptOutput> {
        None
    }
}
