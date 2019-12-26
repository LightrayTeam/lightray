mod lightray_torch;

use lightray_torch::TorchScriptGraph;

struct LightrayModelVersion {
    model_id: u64,
    model_version: u16,
}
struct LightrayModel {
    version: LightrayModelVersion,
    executor: TorchScriptGraph,
}
