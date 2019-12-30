import torch
import torch.nn as nn
from typing import List


class GenericTextBasedModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tokens: List[str], beam_size: int, top_k: int) -> List[str]:
        return input_tokens


generic_text_based_model = torch.jit.script(GenericTextBasedModel())
torch.jit.save(generic_text_based_model, "generic_text_based_model.pt")
