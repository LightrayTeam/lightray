import torch
import torch.nn as nn
from typing import List

import numpy as np
import base64
import io


def write_numpy_b64(array: np.ndarray, path: str):
    f = io.BytesIO()
    np.save(f, array)
    f.seek(0)
    single_vector_numpy = base64.b64encode(f.read())
    with open(path, 'wb+') as file:
        file.write(single_vector_numpy)
    with open(path, 'rb') as file:
        assert (array == np.load(io.BytesIO(base64.b64decode(file.read())))).all()


class GenericTextBasedModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tokens: List[str], beam_size: int, top_k: int) -> List[str]:
        return input_tokens


generic_text_based_model = torch.jit.script(GenericTextBasedModel())
torch.jit.save(generic_text_based_model, "generic_text_based_model.pt")

write_numpy_b64(torch.randn((3,)).numpy(), 'single_vector_numpy.npy')
write_numpy_b64(torch.randn((3, 5)).numpy(), 'single_matrix_numpy.npy')
write_numpy_b64(torch.randn((3, 5, 7)).numpy(), 'single_tensor3_numpy.npy')
write_numpy_b64(torch.randn((3, 5, 7, 9)).numpy(), 'single_tensor5_numpy.npy')
