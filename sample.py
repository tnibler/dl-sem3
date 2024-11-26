import sys
from model import Model
import torch
from torch import Tensor


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} MODEL_FILE", file=sys.stderr)
        sys.exit(1)
    state = torch.load(sys.argv[1], weights_only=True)
    a = {'image_channels': 1, 'nb_channels': 8}
    model = Model(num_blocks=4, cond_channels=2, **a)
    model.load_state_dict(state)
