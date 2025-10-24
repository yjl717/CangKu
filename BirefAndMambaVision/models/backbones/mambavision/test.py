from models.mamba_vision import mamba_vision_T
import torch
if __name__ == '__main__':
    mv = mamba_vision_T()
    mvi = torch.rand((1, 3 , 1024, 1024))
    x = mv(mvi)
    print(x.shape)
