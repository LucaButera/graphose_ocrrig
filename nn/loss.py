from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Module
from torch.nn.functional import relu


class GANHingeLoss(Module):
    def __init__(self):
        super().__init__()
        self.loss_by_mode = {
            "D_real": lambda v: relu(1.0 - v).mean(),
            "D_fake": lambda v: relu(1.0 + v).mean(),
            "G": lambda v: -v.mean(),
        }

    def forward(self, pred: Tensor, mode: str) -> Tensor:
        try:
            return self.loss_by_mode[mode](pred)
        except KeyError:
            raise ValueError(f"Mode {mode} is not one of {self.loss_by_mode.keys()}")


loss_by_name = {
    "Hinge": GANHingeLoss,
    "BCEWL": BCEWithLogitsLoss,
}
