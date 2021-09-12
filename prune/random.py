import numpy as np

import torch

from .magnitude import Pruner


class RandomPruner(Pruner):
    def prune(self):
        masks = self.get_masks()
        with torch.no_grad():
            shifts = [0]
            for _m in masks:
                shifts.append(shifts[-1] + _m.sum().item())

            w_to_prune = np.ones((shifts[-1],), dtype=bool)
            _inds = np.arange(shifts[-1])
            np.random.shuffle(_inds)
            w_to_prune[_inds[:int(self.perc_to_prune * shifts[-1] / 100.)]] = False
            for _i in range(len(shifts) - 1):
                _m = masks[_i]
                _m[_m.clone()] = torch.from_numpy(w_to_prune[shifts[_i]:shifts[_i + 1]]).to(_m.device)
