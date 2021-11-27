import numpy as np
import torch


class StructureMapper(object):
    def __init__(self, masks):
        self.masks = self.get_values(masks)

        self.shapes = None
        self.shifts = None

        self._build()

    @staticmethod
    def get_values(structure):
        if not isinstance(structure, (tuple, list)):
            return StructureMapper.get_values([structure])[0]

        return [_t.detach().cpu().numpy().copy()
                if isinstance(_t, torch.Tensor) else _t
                for _t in structure]

    def _build(self):
        self.shapes = [_m.shape for _m in self.masks]
        self.shifts = []

        _shift = 0
        for _m in self.masks:
            _shift += np.sum(_m)
            self.shifts.append(_shift)

    def structure_to_line(self, structure):
        structure = self.get_values(structure)
        line = np.concatenate([_e[_m] for _e, _m in zip(structure, self.masks)], axis=-1)
        return line

    def line_to_structure(self, line, default_value):
        line = self.get_values(line)
        structure = []

        _begin = 0
        for _shape, _end, _m in zip(self.shapes, self.shifts, self.masks):
            _struct = np.full(_shape, default_value)
            _struct[_m] = line[_begin:_end]
            structure.append(_struct)
            _begin = _end

        return structure


class RestoreTrainQuality(object):
    """
    Change mask algorithm to restore model quality
    """

    def __init__(self, masker, N_max, N_min, T, alpha, target_quality, rf_l2=0.7, mode=None):
        self.masker = masker

        self.N_max = N_max  # max number parameters to restore
        self.N_min = N_min  # min number parameters to restore
        self.T = T  # threshold
        self.alpha = alpha  # mask importance metric parameter
        self.target_quality = target_quality  # target model quality
        self.rf_l2 = rf_l2  # reduce factor for l2 coef in masker
        self.mode = mode  # can be 'random'
        assert self.mode in [None, "random"]

        self.history = []

    def step(self, quality):
        self.history.append({
            "quality": quality,
            "nonzero_weights": self.masker.count_nonzero_weights()
        })

        print("")
        print("RestoreTrainQuality: step")

        if quality > self.target_quality:
            raise StopIteration

        # construct mapper
        with torch.no_grad():
            mapper = StructureMapper([torch.logical_not(_m) for _m in self.masker.get_masks()])

        # get last candles
        _candle_min = mapper.structure_to_line(self.masker.candles[-1][0])
        _candle_max = mapper.structure_to_line(self.masker.candles[-1][1])

        # calculate mask importance
        mi = (np.abs((_candle_min + _candle_max) * 0.5) + self.alpha * (_candle_max - _candle_min)) / (1. + self.alpha)

        # get weights to remove l2
        if self.mode is None:
            _thresh = max(np.percentile(mi, 100. * (1. - self.N_max / mi.shape[0])), self.T)
            _to_remove_mask = mi >= _thresh
        elif self.mode == "random":
            _thresh = float("-inf")
            _to_remove_mask = np.zeros_like(mi, dtype=bool)
            _to_remove_mask[np.random.randint(0, high=mi.shape[0], size=(self.N_max,))] = True
        else:
            raise ValueError("Unknown mode: %s" % str(self.mode))

        _n_to_remove_mask = np.sum(_to_remove_mask)

        print("RestoreTrainQuality: _thresh: %e, _n_to_remove: %d" % (_thresh, int(_n_to_remove_mask)))

        if _n_to_remove_mask < self.N_min:
            self.masker.set_l2_coef(self.masker.l2_coef * self.rf_l2)
            print("RestoreTrainQuality: set l2_coef: %e" % self.masker.l2_coef)
            return  # continue training

        _to_remove_mask = mapper.line_to_structure(_to_remove_mask, False)

        # remove l2 from selected weights
        with torch.no_grad():
            for _m, _remove_mask in zip(self.masker.get_masks(), _to_remove_mask):
                _n = int(np.sum(_remove_mask))
                _remove_mask = torch.from_numpy(_remove_mask).to(_m.device)
                _m[_remove_mask] = torch.full([_n], True).to(_m.device)
