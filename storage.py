import os
import json
import functools
import multiprocessing
from collections import OrderedDict
import shutil

import numpy as np
import pandas as pd
import filelock

import torch


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return super(CustomJsonEncoder, self).default(obj)


class Storage(object):
    def __init__(self, path):
        self.path = path
        self.json_dump = functools.partial(json.dump, cls=CustomJsonEncoder)

    def _storage_lock(self):
        return filelock.SoftFileLock(os.path.join(self.path, "storage.lock"))

    def _block_lock(self, blk_id):
        return filelock.SoftFileLock(os.path.join(self.path, "%d.lock" % blk_id))

    def _read_storage_nl(self):
        with open(os.path.join(self.path, "storage.json"), "r") as f:
            storage = json.load(f)
        return storage

    def _save_storage_nl(self, storage_json):
        with open(os.path.join(self.path, "storage.json"), "w") as f:
            json.dump(storage_json, f)

    def create_storage(self):
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

            with self._storage_lock():
                self._save_storage_nl({"ids": [], "cur_id": 1})

        return self

    def delete_ids(self, id_list):
        id_list = [id_list] if not isinstance(id_list, (tuple, list)) else id_list
        with self._storage_lock():
            storage = self._read_storage_nl()

            for _id in id_list:
                if _id in storage["ids"]:
                    storage["ids"].remove(_id)
                    shutil.rmtree(os.path.join(self.path, "%d" % _id))

            self._save_storage_nl(storage)

    def clear_storage(self):
        with self._storage_lock():
            storage = self._read_storage_nl()

            ids = storage["ids"]
            storage["ids"] = []
            storage["cur_id"] = 1
            for _id in ids:
                shutil.rmtree(os.path.join(self.path, "%d" % _id))

            self._save_storage_nl(storage)

    def _get_id_nl(self):
        storage = self._read_storage_nl()
        cur_id = storage["cur_id"]
        storage["cur_id"] = cur_id + 1
        storage["ids"].append(cur_id)
        self._save_storage_nl(storage)
        return cur_id

    def _create_block_nl(self, blk_id):
        os.makedirs(os.path.join(self.path, "%d" % blk_id), exist_ok=True)

    def save(self, model, update=True):
        if not hasattr(model, "storage_id_history"):
            model.storage_id_history = []

        if not update or model.storage_id is None:
            if model.storage_id is not None:
                model.storage_id_history.append(model.storage_id)
            with self._storage_lock():
                _id = self._get_id_nl()
        else:
            _id = model.storage_id

        model.storage_id = _id

        with self._block_lock(_id):
            self._create_block_nl(_id)
            torch.save(model, os.path.join(self.path, "%d" % _id, "model.torch"))
            with open(os.path.join(self.path, "%d" % _id, "block.json"), "w") as f:
                self.json_dump(model.info, f)

    def load(self, blk_id):
        with self._block_lock(blk_id):
            map_location = None if torch.cuda.is_available() else torch.device("cpu")
            model = torch.load(os.path.join(self.path, "%d" % blk_id, "model.torch"),
                               map_location=map_location)
        return model

    @staticmethod
    def show_models_extract(storage, blk_id, keys):
        with open(os.path.join(storage.path, "%d" % blk_id, "block.json"), "r") as f:
            info = json.load(f)

        keys = set(keys)
        keys.add("id")

        res = dict(
            id=blk_id,
            data=info.get("data", "<unknown>"),
            model=info.get("model", "<unknown>"),
            comment=info.get("comment", ""),
            quality=info.get("quality", np.nan),
            nonzero_weights=info.get("nonzero_weights", np.nan),
            total_weights=info.get("total_weights", np.nan),
            status=info.get("status", "<unknown>")
        )
        info.update(res)

        return {_k: info[_k] for _k in keys}

    def _get_table(self, keys, add_keys):
        default_keys = ["id", "data", "model", "comment", "quality", "nonzero_weights", "total_weights",
                        "status"]
        keys = default_keys if keys is None else keys
        add_keys = [] if add_keys is None else add_keys
        keys = keys + add_keys
        return OrderedDict([(_k, []) for _k in keys])

    def show_models(self, limit=5, id_range=None, keys=None, add_keys=None, n_processes=None):
        if id_range is None:
            limit = 0 if limit is None else limit
            id_range = slice(-limit, None, None)

        with self._storage_lock():
            storage_json = self._read_storage_nl()

        table = self._get_table(keys, add_keys)
        keys = list(table.keys())

        with multiprocessing.Pool(processes=n_processes) as p:
            models_info = p.starmap(self.show_models_extract,
                                    [(self, _id, keys) for _id in reversed(storage_json["ids"][id_range])])

        for _i in models_info:
            for _k in keys:
                table[_k].append(_i[_k])

        table = pd.DataFrame(table)
        return table
