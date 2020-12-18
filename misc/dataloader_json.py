import os

import json
from torch.utils.data import Dataset


class JSONShardDataset(Dataset):
    def __init__(self, shard_dir, shard_names=None, coco=None):
        super().__init__()
        self.shard_dir = shard_dir
        self.shard_names = shard_names
        with open(os.path.join(self.shard_dir, self.shard_names[0]), 'r') as f:
            data = json.load(f)
        self.proposals = data
        if coco is not None:
            cat_ids = coco.getCatIds()
            self.cat_ids_map = {}
            for i in range(len(cat_ids)):
                self.cat_ids_map[cat_ids[i]] = i + 1
        else:
            self.cat_ids_map = []


    def __len__(self):
        return len(self.proposals)

    def __getitem__(self, img_id):
        ppls = self.proposals[str(img_id)]
        for i in range(len(ppls['dets_labels'])):
            # import pdb
            # pdb.set_trace()
            if len(self.cat_ids_map) != 0:
                ppls['dets_labels'][i][4] = self.cat_ids_map[int(ppls['dets_labels'][i][4])]
                # ppls[] = self.cat_ids_map[ppls[i][4]]
        # print(ppls)
        return ppls

    def getAll(self):
        return self.proposals

class JSONSingleDataset(JSONShardDataset):
    def __init__(self, json_path, primary_key=None, stride=1, coco=None):
        super().__init__(
            os.path.dirname(json_path),
            shard_names=[os.path.basename(json_path)],
            coco=coco
        )
