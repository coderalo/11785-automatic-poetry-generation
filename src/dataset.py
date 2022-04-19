import numpy as np
import torch

from torch.utils.data import Dataset


def merge_lines(lines, use_bos, order=None):
    if order is not None:
        try:
            order = list(order)
        except Exception:
            return
        assert isinstance(order, list)
        assert sorted(order) == [0, 1, 2, 3, 4]

        lines = [lines[o] for o in order]

    words = ' <LINE> '.join(lines) + ' <LINE>'
    if use_bos:
        words = '<BOS> ' + words

    words = ' '.join(words.split())

    return words


def reorder(lines, order=None):
    if order is None:
        return lines 
    else:
        new = [(o, i) for i, o in enumerate(order)]
        new = sorted(new)
        new = [o[1] for o in new]

        lines = [lines[o] for o in new]

    return lines


def reverse_line(
        input_ids,
        use_bos,
        tokenizer,
        reverse_last_line=False
):
    pad_token_id = tokenizer.eos_token_id
    start = 0
    for i, id_ in enumerate(input_ids):
        if id_ != pad_token_id:
            init, start = i, i
            break

    tmp_input_ids = input_ids[start:]
    new_input_ids = np.zeros_like(tmp_input_ids)
    
    if use_bos:
        new_input_ids[0] = tmp_input_ids[0]
        start = 1
    else:
        start = 0

    for end in range(1, len(tmp_input_ids)):
        if tmp_input_ids[end] == tokenizer.sep_token_id:
            new_input_ids[start: end] = tmp_input_ids[start: end][::-1]
            new_input_ids[end] = tokenizer.sep_token_id
            start = end + 1
    if reverse_last_line:
        new_input_ids[start:] = tmp_input_ids[start:][::-1]
    else:
        new_input_ids[start:] = tmp_input_ids[start:]

    new_input_ids = np.concatenate([input_ids[:init], new_input_ids], axis=0)
    return new_input_ids


class LimerickDataset(Dataset):
    def __init__(self, data, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.use_bos = config.data.use_bos
        self.order = config.data.order
        self.reverse = config.data.reverse

        self.data = [
            merge_lines(limerick, self.use_bos, self.order)
            for limerick in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def gen_collate_fn(self):
        def collate_fn(batch):
            if not self.reverse:
                batch = self.tokenizer(
                    batch, padding="longest", return_tensors="pt")
            else:
                batch = self.tokenizer(
                    batch, padding="longest", return_tensors="np")
                for i, input_ids in enumerate(batch['input_ids']):
                    batch['input_ids'][i] = reverse_line(
                        batch['input_ids'][i],
                        use_bos=self.use_bos,
                        tokenizer=self.tokenizer)
                batch['input_ids'] = torch.tensor(batch['input_ids'])
                batch['attention_mask'] = torch.tensor(batch['attention_mask'])
            batch['labels'] = torch.clone(batch['input_ids']).detach()

            for key, value in batch.items():
                batch[key] = value.cuda()
            return batch

        return collate_fn