import json
import string as string_utils

from transformers import GPT2Tokenizer


def get_tokenizer(config):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    special_tokens = {
        "sep_token": "<LINE>",
        "pad_token": "<PAD>",
        "bos_token": "<BOS>"
    }

    if not config.data.use_bos:
        special_tokens.pop("bos_token")

    tokenizer.add_special_tokens(special_tokens)

    for key in special_tokens:
        print(key)
        print(
            f"New {key}: {getattr(tokenizer, key)} "
            f"({getattr(tokenizer, key + '_id')})")

    return tokenizer


def load_dataset(config):
    data = json.load(open(f"{config.data.data_dir}/limericks.json"))
    limericks = []

    for _, limerick in data['limericks'].items():
        lines = limerick['lines']
        flag = True

        # Remove the final punctuation of each line
        # (we'll use a special separator instead)
        for idx, line in enumerate(lines):
            if len(line) == 0:
                flag = False
                break
            if line[-1] in string_utils.punctuation:
                lines[idx] = line[:-1]
        
        if flag:
            limericks.append(lines)

    print(f"# of limericks before clean-up: {len(data['limericks'])}")
    print(f"# of limericks after clean-up: {len(limericks)}")

    return limericks