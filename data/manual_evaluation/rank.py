import copy
import json
import os
import signal

input_file = "04_25_poems_rami.json"
output_file = "04_25_poems_rami_evaluated.json"


input_data = json.load(open(input_file))
if os.path.exists(output_file):
    output_data = json.load(open(output_file))
else:
    output_data = {}
keys = sorted(
    list(input_data.keys()),
    key=lambda x: input_data[x]['lexical_diversity'],
    reverse=True)

counter = {1: 0, 2: 0, 3: 0}
for key, value in output_data.items():
    score = value['score']
    counter[score] += 1


def handler(signum, frame):
    with open(output_file, 'w') as file:
        json.dump(output_data, file)
    exit()


signal.signal(signal.SIGINT, handler)


for key in keys:
    if key in output_data:
        continue

    while True:
        os.system("clear")
        print(f"Awesome: {counter[1]}")
        print(f"Meh: {counter[2]}")
        print(f"Trash: {counter[3]}")
        print()
        print(input_data[key]["poem"])
        print()

        x = int(input("poem score: "))
        if x in [1, 2, 3]:
            output_data[key] = copy.deepcopy(input_data[key])
            output_data[key]['score'] = x
            counter[x] += 1
            break
