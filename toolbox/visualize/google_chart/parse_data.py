import json
import numpy as np

with open('test_data.json', 'r') as f:
    data = json.load(f)

print(data)


for i, cell in enumerate(data['data']['fine_tuned']['0']['rows']):

    cell['c'].append({'v': np.random.randint(6)})

    data['data']['fine_tuned']['0']['rows'][i] = cell

with open('test_data_with_cluster.json', 'w') as f:
    json.dump(data, f)