from pathlib import Path
import re

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

output = Path('generated')
previous_trainings = list(map(str, output.glob('cycle_*/selection.csv')))
last = max(int(re.findall("[\d]+", cycle)[0]) for cycle in previous_trainings)

oracle = pd.read_csv('negative_oracle.csv')
oracle.rename({'cnnaffinity': 'oracle'}, axis=1, inplace=True)
init = pd.read_csv('random_starter.csv')
init.rename({'cnnaffinity': 'c0'}, axis=1, inplace=True)

ranges = [len(oracle[oracle.oracle < x]) for x in range(0, -8, -1)]
print(f'Dataset: <-5: {ranges[5]: 4.2f}, <-6: {ranges[6]: 4.2f}, <-7: {ranges[7]: 4.2f}')

for_pd = [oracle.oracle, init.c0]
for i in range(1, last+1):
    selection = pd.read_csv(f"{output}/cycle_{i:04d}/selection.csv")
    size = len(selection)
    print(f'Cycle {i:4d}. Mean: {selection.cnnaffinity.mean():4.2f}, SD: {selection.cnnaffinity.std():4.2f}, Min: {selection.cnnaffinity.min():4.2f}, Max: {selection.cnnaffinity.max():4.2f}, '
          f'Below -6: {sum(selection.cnnaffinity < -6)/size:3.2f}, Below -7: {sum(selection.cnnaffinity < -7)/size:3.2f},')

    newcol = f'c{i}'
    newdf = selection.rename({'cnnaffinity': newcol}, axis=1)
    for_pd.append(newdf[newcol])

# g = sns.catplot(data=pd.concat(for_pd, axis=1), kind="violin")
g = sns.violinplot(data=pd.concat(for_pd, axis=1))
g.set(ylabel="CNNAffinity")
plt.tight_layout()

plt.savefig('violin.png')
plt.show()
# print("Linear regression: ", linregress(range(last+1), means))
