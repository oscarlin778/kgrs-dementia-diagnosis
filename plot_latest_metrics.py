import json
import matplotlib.pyplot as plt
import numpy as np
import os

metrics_path = '/home/wei-chi/Data/script/results/E13_Ensemble/metrics.json'
output_path = '/home/wei-chi/Data/script/results/E13_latest_performance.png'

with open(metrics_path, 'r') as f:
    metrics = json.load(f)

tasks = list(metrics.keys())
accs = [metrics[t]['acc'] for t in tasks]
aucs = [metrics[t]['auc'] for t in tasks]

x = np.arange(len(tasks))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, accs, width, label='Accuracy (ACC)', color='skyblue')
rects2 = ax.bar(x + width/2, aucs, width, label='AUC', color='salmon')

ax.set_ylabel('Score')
ax.set_title('E13 Ensemble Model Performance (Latest)')
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.legend()

# Add labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(output_path)
print(f"Chart saved to {output_path}")
