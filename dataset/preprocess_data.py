import json
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent
DATASET_NAMES = [
    'Laptops_corenlp',
    'Restaurants_corenlp',
    'Tweets_corenlp',
]
SPLITS = ['train', 'test']


def build_short_matrix(heads):
    size = len(heads)
    adjacency = [[0] * size for _ in range(size)]
    for idx, parent in enumerate(heads):
        if parent == 0:
            continue
        adjacency[idx][parent - 1] = 1
        adjacency[parent - 1][idx] = 1

    neighbors = {i: [] for i in range(size)}
    for i in range(size):
        for j in range(size):
            if adjacency[i][j] == 1:
                neighbors[i].append(j)

    short = [[5] * size for _ in range(size)]
    for i in range(size):
        visited = {i}
        short[i][i] = 0
        frontier = [i]
        for distance in range(1, 5):
            next_frontier = []
            for node in frontier:
                for neighbor in neighbors[node]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    short[i][neighbor] = distance
                    next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break
    return short


def process_file(dataset_name, split):
    input_path = DATASET_DIR / dataset_name / f'{split}_with_amr.json'
    output_path = DATASET_DIR / dataset_name / f'{split}_write.json'

    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    for sample in data:
        sample['short'] = build_short_matrix(list(sample['head']))

    with output_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def main():
    for dataset_name in DATASET_NAMES:
        for split in SPLITS:
            process_file(dataset_name, split)


if __name__ == '__main__':
    main()
