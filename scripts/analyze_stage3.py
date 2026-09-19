"""Checksum-verified descriptive paired analysis; no independence-based tests."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from stage2 import atomic_json, digest, verified_result

SEARCH = ('random_search', 'ga_standard', 'ga_mutation_only',
          'ga_tournament_only', 'ga_both')
PAIRS = [(m, 'random_search') for m in SEARCH[1:]] + [
    (m, 'ga_standard') for m in SEARCH[2:]]


def describe(values):
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'sd': statistics.stdev(values) if len(values) > 1 else None,
            'median': statistics.median(values) if values else None,
            'min': min(values) if values else None,
            'max': max(values) if values else None}


def analyze(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    config = manifest['identity']['config']
    results, missing, partitions = {}, [], {}
    for seed in config['seeds']:
        for method in config['methods']:
            unit = root / 'units' / f'{seed}-{method}'
            status_path = unit / 'status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {'state': 'pending'}
            if status['state'] != 'complete':
                missing.append({'seed': seed, 'method': method, 'state': status['state']})
                continue
            result = verified_result(unit, status)
            if (result['seed'], result['method']) != (seed, method):
                raise ValueError('Checkpoint does not match requested unit')
            split = (result['train_indices'], result['test_indices'])
            if set(split[0]) & set(split[1]):
                raise ValueError('Training and test rows overlap')
            if seed in partitions and partitions[seed] != split:
                raise ValueError('Methods use different outer splits')
            partitions[seed] = split
            if method in SEARCH:
                search = result['search']
                if (search['unique_evaluations'] != config['budget'] or
                        search['cv_model_fits'] != 3 * config['budget'] or
                        len(search['history']) != config['budget'] or
                        len({r['mask'] for r in search['history']}) != config['budget']):
                    raise ValueError('Search budget mismatch')
            results[seed, method] = result
    summaries = []
    stability = []
    for method in config['methods']:
        units = [results[s, method] for s in config['seeds'] if (s, method) in results]
        for classifier in ('Logistic Regression', 'Random Forest', 'SVM'):
            summaries.append({'method': method, 'classifier': classifier,
                              'requested_seeds': len(config['seeds']),
                              'included_seeds': [r['seed'] for r in units],
                              **{key: describe([r['classifiers'][classifier][key] for r in units])
                                 for key in ('macro_f1', 'accuracy', 'balanced_accuracy',
                                             'fit_seconds', 'predict_seconds')},
                              **{key: describe([r[key] for r in units]) for key in
                                 ('required_original_inputs', 'representation_seconds')}})
        if method in (*SEARCH, 'mi8', 'rfe8'):
            sets = [set(r['selected_original_names']) for r in units]
            overlaps = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
            frequencies = Counter(name for group in sets for name in group)
            stability.append({'method': method, 'completed_seeds': len(units),
                              'feature_counts': dict(sorted(frequencies.items())),
                              'pairwise_jaccard': describe(overlaps)})
    pairs = []
    for left, right in PAIRS:
        if left not in config['methods'] or right not in config['methods']:
            continue
        seeds = [s for s in config['seeds'] if (s, left) in results and (s, right) in results]
        for classifier in ('Logistic Regression', 'Random Forest', 'SVM'):
            differences = [results[s, left]['classifiers'][classifier]['macro_f1'] -
                           results[s, right]['classifiers'][classifier]['macro_f1'] for s in seeds]
            pairs.append({'left': left, 'right': right, 'classifier': classifier,
                          'included_seeds': seeds, 'macro_f1_difference': describe(differences),
                          'wins': sum(d > 1e-12 for d in differences),
                          'ties': sum(abs(d) <= 1e-12 for d in differences),
                          'losses': sum(d < -1e-12 for d in differences)})
    return {'scope': 'Descriptive repeated-holdout summaries; no independent-sample inference.',
            'input_manifest_sha256': digest(root / 'manifest.json'),
            'analysis_script_sha256': digest(__file__), 'config': config,
            'missing_units': missing, 'summaries': summaries,
            'paired_comparisons': pairs, 'subset_stability': stability}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.input)
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / 'analysis.json', report)
    print(f'Analysis saved; {len(report["missing_units"])} missing units.')


if __name__ == '__main__':
    main()
