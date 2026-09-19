"""Run the fixed stage-3 protocol through the provenance-aware stage-2 engine."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE = ['baseline', 'random_search', 'ga_standard', 'ga_mutation_only',
        'ga_tournament_only', 'ga_both', 'mi8', 'rfe8', 'pca6']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', choices=['core', 'nca'], default='core')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    command = [sys.executable, str(ROOT / 'src/stage2.py'), '--output', str(args.output),
               '--seeds', *map(str, range(1001, 1021)), '--methods',
               *(CORE if args.suite == 'core' else ['nca6']),
               '--budget', '200', '--population', '20',
               '--unit-timeout', '600', '--nca-timeout', '3600']
    if args.resume:
        command.append('--resume')
    if args.retry_failed:
        command.append('--retry-failed')
    return subprocess.call(command)


if __name__ == '__main__':
    sys.exit(main())
