from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_existing_run_is_not_overwritten(tmp_path):
    sentinel = tmp_path / 'manifest.json'
    sentinel.write_text('preserve me')
    result = subprocess.run([sys.executable, 'src/main.py', '--output', str(tmp_path)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'FileExistsError' in result.stderr
    assert sentinel.read_text() == 'preserve me'


def test_duplicate_seeds_rejected_before_creating_run(tmp_path):
    destination = tmp_path / 'new-run'
    result = subprocess.run([sys.executable, 'src/main.py', '--output', str(destination),
                             '--seeds', '42', '42'], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Seeds must be unique' in result.stderr
    assert not destination.exists()
