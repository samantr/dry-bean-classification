import numpy as np
import pandas as pd
import pytest
from statistical_analysis import holm_adjust, run_pairwise_wilcoxon, validate_results


def test_holm():
    assert holm_adjust([0.04, 0.001, 0.03]).tolist() == pytest.approx([0.06, 0.003, 0.06])
    assert holm_adjust([]).size == 0
    assert holm_adjust([1, 1]).tolist() == [1, 1]
    with pytest.raises(ValueError):
        holm_adjust([np.nan])


def test_duplicates_rejected():
    df = pd.DataFrame([dict(seed=42, method='baseline', classifier='SVM', macro_f1=0.9)] * 2)
    with pytest.raises(ValueError, match='Duplicate'):
        validate_results(df)


def test_tied_pair():
    df = pd.DataFrame([dict(seed=seed, method=method, classifier='SVM', macro_f1=0.9)
                       for seed in [1, 2, 3] for method in ['baseline', 'vanilla_ga']])
    result = run_pairwise_wilcoxon(df)
    assert result.iloc[0].holm_p_value == 1
    assert not result.iloc[0].significant_at_0_05
