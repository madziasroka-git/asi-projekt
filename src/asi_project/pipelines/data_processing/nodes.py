"""Nody przygotowania danych przed modelowaniem."""

import pandas as pd

from asi_project.config import DatasetContract
from asi_project.data import assert_valid, normalize_column_names


def validate_data(
    df: pd.DataFrame,
    contract: DatasetContract,
) -> pd.DataFrame:
    """Sprawdza kontrakt i przekazuje poprawne dane do kolejnego kroku."""
    df = normalize_column_names(df)
    assert_valid(df, contract)
    return df
