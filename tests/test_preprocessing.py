import numpy as np
import pandas as pd
import pytest

from spendsense.preprocessing import (bin_amounts, evaluate_imputation, impute_amounts, infer_payment_mode,
                                      load_statement, normalise_merchant, remove_duplicates)


def test_merchant_normalisation_handles_bank_noise():
    assert normalise_merchant("UPI/SWIGGY/483920193/dinner") == "Swiggy"
    assert normalise_merchant("pos 4274xx zomato bangalore biryani") == "Zomato"
    assert normalise_merchant("POS 1234XX DOMINOS BANGALORE") == "Domino's"
    assert normalise_merchant("ATM WDL 4421 cash") == "ATM Cash Withdrawal"
    assert normalise_merchant("NEFT-99887766-AMAZON-clothes") == "Amazon"


def test_payment_mode_inference():
    assert infer_payment_mode("UPI/SWIGGY/1/") == "UPI"
    assert infer_payment_mode("POS 1111XX SHELL") == "Card"
    assert infer_payment_mode("NEFT-1-HOUSE RENT") == "NetBanking"


def test_duplicate_removal_counts():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    out, removed = remove_duplicates(df)
    assert removed == 1 and len(out) == 2


def test_load_statement_accepts_debit_credit_columns():
    raw = pd.DataFrame({"Txn Date": ["01-04-2026", "02-04-2026"], "Narration": ["a", "b"],
                        "Withdrawal Amt.": ["1,200.50", ""], "Deposit Amt.": ["", "5,000"]})
    df, notes = load_statement(raw)
    assert list(df["type"]) == ["debit", "credit"]
    assert df["amount"].tolist() == [1200.5, 5000.0]
    assert notes


def test_load_statement_rejects_missing_columns():
    with pytest.raises(ValueError):
        load_statement(pd.DataFrame({"foo": [1]}))


def test_binning_strategies():
    amounts = pd.Series([50, 300, 1500, 8000, 50000])
    assert list(bin_amounts(amounts, "fixed").astype(str)) == ["Micro", "Small", "Medium", "Large", "Major"]
    assert bin_amounts(amounts, "equal_width", 3).nunique() <= 3


def test_mle_imputation_recovers_fixed_amounts():
    df = pd.DataFrame({"merchant": ["Rent"] * 6, "category": ["Rent"] * 6, "type": ["debit"] * 6,
                       "amount": [18000.0] * 5 + [np.nan]})
    out = impute_amounts(df, "mle")
    assert out["amount"].iloc[-1] == pytest.approx(18000.0)
    assert out["amount_imputed"].sum() == 1


def test_imputation_evaluation_ranks_mle_first(prepared):
    table = evaluate_imputation(prepared.clean, repeats=3)
    assert table.iloc[0]["method"].startswith("mle")
