import logging

import pytest

from math_verify import StringExtractionConfig, parse, parser
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric


def test_metric_scores_empty_prediction_list_as_zero():
    score, extracted = math_metric()(["2"], [])

    assert score == 0.0
    assert extracted is not None
    assert extracted[1] == []


@pytest.mark.parametrize("golds", [[], ["unparseable reference"]])
def test_empty_predictions_still_require_valid_gold(golds):
    with pytest.raises(ValueError, match="No gold targets found"):
        math_metric()(golds, [])


@pytest.mark.parametrize(
    ("prediction", "labels", "expected"),
    [
        ("Final answer is Apple.", ("A", "B"), []),
        ("Final answer is A1.", ("A", "B"), []),
        ("Final answer is A. I hope that helps.", ("A", "B"), ["a"]),
        ("Final answer is AB.", ("A", "AB"), ["ab"]),
        ("Final answer is A+.", ("A", "A+"), ["a+"]),
    ],
)
def test_final_answer_label_has_a_suffix_boundary(prediction, labels, expected):
    config = StringExtractionConfig(strings=labels, try_extract_without_anchor=False)
    assert parse(prediction, [config], fallback_mode="no_fallback") == expected


def test_metric_diagnostics_omit_raw_answers(caplog):
    with pytest.raises(ValueError, match="No gold targets found") as error:
        math_metric()(["private reference"], ["private prediction"])
    assert "private" not in str(error.value)

    score, _ = math_metric()(["2"], ["private prediction"])
    assert score == 0.0
    assert any(record.levelno == logging.WARNING for record in caplog.records)
    assert "private" not in caplog.text


@pytest.mark.parametrize("failure", [ValueError, RuntimeError, TimeoutException])
def test_parser_diagnostics_omit_raw_answers(monkeypatch, caplog, failure):
    def fail(*args, **kwargs):
        raise failure("private answer")

    monkeypatch.setattr(parser, "extract_target_from_pred", fail)
    with caplog.at_level(logging.DEBUG):
        assert parse("private answer") == []
    assert caplog.records
    assert "private answer" not in caplog.text


def test_raised_parser_timeout_omits_raw_answer(monkeypatch):
    def fail(*args, **kwargs):
        raise TimeoutException("private answer")

    monkeypatch.setattr(parser, "extract_target_from_pred", fail)
    with pytest.raises(TimeoutException, match="^Timeout during parsing$"):
        parse("private answer", raise_on_error=True)


def test_csv_counts_failed_rows_without_inheriting_a_verdict(monkeypatch):
    pandas = pytest.importorskip("pandas")
    import evaluate_model_outputs

    def grade(golds, predictions):
        if predictions == ["ungraded"]:
            raise ValueError("grading failed")
        if predictions == ["invalid details"]:
            return 1.0, (["2"],)
        return float(predictions == golds), (golds, predictions)

    monkeypatch.setattr(evaluate_model_outputs, "math_metric", lambda **kwargs: grade)
    rows = pandas.DataFrame(
        {
            "gold": ["2"] * 4,
            "answer": ["2", "ungraded", "invalid details", "3"],
        }
    )
    result = evaluate_model_outputs.process_answers(rows, gold_is_latex=False)

    assert result["is_correct"].tolist() == [True, False, False, False]
    assert result["error"].notna().tolist() == [False, True, True, False]
    assert result.attrs == {
        "accuracy": 0.25,
        "correct_count": 1,
        "total_count": 4,
        "failed_count": 2,
    }


def test_csv_preserves_numeric_answers_as_text(tmp_path):
    pytest.importorskip("pandas")
    import evaluate_model_outputs

    path = tmp_path / "answers.csv"
    path.write_text("answer,gold\n2,$2$\n3,$4$\n,$5$\n")

    rows = evaluate_model_outputs.load_csv_data(path)
    assert rows["answer"].tolist() == ["2", "3", ""]
    result = evaluate_model_outputs.process_answers(rows, gold_is_latex=True)
    assert result["is_correct"].tolist() == [True, False, False]
    assert result.attrs["failed_count"] == 0
