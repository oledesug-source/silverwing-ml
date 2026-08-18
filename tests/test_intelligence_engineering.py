"""Tests for M15.3: code engineering engine."""

from __future__ import annotations

from unittest.mock import MagicMock

from intelligence.engineering import CodeEngineer, CodeResult, CodeTask
from intelligence.engineering.engineer import CodeTaskType


def _mock_generator(output: str = '```python\ndef hello():\n    pass\n```') -> MagicMock:
    gen = MagicMock()
    result = MagicMock()
    result.text = output
    gen.generate.return_value = result
    return gen


def test_generate_code():
    gen = _mock_generator()
    engineer = CodeEngineer(gen)
    result = engineer.generate("hello world function")
    assert isinstance(result, CodeResult)
    assert gen.generate.called
    assert result.task.task_type == CodeTaskType.GENERATE


def test_extract_code_block():
    gen = _mock_generator('some text\n```python\nprint("hello")\n```\nmore text')
    engineer = CodeEngineer(gen)
    result = engineer.generate("test")
    assert result.extracted_code == 'print("hello")'


def test_explain_code():
    gen = _mock_generator("This function prints hello to the console.")
    engineer = CodeEngineer(gen)
    result = engineer.explain('def hello(): print("hello")')
    assert result.task.task_type == CodeTaskType.EXPLAIN
    assert result.explanation


def test_review_code():
    gen = _mock_generator("- Unused import\n- Missing type hints\n- No bugs found")
    engineer = CodeEngineer(gen)
    result = engineer.review("import os\ndef foo(): pass")
    assert result.task.task_type == CodeTaskType.REVIEW
    assert len(result.issues) == 3
    assert "Unused import" in result.issues[0]


def test_review_no_issues_for_clean_code():
    gen = _mock_generator("Code looks good, no issues found.")
    engineer = CodeEngineer(gen)
    result = engineer.review("x = 1")
    assert len(result.issues) == 0


def test_code_task_types():
    for task_type in CodeTaskType:
        task = CodeTask(task_type=task_type, code="test", description="test")
        assert task.task_type == task_type


def test_code_result_success():
    r = CodeResult(
        task=CodeTask(task_type=CodeTaskType.GENERATE),
        raw_response="",
        extracted_code="print('hi')",
    )
    assert r.success is True

    r2 = CodeResult(
        task=CodeTask(task_type=CodeTaskType.GENERATE),
        raw_response="",
        extracted_code="",
    )
    assert r2.success is False


def test_code_block_no_fallback():
    gen = _mock_generator("just plain text without any code blocks")
    engineer = CodeEngineer(gen)
    result = engineer.generate("test")
    assert result.extracted_code == "just plain text without any code blocks"
