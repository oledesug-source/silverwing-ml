"""Tests for M15.1: mathematical problem solver."""

from __future__ import annotations

from unittest.mock import MagicMock

from intelligence.mathematics import MathProblem, MathResult, MathSolver


def _mock_generator(output: str = "Step 1. 2+2=4\nFinal Answer: 4") -> MagicMock:
    gen = MagicMock()
    result = MagicMock()
    result.text = output
    gen.generate.return_value = result
    return gen


def test_solve_returns_result():
    gen = _mock_generator()
    solver = MathSolver(gen)
    result = solver.solve(MathProblem(text="What is 2+2?"))
    assert isinstance(result, MathResult)
    assert gen.generate.called


def test_extract_answer_final_answer():
    gen = _mock_generator("Step 1. compute\nFinal Answer: 42")
    solver = MathSolver(gen)
    result = solver.solve(MathProblem(text="What is 6*7?"))
    assert result.answer == "42"


def test_extract_answer_answer_prefix():
    gen = _mock_generator("Reasoning...\nAnswer: 100")
    solver = MathSolver(gen)
    result = solver.solve(MathProblem(text="sqrt(10000)?"))
    assert result.answer == "100"


def test_extract_answer_fallback_number():
    gen = _mock_generator("The result is 3.14 and nothing else")
    solver = MathSolver(gen)
    result = solver.solve(MathProblem(text="pi?"))
    assert "3.14" in result.answer


def test_extract_steps():
    gen = _mock_generator("Step 1. first step\nStep 2. second step\nAnswer: done")
    solver = MathSolver(gen)
    result = solver.solve(MathProblem(text="test"))
    assert len(result.reasoning_steps) == 2
    assert result.reasoning_steps[0] == "first step"
    assert result.reasoning_steps[1] == "second step"


def test_solve_batch():
    gen = _mock_generator("Answer: 1")
    solver = MathSolver(gen)
    problems = [MathProblem(text=f"q{i}") for i in range(3)]
    results = solver.solve_batch(problems)
    assert len(results) == 3
    assert gen.generate.call_count == 3


def test_verify_calls_generator_twice():
    gen = _mock_generator("yes, the answer is correct")
    solver = MathSolver(gen, verify=True)
    result = solver.solve(MathProblem(text="2+2=4?"))
    assert gen.generate.call_count == 2
    assert result.verified is True


def test_verify_not_called_by_default():
    gen = _mock_generator()
    solver = MathSolver(gen, verify=False)
    result = solver.solve(MathProblem(text="2+2?"))
    assert gen.generate.call_count == 1
    assert result.verified is None


def test_problem_domain():
    p = MathProblem(text="x^2 + 1 = 0", domain="algebra", difficulty=2.0)
    assert p.domain == "algebra"
    assert p.difficulty == 2.0


def test_result_correct_property():
    r = MathResult(
        problem=MathProblem(text="test"),
        raw_response="",
        answer="42",
        verified=True,
    )
    assert r.correct is True

    r2 = MathResult(
        problem=MathProblem(text="test"),
        raw_response="",
        answer="42",
        verified=False,
    )
    assert r2.correct is False

    r3 = MathResult(
        problem=MathProblem(text="test"),
        raw_response="",
        answer="42",
    )
    assert r3.correct is None
