"""Tests for M15.2: logical reasoning engine."""

from __future__ import annotations

from unittest.mock import MagicMock

from intelligence.reasoning import (
    ReasoningChain,
    ReasoningEngine,
    ReasoningMode,
    ReasoningResult,
    ReasoningStep,
)


def _mock_generator(output: str = "1. Premise A is true\n2. Premise B follows\nConclusion: therefore C") -> MagicMock:
    gen = MagicMock()
    result = MagicMock()
    result.text = output
    gen.generate.return_value = result
    return gen


def test_deduction_basic():
    gen = _mock_generator()
    engine = ReasoningEngine(gen)
    result = engine.reason(
        premises=["All men are mortal", "Socrates is a man"],
        mode=ReasoningMode.DEDUCTION,
    )
    assert isinstance(result, ReasoningResult)
    assert gen.generate.called
    assert result.chain.mode == ReasoningMode.DEDUCTION


def test_conclusion_extracted():
    gen = _mock_generator("1. analysis\nConclusion: Socrates is mortal")
    engine = ReasoningEngine(gen)
    result = engine.reason(premises=["test"], mode=ReasoningMode.DEDUCTION)
    assert "Socrates is mortal" in result.conclusion


def test_steps_extracted():
    gen = _mock_generator("1. first point\n2. second point\n3. third point\nConclusion: done")
    engine = ReasoningEngine(gen)
    result = engine.reason(premises=["test"], mode=ReasoningMode.DEDUCTION)
    assert len(result.steps) == 3


def test_all_modes_produce_results():
    gen = _mock_generator("Conclusion: test conclusion")
    engine = ReasoningEngine(gen)
    for mode in ReasoningMode:
        result = engine.reason(premises=["test premise"], mode=mode)
        assert isinstance(result, ReasoningResult)
        assert gen.generate.called


def test_chain_reason():
    gen = _mock_generator("Conclusion: step 1 result")
    engine = ReasoningEngine(gen)
    results = engine.chain_reason(
        premises=["A"],
        modes=[ReasoningMode.DEDUCTION, ReasoningMode.INDUCTION],
    )
    assert len(results) == 2
    assert gen.generate.call_count == 2


def test_chain_reason_feeds_conclusions():
    call_count = [0]
    outputs = ["Conclusion: intermediate", "Conclusion: final"]

    def side_effect(*args, **kwargs):
        r = MagicMock()
        r.text = outputs[min(call_count[0], len(outputs) - 1)]
        call_count[0] += 1
        return r

    gen = MagicMock()
    gen.generate.side_effect = side_effect
    engine = ReasoningEngine(gen)
    results = engine.chain_reason(
        premises=["start"],
        modes=[ReasoningMode.DEDUCTION, ReasoningMode.CAUSAL],
    )
    assert len(results) == 2


def test_result_valid():
    r = ReasoningResult(
        chain=ReasoningChain(premises=["test"]),
        raw_response="",
        conclusion="yes",
    )
    assert r.valid is True

    r2 = ReasoningResult(
        chain=ReasoningChain(premises=["test"]),
        raw_response="",
        conclusion="",
    )
    assert r2.valid is False


def test_reasoning_step():
    s = ReasoningStep(text="step text", step_type="inference", confidence=0.9)
    assert s.text == "step text"
    assert s.confidence == 0.9
