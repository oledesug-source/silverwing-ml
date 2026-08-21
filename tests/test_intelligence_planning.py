"""Tests for M15.5: task planning and decomposition."""

from __future__ import annotations

from unittest.mock import MagicMock

from intelligence.planning import Plan, Planner, PlanStatus, PlanStep


def _mock_generator(output: str = "1. Step one\n2. Step two\n3. Step three") -> MagicMock:
    gen = MagicMock()
    result = MagicMock()
    result.text = output
    gen.generate.return_value = result
    return gen


def test_decompose():
    gen = _mock_generator()
    planner = Planner(gen)
    plan = planner.decompose("Train a model")
    assert isinstance(plan, Plan)
    assert plan.goal == "Train a model"
    assert len(plan.steps) == 3
    assert gen.generate.called


def test_plan_steps_have_ids():
    gen = _mock_generator("1. First\n2. Second")
    planner = Planner(gen)
    plan = planner.decompose("test")
    assert plan.steps[0].step_id == 1
    assert plan.steps[1].step_id == 2


def test_plan_progress():
    plan = Plan(goal="test")
    plan.steps = [
        PlanStep(description="a", status=PlanStatus.DONE),
        PlanStep(description="b", status=PlanStatus.PENDING),
    ]
    assert plan.progress == 0.5
    assert not plan.all_done
    assert not plan.any_failed


def test_plan_all_done():
    plan = Plan(goal="test")
    plan.steps = [
        PlanStep(description="a", status=PlanStatus.DONE),
        PlanStep(description="b", status=PlanStatus.SKIPPED),
    ]
    assert plan.all_done
    assert plan.progress == 1.0


def test_plan_any_failed():
    plan = Plan(goal="test")
    plan.steps = [
        PlanStep(description="a", status=PlanStatus.DONE),
        PlanStep(description="b", status=PlanStatus.FAILED),
    ]
    assert plan.any_failed


def test_plan_next_step():
    plan = Plan(goal="test")
    plan.steps = [
        PlanStep(description="done", status=PlanStatus.DONE),
        PlanStep(description="next", status=PlanStatus.PENDING),
        PlanStep(description="later", status=PlanStatus.PENDING),
    ]
    nxt = plan.next_step()
    assert nxt is not None
    assert nxt.description == "next"


def test_plan_next_step_when_all_done():
    plan = Plan(goal="test")
    plan.steps = [PlanStep(description="a", status=PlanStatus.DONE)]
    assert plan.next_step() is None


def test_plan_completed_steps():
    plan = Plan(goal="test")
    plan.steps = [
        PlanStep(description="a", status=PlanStatus.DONE),
        PlanStep(description="b", status=PlanStatus.PENDING),
        PlanStep(description="c", status=PlanStatus.SKIPPED),
    ]
    completed = plan.completed_steps()
    assert len(completed) == 2


def test_plan_empty():
    plan = Plan(goal="test")
    assert plan.all_done is False
    assert plan.progress == 0.0
    assert plan.next_step() is None


def test_replan():
    gen = _mock_generator("1. New step A\n2. New step B")
    planner = Planner(gen)
    old_plan = Plan(
        goal="test",
        steps=[PlanStep(description="failed", step_id=1, status=PlanStatus.FAILED)],
    )
    new_plan = planner.replan(old_plan, old_plan.steps[0], "it broke")
    assert len(new_plan.steps) == 2
    assert gen.generate.called


def test_summarize_plan():
    gen = _mock_generator()
    planner = Planner(gen)
    plan = Plan(goal="Build it", steps=[
        PlanStep(description="design", step_id=1, status=PlanStatus.DONE),
        PlanStep(description="implement", step_id=2, status=PlanStatus.PENDING),
    ])
    summary = planner.summarize_plan(plan)
    assert "Build it" in summary
    assert "design" in summary
    assert "implement" in summary
    assert "50%" in summary


def test_step_completed_property():
    s = PlanStep(description="test", status=PlanStatus.DONE)
    assert s.completed
    s2 = PlanStep(description="test", status=PlanStatus.SKIPPED)
    assert s2.completed
    s3 = PlanStep(description="test", status=PlanStatus.PENDING)
    assert not s3.completed
