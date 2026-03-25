import csv
import io
import json
import os
import tempfile
from datetime import datetime, timezone, timedelta

import pytest
from agent_cost_monitor import CostTracker, BudgetExceededError, Session, track_usage
from agent_cost_monitor.tracker import Usage


def test_record_and_total_cost():
    tracker = CostTracker()
    tracker.record("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)
    expected = 1000 * (3.0 / 1_000_000) + 500 * (15.0 / 1_000_000)
    assert abs(tracker.total_cost - expected) < 1e-9


def test_multiple_records():
    tracker = CostTracker()
    tracker.record("claude-sonnet-4-6", 1000, 500)
    tracker.record("claude-haiku-4-5", 2000, 1000)
    assert tracker.total_input_tokens == 3000
    assert tracker.total_output_tokens == 1500
    assert tracker.summary()["num_requests"] == 2


def test_budget_not_exceeded():
    tracker = CostTracker(budget=1.0)
    tracker.record("claude-sonnet-4-6", 100, 50)
    assert not tracker.is_over_budget


def test_budget_exceeded():
    tracker = CostTracker(budget=0.0)
    tracker.record("claude-sonnet-4-6", 1000, 500)
    assert tracker.is_over_budget


def test_no_budget():
    tracker = CostTracker()
    tracker.record("claude-sonnet-4-6", 100000, 50000)
    assert not tracker.is_over_budget


def test_cost_by_model():
    tracker = CostTracker()
    tracker.record("claude-sonnet-4-6", 1000, 500)
    tracker.record("claude-haiku-4-5", 2000, 1000)
    by_model = tracker.cost_by_model()
    assert "claude-sonnet-4-6" in by_model
    assert "claude-haiku-4-5" in by_model


def test_reset():
    tracker = CostTracker()
    tracker.record("claude-sonnet-4-6", 1000, 500)
    tracker.reset()
    assert tracker.total_cost == 0.0
    assert tracker.summary()["num_requests"] == 0


def test_default_pricing():
    tracker = CostTracker()
    tracker.record("unknown-model", 1000, 500)
    expected = 1000 * (3.0 / 1_000_000) + 500 * (15.0 / 1_000_000)
    assert abs(tracker.total_cost - expected) < 1e-9


def test_summary_fields():
    tracker = CostTracker(budget=5.0)
    tracker.record("claude-opus-4-6", 500, 200)
    s = tracker.summary()
    assert "total_cost" in s
    assert "total_input_tokens" in s
    assert "total_output_tokens" in s
    assert "num_requests" in s
    assert "budget" in s
    assert "over_budget" in s
    assert s["budget"] == 5.0
    assert not s["over_budget"]


def test_openai_gpt4o_pricing():
    tracker = CostTracker()
    tracker.record("gpt-4o", input_tokens=1000, output_tokens=500)
    expected = 1000 * (2.50 / 1_000_000) + 500 * (10.0 / 1_000_000)
    assert abs(tracker.total_cost - expected) < 1e-9


def test_openai_gpt4o_mini_pricing():
    tracker = CostTracker()
    tracker.record("gpt-4o-mini", input_tokens=2000, output_tokens=1000)
    expected = 2000 * (0.15 / 1_000_000) + 1000 * (0.60 / 1_000_000)
    assert abs(tracker.total_cost - expected) < 1e-9


def test_openai_gpt41_pricing():
    tracker = CostTracker()
    tracker.record("gpt-4.1", input_tokens=1000, output_tokens=500)
    expected = 1000 * (2.0 / 1_000_000) + 500 * (8.0 / 1_000_000)
    assert abs(tracker.total_cost - expected) < 1e-9


def test_openai_gpt41_mini_pricing():
    tracker = CostTracker()
    tracker.record("gpt-4.1-mini", input_tokens=1000, output_tokens=500)
    expected = 1000 * (0.40 / 1_000_000) + 500 * (1.60 / 1_000_000)
    assert abs(tracker.total_cost - expected) < 1e-9


def test_gemini_25_pro_pricing():
    tracker = CostTracker()
    tracker.record("gemini-2.5-pro", input_tokens=1000, output_tokens=500)
    expected = 1000 * (1.25 / 1_000_000) + 500 * (10.0 / 1_000_000)
    assert abs(tracker.total_cost - expected) < 1e-9


def test_gemini_25_flash_pricing():
    tracker = CostTracker()
    tracker.record("gemini-2.5-flash", input_tokens=2000, output_tokens=1000)
    expected = 2000 * (0.15 / 1_000_000) + 1000 * (0.60 / 1_000_000)
    assert abs(tracker.total_cost - expected) < 1e-9


# --- Budget callback and BudgetExceededError tests ---


def test_on_budget_exceeded_callback_called():
    calls = []
    tracker = CostTracker(
        budget=0.0,
        on_budget_exceeded=lambda usage, t: calls.append((usage, t)),
    )
    tracker.record("claude-sonnet-4-6", 1000, 500)
    assert len(calls) == 1
    usage, t = calls[0]
    assert usage.model == "claude-sonnet-4-6"
    assert t is tracker


def test_on_budget_exceeded_callback_not_called_when_under_budget():
    calls = []
    tracker = CostTracker(
        budget=100.0,
        on_budget_exceeded=lambda usage, t: calls.append((usage, t)),
    )
    tracker.record("claude-sonnet-4-6", 100, 50)
    assert len(calls) == 0


def test_raise_on_budget_raises():
    tracker = CostTracker(budget=0.0, raise_on_budget=True)
    with pytest.raises(BudgetExceededError):
        tracker.record("claude-sonnet-4-6", 1000, 500)


def test_raise_on_budget_false_does_not_raise():
    tracker = CostTracker(budget=0.0, raise_on_budget=False)
    usage = tracker.record("claude-sonnet-4-6", 1000, 500)
    assert usage is not None


def test_callback_and_raise_together():
    calls = []
    tracker = CostTracker(
        budget=0.0,
        on_budget_exceeded=lambda usage, t: calls.append(usage),
        raise_on_budget=True,
    )
    with pytest.raises(BudgetExceededError):
        tracker.record("claude-sonnet-4-6", 1000, 500)
    # Callback should have been called before the raise
    assert len(calls) == 1


# --- track_usage decorator tests ---


class _FakeUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, model, input_tokens, output_tokens):
        self.model = model
        self.usage = _FakeUsage(input_tokens, output_tokens)


def test_track_usage_decorator_extracts_usage():
    tracker = CostTracker()

    @track_usage(tracker)
    def call_api():
        return _FakeResponse("gpt-4o", 500, 200)

    resp = call_api()
    assert resp.model == "gpt-4o"
    assert tracker.total_input_tokens == 500
    assert tracker.total_output_tokens == 200
    assert tracker.summary()["num_requests"] == 1


def test_track_usage_decorator_model_override():
    tracker = CostTracker()

    @track_usage(tracker, model="claude-sonnet-4-6")
    def call_api():
        return _FakeResponse("gpt-4o", 300, 100)

    call_api()
    by_model = tracker.cost_by_model()
    assert "claude-sonnet-4-6" in by_model
    assert "gpt-4o" not in by_model


def test_track_usage_decorator_preserves_return_value():
    tracker = CostTracker()

    @track_usage(tracker)
    def call_api():
        return _FakeResponse("gpt-4o", 100, 50)

    resp = call_api()
    assert isinstance(resp, _FakeResponse)
    assert resp.usage.input_tokens == 100


# --- Export method tests ---


def test_to_json_returns_valid_json():
    tracker = CostTracker()
    tracker.record("claude-sonnet-4-6", 1000, 500)
    tracker.record("gpt-4o-mini", 2000, 1000)
    data = json.loads(tracker.to_json())
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["model"] == "claude-sonnet-4-6"
    assert data[1]["model"] == "gpt-4o-mini"
    for rec in data:
        assert "timestamp" in rec
        assert "input_tokens" in rec
        assert "output_tokens" in rec
        assert "cost" in rec


def test_to_json_empty():
    tracker = CostTracker()
    data = json.loads(tracker.to_json())
    assert data == []


def test_to_csv_has_correct_headers():
    tracker = CostTracker()
    tracker.record("claude-sonnet-4-6", 1000, 500)
    csv_str = tracker.to_csv()
    reader = csv.reader(io.StringIO(csv_str))
    headers = next(reader)
    assert headers == ["timestamp", "model", "input_tokens",
                       "output_tokens", "cost"]


def test_to_csv_has_correct_rows():
    tracker = CostTracker()
    tracker.record("claude-sonnet-4-6", 1000, 500)
    tracker.record("gpt-4o", 2000, 800)
    csv_str = tracker.to_csv()
    reader = csv.reader(io.StringIO(csv_str))
    rows = list(reader)
    assert len(rows) == 3  # header + 2 data rows
    assert rows[1][1] == "claude-sonnet-4-6"
    assert rows[2][1] == "gpt-4o"


def test_to_csv_empty():
    tracker = CostTracker()
    csv_str = tracker.to_csv()
    reader = csv.reader(io.StringIO(csv_str))
    rows = list(reader)
    assert len(rows) == 1  # header only


def test_report_contains_cost_info():
    tracker = CostTracker(budget=1.00)
    tracker.record("claude-sonnet-4-6", 1000, 500)
    tracker.record("gpt-4o-mini", 2000, 1000)
    report = tracker.report()
    assert "Total Cost:" in report
    assert "Total Requests:" in report
    assert "Budget:" in report
    assert "Cost by Model:" in report
    assert "claude-sonnet-4-6" in report
    assert "gpt-4o-mini" in report


def test_report_no_budget():
    tracker = CostTracker()
    tracker.record("gpt-4o", 500, 200)
    report = tracker.report()
    assert "Total Cost:" in report
    assert "Budget:" not in report


def test_report_shows_correct_request_count():
    tracker = CostTracker()
    tracker.record("gpt-4o", 100, 50)
    tracker.record("gpt-4o", 200, 100)
    tracker.record("gpt-4o", 300, 150)
    report = tracker.report()
    assert "3" in report


# --- Session tests ---


def test_session_tracks_cost_independently():
    tracker = CostTracker()
    s1 = tracker.session("task-1")
    s2 = tracker.session("task-2")
    s1.record("gpt-4o", 1000, 500)
    s2.record("claude-sonnet-4-6", 2000, 1000)
    # Each session only sees its own cost
    expected_s1 = 1000 * (2.50 / 1e6) + 500 * (10.0 / 1e6)
    expected_s2 = 2000 * (3.0 / 1e6) + 1000 * (15.0 / 1e6)
    assert abs(s1.total_cost - expected_s1) < 1e-9
    assert abs(s2.total_cost - expected_s2) < 1e-9


def test_session_records_propagate_to_parent_tracker():
    tracker = CostTracker()
    s = tracker.session("task-1")
    s.record("gpt-4o", 1000, 500)
    s.record("gpt-4o", 2000, 800)
    # Parent tracker should have both records
    assert tracker.total_input_tokens == 3000
    assert tracker.total_output_tokens == 1300
    assert tracker.summary()["num_requests"] == 2
    assert abs(tracker.total_cost - s.total_cost) < 1e-9


def test_cost_by_session_returns_correct_breakdown():
    tracker = CostTracker()
    s1 = tracker.session("task-1")
    s2 = tracker.session("task-2")
    s1.record("gpt-4o", 1000, 500)
    s2.record("gpt-4o", 2000, 1000)
    by_session = tracker.cost_by_session()
    assert "task-1" in by_session
    assert "task-2" in by_session
    expected_s1 = round(1000 * (2.50 / 1e6) + 500 * (10.0 / 1e6), 6)
    expected_s2 = round(2000 * (2.50 / 1e6) + 1000 * (10.0 / 1e6), 6)
    assert by_session["task-1"] == expected_s1
    assert by_session["task-2"] == expected_s2


def test_session_context_manager():
    tracker = CostTracker()
    with tracker.session("ctx-task") as s:
        s.record("claude-sonnet-4-6", 500, 200)
        s.record("claude-sonnet-4-6", 300, 100)
    assert s.summary()["num_requests"] == 2
    assert tracker.summary()["num_requests"] == 2
    assert abs(tracker.total_cost - s.total_cost) < 1e-9


def test_session_summary_includes_name():
    tracker = CostTracker()
    s = tracker.session("my-session")
    s.record("gpt-4o", 100, 50)
    summary = s.summary()
    assert summary["session_name"] == "my-session"
    assert summary["num_requests"] == 1
    assert summary["total_input_tokens"] == 100
    assert summary["total_output_tokens"] == 50


def test_session_returns_same_instance_for_same_name():
    tracker = CostTracker()
    s1 = tracker.session("task-1")
    s2 = tracker.session("task-1")
    assert s1 is s2


def test_cost_by_session_empty():
    tracker = CostTracker()
    assert tracker.cost_by_session() == {}


def test_session_with_budget_enforcement():
    tracker = CostTracker(budget=0.0, raise_on_budget=True)
    s = tracker.session("expensive-task")
    with pytest.raises(BudgetExceededError):
        s.record("gpt-4o", 1000, 500)


# --- Persistence tests ---


def test_save_creates_file():
    tracker = CostTracker(budget=5.0)
    tracker.record("gpt-4o", 1000, 500)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        tracker.save(path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert data["budget"] == 5.0
        assert len(data["usages"]) == 1


def test_load_restores_state():
    tracker = CostTracker(budget=2.0)
    tracker.record("gpt-4o", 1000, 500)
    tracker.record("claude-sonnet-4-6", 2000, 1000)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        tracker.save(path)
        loaded = CostTracker.load(path)
    assert loaded.budget == 2.0
    assert loaded.total_input_tokens == 3000
    assert loaded.total_output_tokens == 1500
    assert abs(loaded.total_cost - tracker.total_cost) < 1e-9
    assert loaded.summary()["num_requests"] == 2


def test_load_restores_sessions():
    tracker = CostTracker()
    s = tracker.session("task-1")
    s.record("gpt-4o", 1000, 500)
    s.record("gpt-4o", 500, 200)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        tracker.save(path)
        loaded = CostTracker.load(path)
    by_session = loaded.cost_by_session()
    assert "task-1" in by_session
    assert abs(by_session["task-1"] - tracker.cost_by_session()["task-1"]) < 1e-9


def test_load_restores_budget():
    tracker = CostTracker(budget=10.0)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.json")
        tracker.save(path)
        loaded = CostTracker.load(path)
    assert loaded.budget == 10.0


def test_auto_save_writes_on_each_record():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "auto.json")
        tracker = CostTracker(auto_save=path)
        tracker.record("gpt-4o", 1000, 500)
        # File should exist after first record
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data["usages"]) == 1
        tracker.record("gpt-4o", 2000, 800)
        with open(path) as f:
            data = json.load(f)
        assert len(data["usages"]) == 2


def test_load_missing_file_returns_fresh_tracker():
    loaded = CostTracker.load("/nonexistent/path/state.json")
    assert loaded.total_cost == 0.0
    assert loaded.summary()["num_requests"] == 0
    assert loaded.budget is None


def test_load_corrupted_file_returns_fresh_tracker():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bad.json")
        with open(path, "w") as f:
            f.write("not valid json {{{")
        loaded = CostTracker.load(path)
        assert loaded.total_cost == 0.0
        assert loaded.summary()["num_requests"] == 0


def test_load_non_dict_json_returns_fresh_tracker():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bad.json")
        with open(path, "w") as f:
            json.dump([1, 2, 3], f)
        loaded = CostTracker.load(path)
        assert loaded.total_cost == 0.0


# --- Anomaly detection tests ---


def test_anomaly_detected_on_spike():
    tracker = CostTracker()
    # Record 5 normal usages (small)
    for _ in range(5):
        tracker.record("gpt-4o-mini", 100, 50)
    # Now record a spike (much larger)
    spike_usage = tracker.record("claude-opus-4-6", 100000, 50000)
    anomaly = tracker.check_anomaly(spike_usage)
    assert anomaly is not None
    assert anomaly["type"] == "spike"
    assert anomaly["ratio"] > 3.0
    assert "cost" in anomaly
    assert "avg_cost" in anomaly


def test_no_anomaly_on_normal_usage():
    tracker = CostTracker()
    # Record 5 normal usages with the same model and tokens
    for _ in range(5):
        tracker.record("gpt-4o", 1000, 500)
    # Record another similar usage
    normal_usage = tracker.record("gpt-4o", 1000, 500)
    anomaly = tracker.check_anomaly(normal_usage)
    assert anomaly is None


def test_no_anomaly_with_fewer_than_5_records():
    tracker = CostTracker()
    for _ in range(4):
        tracker.record("gpt-4o", 1000, 500)
    # Only 4 records, spike should not trigger anomaly
    spike_usage = tracker.record("claude-opus-4-6", 100000, 50000)
    anomaly = tracker.check_anomaly(spike_usage)
    assert anomaly is None


def test_on_anomaly_callback_called():
    anomalies = []
    tracker = CostTracker(
        on_anomaly=lambda anomaly, usage, t: anomalies.append(
            (anomaly, usage, t)
        )
    )
    for _ in range(5):
        tracker.record("gpt-4o-mini", 100, 50)
    tracker.record("claude-opus-4-6", 100000, 50000)
    assert len(anomalies) == 1
    anomaly, usage, t = anomalies[0]
    assert anomaly["type"] == "spike"
    assert usage.model == "claude-opus-4-6"
    assert t is tracker


def test_on_anomaly_callback_not_called_on_normal():
    anomalies = []
    tracker = CostTracker(
        on_anomaly=lambda anomaly, usage, t: anomalies.append(anomaly)
    )
    for _ in range(6):
        tracker.record("gpt-4o", 1000, 500)
    assert len(anomalies) == 0


# --- Rate tracking tests ---


def test_cost_per_minute():
    tracker = CostTracker()
    base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    # Manually create usages with controlled timestamps
    for i in range(5):
        usage = Usage(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            timestamp=(base_time + timedelta(minutes=i)).isoformat(),
        )
        tracker._usages.append(usage)
        tracker._total_input_tokens += usage.input_tokens
        tracker._total_output_tokens += usage.output_tokens
        tracker._total_cost += usage.cost
    # 5 usages over 4 minutes
    cpm = tracker.cost_per_minute()
    assert cpm > 0
    expected_total_cost = tracker.total_cost
    expected_cpm = expected_total_cost / 4.0
    assert abs(cpm - expected_cpm) < 1e-9


def test_requests_per_minute():
    tracker = CostTracker()
    base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(6):
        usage = Usage(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            timestamp=(base_time + timedelta(minutes=i * 2)).isoformat(),
        )
        tracker._usages.append(usage)
        tracker._total_input_tokens += usage.input_tokens
        tracker._total_output_tokens += usage.output_tokens
        tracker._total_cost += usage.cost
    # 6 usages over 10 minutes
    rpm = tracker.requests_per_minute()
    expected_rpm = 6 / 10.0
    assert abs(rpm - expected_rpm) < 1e-9


def test_cost_per_minute_single_record():
    tracker = CostTracker()
    tracker.record("gpt-4o", 1000, 500)
    assert tracker.cost_per_minute() == 0.0


def test_requests_per_minute_single_record():
    tracker = CostTracker()
    tracker.record("gpt-4o", 1000, 500)
    assert tracker.requests_per_minute() == 0.0
