from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_accuracy_eval_metrics_cookbook_uses_run_metrics_keyword():
    source = (REPO_ROOT / "cookbook" / "09_evals" / "accuracy" / "accuracy_eval_metrics.py").read_text(encoding="utf-8")

    assert "run_metrics=run_output.metrics" in source
    assert "run_response=run_output" not in source


def test_team_metrics_cookbooks_use_leader_and_member_labels():
    team_metrics_source = (REPO_ROOT / "cookbook" / "03_teams" / "22_metrics" / "01_team_metrics.py").read_text(
        encoding="utf-8"
    )
    team_tool_metrics_source = (
        REPO_ROOT / "cookbook" / "03_teams" / "22_metrics" / "04_team_tool_metrics.py"
    ).read_text(encoding="utf-8")

    assert "TEAM LEADER RUN METRICS" in team_metrics_source
    assert "AGGREGATED TEAM METRICS" not in team_metrics_source
    assert "TEAM LEADER RUN METRICS" in team_tool_metrics_source
    assert "MEMBER RUN METRICS" in team_tool_metrics_source
    assert "AGGREGATED TEAM METRICS" not in team_tool_metrics_source
