"""Tests for bot/healthcheck.py — CR32 heartbeat-based container health."""
import time

import bot.healthcheck as hc


def test_unhealthy_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "HEARTBEAT_FILE", tmp_path / "nope")
    assert hc.is_healthy() is False
    assert hc.main() == 1


def test_healthy_with_fresh_heartbeat(tmp_path, monkeypatch):
    hb = tmp_path / "hb"
    hb.write_text(str(time.time()))
    monkeypatch.setattr(hc, "HEARTBEAT_FILE", hb)
    assert hc.is_healthy() is True
    assert hc.main() == 0


def test_unhealthy_with_stale_heartbeat(tmp_path, monkeypatch):
    hb = tmp_path / "hb"
    hb.write_text(str(time.time() - (hc.MAX_AGE_SECONDS + 10)))
    monkeypatch.setattr(hc, "HEARTBEAT_FILE", hb)
    assert hc.is_healthy() is False


def test_unhealthy_with_garbage_heartbeat(tmp_path, monkeypatch):
    hb = tmp_path / "hb"
    hb.write_text("not-a-timestamp")
    monkeypatch.setattr(hc, "HEARTBEAT_FILE", hb)
    assert hc.is_healthy() is False
