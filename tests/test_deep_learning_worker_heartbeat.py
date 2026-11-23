from __future__ import annotations

import threading
import time

from src.deep_learning.worker import _WorkerHeartbeat


class _HeartbeatRegistry:
    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()

    def heartbeat_job(self, job_id: str, *, worker_token: str) -> bool:
        assert job_id == "job-test"
        assert worker_token == "lease-test"
        with self.lock:
            self.calls += 1
        return True


def test_worker_heartbeat_refreshes_lease_and_stops_cleanly(monkeypatch):
    monkeypatch.setenv("EVA_DL_WORKER_HEARTBEAT_INTERVAL_SECONDS", "1")
    registry = _HeartbeatRegistry()
    heartbeat = _WorkerHeartbeat("job-test", "lease-test", registry)

    heartbeat.start()
    time.sleep(1.15)
    heartbeat.stop()
    calls_after_stop = registry.calls
    time.sleep(0.1)

    assert calls_after_stop >= 1
    assert registry.calls == calls_after_stop