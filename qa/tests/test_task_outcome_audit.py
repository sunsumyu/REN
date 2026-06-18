import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class TaskOutcomeAuditTests(unittest.TestCase):
    def test_write_task_outcome_appends_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "generation_task_outcomes.jsonl"
            with patch.object(main, "TASK_OUTCOMES_PATH", str(target)), patch.object(main, "LOGS_DIR", tmpdir):
                main.write_task_outcome({
                    "task_label": "Task-1",
                    "final_status": "exception",
                    "failed_stage": "runtime",
                    "root_cause": "ValueError: bad payload",
                })

            rows = target.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 1)
            payload = json.loads(rows[0])
            self.assertEqual(payload["task_label"], "Task-1")
            self.assertEqual(payload["final_status"], "exception")
            self.assertEqual(payload["failed_stage"], "runtime")
            self.assertIn("time", payload)


if __name__ == "__main__":
    unittest.main()
