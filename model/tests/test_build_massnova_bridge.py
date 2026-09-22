import json
import subprocess
import sys
import unittest
from pathlib import Path


class MassNovaBridgeBuildLayoutTests(unittest.TestCase):
    def test_default_layout_keeps_all_outputs_under_model_build(self):
        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / "model" / "tools" / "build_massnova_bridge.py"

        completed = subprocess.run(
            [sys.executable, str(script), "--print-layout"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        layout = json.loads(completed.stdout)

        build_root = (repo_root / "model" / "build").resolve()
        for name in ("generated", "temp", "python"):
            path = Path(layout[name]).resolve()
            self.assertTrue(path.is_relative_to(build_root), (name, path))
        self.assertEqual(Path(layout["extensions"]).resolve(),
                         (build_root / "python" / "inference").resolve())


if __name__ == "__main__":
    unittest.main()
