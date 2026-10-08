"""Check launcher budgets, device propagation and working directory without training."""
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_root / 'scripts'), str(_root / 'src')]
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def launch(self, **settings):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher = root / 'scripts/canonical_sweep.sh'
            launcher.parent.mkdir(parents=True)
            source = Path(__file__).resolve().parents[1].joinpath('scripts', 'canonical_sweep.sh').read_text()
            # Capture output with subprocess instead of tee's process
            # substitution, which some test sandboxes prohibit.
            source = source.replace('exec > >(tee -a "$STORE_DIR/sweep.log") 2>&1', ':')
            launcher.write_text(source)
            recorder = root / 'python-recorder'
            recorder.write_text(f'#!{sys.executable}\n' +
                'import json, os, sys\n'
                'with open(os.environ["LAUNCHER_TEST_LOG"], "a") as f:\n'
                '    f.write(json.dumps({"args": sys.argv[1:], "cwd": os.getcwd()}) + "\\n")\n'
                'if sys.argv[1:] == ["-"]: sys.stdin.read()\n')
            recorder.chmod(0o700)
            log = root / 'calls.jsonl'
            (root / 'elsewhere').mkdir()
            env = {key: value for key, value in os.environ.items()
                   if key in ('PATH', 'HOME', 'SYSTEMROOT', 'TMPDIR')}
            env.update(PY=str(recorder), STORE_DIR=str(root / 'store'),
                       LAUNCHER_TEST_LOG=str(log), DEVICE='cpu', N_GPUS='1',
                       **settings)
            # Start elsewhere: the launcher must change to the root it lives in.
            subprocess.run(['bash', str(launcher)], cwd=root / 'elsewhere', env=env,
                           check=True, capture_output=True, text=True, timeout=30)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertTrue(calls)
            for call in calls:
                # The launcher calls its scripts by paths relative to the repository root.
                self.assertEqual(Path(call['cwd']).resolve(), root.resolve(), call['args'])
            return {Path(call['args'][0]).name: call['args'] for call in calls
                    if call['args'] and call['args'][0].endswith('.py')}

    def test_smoke_budgets_and_cpu_device(self):
        calls = self.launch(SMOKE='1')
        resolver = calls['resolve_monotonicity_violations.py']
        self.assertEqual(resolver[resolver.index('--K') + 1], '4')
        self.assertEqual(resolver[resolver.index('--steps-base') + 1], '1500')
        for args in calls.values():
            self.assertEqual(args[args.index('--device') + 1], 'cpu')

    def test_full_budgets(self):
        resolver = self.launch(SMOKE='0')['resolve_monotonicity_violations.py']
        self.assertEqual(resolver[resolver.index('--K') + 1], '30')
        self.assertEqual(resolver[resolver.index('--steps-base') + 1], '24000')

    def test_smoke_budget_overrides(self):
        resolver = self.launch(SMOKE='1', RESOLVE_K='7', RESOLVE_STEPS_BASE='2500')[
            'resolve_monotonicity_violations.py']
        self.assertEqual(resolver[resolver.index('--K') + 1], '7')
        self.assertEqual(resolver[resolver.index('--steps-base') + 1], '2500')


if __name__ == '__main__':
    unittest.main()
