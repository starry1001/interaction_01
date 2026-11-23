import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import generate


class GenerateSelectionTests(unittest.TestCase):
    def test_help_describes_token_budget(self):
        output = io.StringIO()
        with patch('sys.argv', ['generate.py', '--help']), contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as result:
                generate.main()
        self.assertEqual(result.exception.code, 0)
        self.assertIn('60%', output.getvalue())

    def test_only_requested_ids_are_submitted(self):
        for option in ('--task', '--ids'):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / 'data').mkdir()
                args = ['generate.py', '--data', str(root / 'data.jsonl'), '--result', str(root / 'out'),
                        '--model', 'mock', option, '3', '--setting', '1']
                with patch('sys.argv', args), patch.object(generate.utils, 'load_examples', return_value=[{'id': i} for i in (1, 2, 3)]), \
                     patch.object(generate, 'process_task', return_value=('3', 'Done')) as call, contextlib.redirect_stdout(io.StringIO()):
                    generate.main()
                self.assertEqual(call.call_count, 1)
                self.assertEqual(call.call_args.args[0]['id'], 3)
                self.assertEqual(call.call_args.args[-1], 1)

    def test_missing_ids_fail_before_any_task_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'data').mkdir()
            args = ['generate.py', '--data', str(root / 'data.jsonl'), '--result', str(root / 'out'), '--task', '999']
            with patch('sys.argv', args), patch.object(generate.utils, 'load_examples', return_value=[{'id': 1}]), \
                 patch.object(generate, 'process_task') as call, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    generate.main()
            call.assert_not_called()


if __name__ == '__main__':
    unittest.main()
