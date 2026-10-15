import os
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from s1ced import hub
from s1ced import playground as app


class ResolveTests(unittest.TestCase):
    def test_local_release_folder_is_used_as_is(self):
        with tempfile.TemporaryDirectory() as d:
            for f in hub.REQUIRED: Path(d, f).write_text('{}')
            self.assertEqual(hub.resolve_release(d), Path(d))

    def test_incomplete_folder_names_missing_files(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'config.json').write_text('{}')
            with self.assertRaisesRegex(FileNotFoundError, 'model.safetensors'): hub.resolve_release(d)

    def test_repo_id_downloads_only_release_files(self):
        with patch('huggingface_hub.snapshot_download', return_value='/cache/snap') as dl, \
             patch('huggingface_hub.try_to_load_from_cache', return_value='/cache/snap/model.safetensors'):
            self.assertEqual(hub.resolve_release('someone/some-model'), Path('/cache/snap'))
        patterns = dl.call_args.kwargs['allow_patterns']
        self.assertIn('model.safetensors', patterns); self.assertNotIn('*', patterns)

    def test_default_and_environment_override(self):
        with patch('huggingface_hub.snapshot_download', side_effect=lambda repo, **k: repo) as dl, \
             patch('huggingface_hub.try_to_load_from_cache', return_value=None), \
             patch.dict(os.environ, {'S1CED_MODEL': ''}):
            self.assertEqual(str(hub.resolve_release()), str(Path(hub.DEFAULT_MODEL)))
            with patch.dict(os.environ, {'S1CED_MODEL': 'org/other'}):
                self.assertEqual(str(hub.resolve_release()), str(Path('org/other')))

    def test_nonsense_is_a_clear_error_without_network(self):
        with patch('huggingface_hub.snapshot_download', side_effect=AssertionError('no network')):
            with self.assertRaisesRegex(FileNotFoundError, 'neither'): hub.resolve_release('no/such/folder/here')


class BindTests(unittest.TestCase):
    def args(self, port=None):
        return SimpleNamespace(host='127.0.0.1', port=port, remote_url='https://example.test', remote_model='test')

    def test_busy_default_port_moves_to_next_free_port(self):
        with socket.socket() as busy:
            busy.bind(('127.0.0.1', 0)); busy.listen()
            with patch.object(app, 'DEFAULT_PORT', busy.getsockname()[1]):
                a = self.args(); srv = app.bind(SimpleNamespace(), SimpleNamespace(key=None), a)
                try:
                    self.assertNotEqual(a.port, busy.getsockname()[1])
                    self.assertEqual(srv.server_address[1], a.port)
                finally: srv.server_close()

    def test_busy_explicit_port_exits_with_message(self):
        with socket.socket() as busy:
            busy.bind(('127.0.0.1', 0)); busy.listen()
            with self.assertRaises(SystemExit) as cm:
                app.bind(SimpleNamespace(), SimpleNamespace(key=None), self.args(busy.getsockname()[1]))
            self.assertIn('--port', str(cm.exception.code))


if __name__ == '__main__':
    unittest.main()
