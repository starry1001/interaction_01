import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from config import PROMPT_TEMPLATE
from generate import process_task
from utils import core
from utils.screenshot_delta import analyze_pair, prepare_delta_screenshots


class ScreenshotDeltaTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def save(self, name, image):
        path = self.root / name
        Image.fromarray(image).save(path)
        return str(path)

    def test_local_change_has_context_and_preserves_all_sides(self):
        before = np.full((300, 500, 3), 240, dtype=np.uint8)
        after = before.copy()
        after[100:140, 200:260] = 20
        decision = analyze_pair(before, after)
        self.assertFalse(decision['scene_change'])
        x1, y1, x2, y2 = decision['box']
        self.assertLess(x1, 200)
        self.assertLess(y1, 100)
        self.assertGreater(x2, 260)
        self.assertGreater(y2, 140)

    def test_thin_progress_bar_and_small_text_edit_survive(self):
        before = np.full((300, 500, 3), 240, dtype=np.uint8)
        for box in ((200, 200, 260, 202), (210, 80, 213, 89)):
            x1, y1, x2, y2 = box
            after = before.copy()
            after[y1:y2, x1:x2] = 0
            self.assertIsNotNone(analyze_pair(before, after)['box'])

    def test_noise_and_identical_frames_keep_full(self):
        before = np.full((300, 500, 3), 240, dtype=np.uint8)
        self.assertIsNone(analyze_pair(before, before)['box'])
        noise = np.random.default_rng(0).integers(-3, 4, before.shape)
        after = (before.astype(np.int16) + noise).astype(np.uint8)
        self.assertIsNone(analyze_pair(before, after)['box'])

    def test_scene_change_dimensions_and_dispersed_changes_keep_full(self):
        before = np.full((300, 500, 3), 240, dtype=np.uint8)
        self.assertTrue(analyze_pair(before, np.zeros_like(before))['scene_change'])
        self.assertTrue(analyze_pair(before, before[:200])['scene_change'])
        after = before.copy()
        after[:40, :40] = 0
        after[-40:, -40:] = 0
        self.assertIsNone(analyze_pair(before, after)['box'])

    def test_scene_union_and_exact_crop_pixels(self):
        first = np.full((300, 500, 3), 240, dtype=np.uint8)
        second = first.copy()
        second[80:110, 200:230] = 0
        third = second.copy()
        third[160:190, 200:230] = 0
        paths = [self.save(f'step_{i}.png', im) for i, im in enumerate((first, second, third))]
        original = [Path(p).read_bytes() for p in paths]
        out = self.root / 'prepared'
        prepared = prepare_delta_screenshots(paths, out)
        records = json.loads((out / 'screenshot_manifest.json').read_text())['screenshots']
        self.assertEqual(records[1]['box'], records[2]['box'])
        self.assertEqual(Path(prepared[0]).read_bytes(), original[0])
        for index in (1, 2):
            x1, y1, x2, y2 = records[index]['box']
            with Image.open(prepared[index]) as cropped:
                np.testing.assert_array_equal(np.array(cropped), (second, third)[index - 1][y1:y2, x1:x2])
        self.assertEqual([Path(p).read_bytes() for p in paths], original)

    def test_generation_payload_coordinates_order_and_unchanged_prompt(self):
        first = np.full((300, 500, 3), 240, dtype=np.uint8)
        second = first.copy()
        second[80:110, 200:230] = 0
        paths = [self.save(f'step_{i}.png', im) for i, im in enumerate((first, second))]
        asset = self.save('asset.png', first[:20, :20])
        captured = []
        def fake_call(model, content, **kwargs):
            captured.extend(content)
            return '<html>ok</html>'
        with patch.object(core, 'call_mllm', side_effect=fake_call):
            core.call_mllm_html('test-model', [asset], paths, str(self.root / 'payload'), '500x300', setting=3)
        self.assertEqual(captured[0]['text'], PROMPT_TEMPLATE.format(resolution='500x300'))
        self.assertEqual([p['type'] for p in captured],
                         ['text', 'image_url', 'text', 'image_url', 'text', 'image_url'])
        self.assertIn('screenshot_0', captured[2]['text'])
        self.assertIn('Full webpage', captured[2]['text'])
        self.assertIn('crop bounds', captured[4]['text'])
        self.assertIn('screenshot_0', captured[4]['text'])
        crop_bytes = base64.b64decode(captured[5]['image_url']['url'].split(',', 1)[1])
        self.assertLess(Image.open(io.BytesIO(crop_bytes)).width, 500)

    def test_direct_setting_keeps_dimensions(self):
        source = np.zeros((300, 600, 3), dtype=np.uint8)
        paths = [self.save(f'previous_{i}.png', source) for i in range(5)]
        for setting, expected in ((0, [(600, 300)] * 5),):
            result = core.prepare_images_in_temp_folder([], paths, str(self.root / f'setting{setting}'), {}, setting)
            sizes = []
            for path in result:
                with Image.open(path) as image:
                    sizes.append(image.size)
            self.assertEqual(sizes, expected)

    def test_mock_generation_saves_manifest_with_backup(self):
        task = self.root / 'data' / '1' / 'checkpoints'
        task.mkdir(parents=True)
        Image.new('RGB', (500, 300), 'white').save(task / 'step_0.png')
        status = process_task({'id': 1}, str(self.root / 'data'), str(self.root / 'outputs'), 'mock', setting=3)
        self.assertEqual(status, ('1', 'Done'))
        output = self.root / 'outputs' / '1' / 'mock_setting3'
        self.assertTrue((output / 'screenshot_manifest.json').exists())
        self.assertEqual(len(list((output / 'backups').glob('*/screenshot_manifest.json'))), 1)


if __name__ == '__main__':
    unittest.main()
