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
from preview_setting3 import render_task
from setting3 import DeltaOptions


class Setting3GenerationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def save(self, name, image):
        path = self.root / name
        Image.fromarray(image).save(path)
        return str(path)

    def test_generation_payload_contains_only_original_prompt_and_processed_images(self):
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
                         ['text', 'image_url', 'image_url', 'image_url'])
        self.assertEqual(base64.b64decode(captured[1]['image_url']['url'].split(',', 1)[1]), Path(asset).read_bytes())
        self.assertEqual(base64.b64decode(captured[2]['image_url']['url'].split(',', 1)[1]), Path(paths[0]).read_bytes())
        crop_bytes = base64.b64decode(captured[3]['image_url']['url'].split(',', 1)[1])
        record = json.loads((self.root / 'payload/screenshot_manifest.json').read_text())['screenshots'][1]
        self.assertFalse(json.loads((self.root / 'payload/screenshot_manifest.json').read_text())['preview_only'])
        with Image.open(io.BytesIO(crop_bytes)) as image:
            x1, y1, x2, y2 = record['box']
            np.testing.assert_array_equal(np.array(image), second[y1:y2, x1:x2])
        self.assertLess(Image.open(io.BytesIO(crop_bytes)).width, 500)

    def test_direct_setting_keeps_dimensions(self):
        source = np.zeros((300, 1200, 3), dtype=np.uint8)
        paths = [self.save(f'previous_{i}.png', source) for i in range(5)]
        for setting, expected in ((0, [(1200, 300)] * 5),):
            result = core.prepare_images_in_temp_folder([], paths, str(self.root / f'setting{setting}'), setting=setting)
            sizes = []
            for path in result:
                with Image.open(path) as image:
                    sizes.append(image.size)
            self.assertEqual(sizes, expected)

    def test_generate_and_preview_produce_identical_unscaled_images(self):
        task = self.root / 'data/1'
        (task / 'checkpoints').mkdir(parents=True)
        before = np.full((1600, 1000, 3), 240, np.uint8)
        after = before.copy()
        after[500:1100, 300:800] = 0
        for i, frame in enumerate((before, after)):
            Image.fromarray(frame).save(task / 'checkpoints' / f'step_{i}.png')
        render_task(task, self.root / 'preview/1', DeltaOptions())
        status = process_task({'id': 1}, str(self.root / 'data'), str(self.root / 'generation'),
                              'mock', setting=3)
        self.assertEqual(status, ('1', 'Done'))
        output = self.root / 'generation/1/mock_setting3'
        manifest = json.loads((output / 'screenshot_manifest.json').read_text())
        for record in manifest['screenshots']:
            name = record['output']
            self.assertEqual((output / 'temp_images' / name).read_bytes(),
                             (self.root / 'preview/1/temp_images' / name).read_bytes())
            x1, y1, x2, y2 = record['box']
            self.assertEqual(record['output_size'], [x2 - x1, y2 - y1])

    def test_mock_generation_saves_manifest_without_backup(self):
        task = self.root / 'data' / '1' / 'checkpoints'
        task.mkdir(parents=True)
        Image.new('RGB', (500, 300), 'white').save(task / 'step_0.png')
        status = process_task({'id': 1}, str(self.root / 'data'), str(self.root / 'outputs'), 'mock', setting=3)
        self.assertEqual(status, ('1', 'Done'))
        output = self.root / 'outputs' / '1' / 'mock_setting3'
        self.assertTrue((output / 'screenshot_manifest.json').exists())
        self.assertTrue((output / 'index.html').exists())
        self.assertTrue((output / 'raw_response.txt').exists())
        self.assertFalse((output / 'backups').exists())


if __name__ == '__main__':
    unittest.main()
