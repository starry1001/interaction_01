import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils import core
from utils.screenshot_delta_adaptive import prepare_delta_screenshots as production_prepare
from utils.screenshot_delta_preview import analyze_pair, prepare_delta_screenshots


ROOT = Path(__file__).resolve().parents[1]


class PreviewDeltaTests(unittest.TestCase):
    def test_production_uses_promoted_adaptive_algorithm(self):
        self.assertIs(core.prepare_delta_screenshots, production_prepare)

    def test_flat_recolor_is_not_enough_to_claim_same_layout(self):
        before = np.full((300, 500, 3), 40, np.uint8)
        after = np.full_like(before, 160)
        self.assertIsNone(analyze_pair(before, after)['box'])

    def test_crop_snapping_preserves_original_coordinates_and_first_image(self):
        before = np.full((1600, 1000, 3), 240, np.uint8)
        after = before.copy()
        after[500:1100, 300:800] = 0
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = []
            for i, frame in enumerate((before, after)):
                path = root / f'step_{i}.png'
                Image.fromarray(frame).save(path)
                paths.append(str(path))
            manifests = []
            for margin in (0, 100):
                output = root / str(margin)
                prepared = prepare_delta_screenshots(paths, output, tile_snap_margin=margin)
                records = json.loads((output / 'screenshot_manifest.json').read_text())['screenshots']
                self.assertEqual(Path(prepared[0]).read_bytes(), Path(paths[0]).read_bytes())
                manifests.append(records)
            a, b = manifests[0][1], manifests[1][1]
            self.assertEqual(a['box'], b['box'])
            self.assertEqual(a['mode'], 'crop')
            self.assertGreater(b['tile_snap']['extra_saved_pixels'], 0)
            self.assertLess(b['tile_snap']['estimated_tiles_after'], b['tile_snap']['estimated_tiles_before'])

    def test_resized_crop_payload_reports_actual_size(self):
        from unittest.mock import patch
        before = np.full((1600, 1000, 3), 240, np.uint8)
        after = before.copy()
        after[500:1100, 300:800] = 0
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = []
            for i, frame in enumerate((before, after)):
                path = root / f'step_{i}.png'
                Image.fromarray(frame).save(path)
                paths.append(str(path))
            with patch.object(core, 'call_mllm', return_value='<html></html>') as call:
                core.call_mllm_html('gpt-5', [], paths, str(root / 'payload'), '1000x1600', setting=3)
            text = '\n'.join(block.get('text', '') for block in call.call_args.args[1])
            self.assertIn('proportionally resized', text)
            self.assertNotIn('Unscaled crop', text)

    def test_per_step_boxes_do_not_inherit_distant_changes(self):
        a = np.full((400, 600, 3), 240, np.uint8)
        b = a.copy()
        b[80:110, 450:490] = 0
        c = b.copy()
        c[200:230, 100:140] = 0
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = []
            for i, frame in enumerate((a, b, c)):
                path = root / f'step_{i}.png'
                Image.fromarray(frame).save(path)
                paths.append(str(path))
            output = root / 'out'
            prepared = prepare_delta_screenshots(paths, output)
            records = json.loads((output / 'screenshot_manifest.json').read_text())['screenshots']
            self.assertGreater(records[1]['box'][0], 400)
            self.assertLess(records[2]['box'][2], 200)
            for i in (1, 2):
                x1, y1, x2, y2 = records[i]['box']
                with Image.open(prepared[i]) as image:
                    np.testing.assert_array_equal(np.array(image), (b, c)[i - 1][y1:y2, x1:x2])

    def test_pomofocus_theme_switch_keeps_scene_and_records_palette(self):
        directory = ROOT / 'data/data_backup/3/checkpoints'
        if not directory.is_dir():
            self.skipTest('Example dataset is not installed')
        for index in (1, 2):
            previous = cv2.imread(str(directory / f'step_{index - 1}.png'))
            current = cv2.imread(str(directory / f'step_{index}.png'))
            result = analyze_pair(previous, current)
            self.assertFalse(result['scene_change'])
            self.assertIsNotNone(result['box'])
            self.assertTrue(result['theme_changes'])
            # The whole timer is inside the crop, including unchanged digits at the right.
            self.assertLess(result['box'][0], 125)
            self.assertGreater(result['box'][2], 520)
            self.assertLess(result['metrics']['structural_crop_fraction'], 0.65)

    def test_music_page_switch_still_keeps_full(self):
        directory = ROOT / 'data/data_backup/2/checkpoints'
        if not directory.is_dir():
            self.skipTest('Example dataset is not installed')
        previous = cv2.imread(str(directory / 'step_1.png'))
        current = cv2.imread(str(directory / 'step_2.png'))
        result = analyze_pair(previous, current)
        self.assertTrue(result['scene_change'])
        self.assertIsNone(result['box'])


if __name__ == '__main__':
    unittest.main()
