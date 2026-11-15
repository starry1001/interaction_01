import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from utils import core
from utils.pixel_budget import allocate_targets, resize_to_area, snap_to_tile_boundary, tile_count
from generate import process_task


class PixelBudgetTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def frames(self, count=5):
        paths = []
        frame = np.full((300, 600, 3), 240, np.uint8)
        for i in range(count):
            frame = frame.copy()
            frame[70:110, 160 + i * 20:190 + i * 20] = i * 20
            path = self.root / f'step_{i}.png'
            Image.fromarray(frame).save(path)
            paths.append(str(path))
        return paths

    def test_equal_savings_and_saturation(self):
        self.assertEqual(allocate_targets([1000, 1000, 1000, 1000], 2000), [500] * 4)
        self.assertEqual(allocate_targets([3000, 2000], 2000, [3, 2]), [1800, 1200])
        result = allocate_targets([10, 1000], 800)
        self.assertEqual(result, [1, 209])

    def test_both_settings_match_actual_setting3_area(self):
        paths = self.frames()
        original = [Path(p).read_bytes() for p in paths]
        reference = core.prepare_images_in_temp_folder([], paths, str(self.root / 's3'), {}, 3)
        target = 0
        for path in reference:
            with Image.open(path) as image:
                target += image.width * image.height
        for setting in (1, 2):
            out = self.root / f's{setting}'
            result = core.prepare_images_in_temp_folder([], paths, str(out), {}, setting)
            budget = json.loads((out / 'compression_budget.json').read_text())
            self.assertEqual(budget['setting3_target_pixels'], target)
            self.assertLessEqual(budget['actual_screenshot_pixels'], target)
            self.assertLess(budget['rounding_shortfall_pixels'], target * 0.02)
            self.assertAlmostEqual(sum(r['target_pixels'] for r in budget['allocations']), target)
            if setting == 1:
                self.assertEqual(Path(result[0]).read_bytes(), original[0])
                savings = [r['assigned_saved_pixels'] for r in budget['allocations'][1:]]
                self.assertTrue(all(abs(s - savings[0]) < 1e-6 for s in savings))
            else:
                self.assertEqual(len(result), 2)
                first, second = budget['allocations']
                self.assertEqual((first['screenshot_count'], second['screenshot_count']), (3, 2))
                self.assertAlmostEqual(first['assigned_saved_pixels'] / second['assigned_saved_pixels'], 1.5)
                self.assertEqual(first['raw_sheet_size'], [1800, 332])
                self.assertEqual(second['raw_sheet_size'], [1200, 332])
            for record in budget['allocations']:
                w, h = record['output_size']
                original_w, original_h = record.get('raw_sheet_size', [600, 300])
                self.assertLess(abs(w / original_w - h / original_h), 0.01)
        self.assertEqual([Path(p).read_bytes() for p in paths], original)

    def test_one_frame_and_zero_savings(self):
        paths = self.frames(1)
        for setting in (1, 2):
            out = self.root / f'one{setting}'
            core.prepare_images_in_temp_folder([], paths, str(out), {}, setting)
            budget = json.loads((out / 'compression_budget.json').read_text())
            self.assertEqual(budget['saved_pixels'], 0)
            self.assertLessEqual(budget['actual_screenshot_pixels'], 180000)
        duplicate = self.root / 'duplicate.png'
        duplicate.write_bytes(Path(paths[0]).read_bytes())
        out = self.root / 'zero'
        result = core.prepare_images_in_temp_folder([], [paths[0], str(duplicate)], str(out), {}, 1)
        with Image.open(result[1]) as image:
            self.assertEqual(image.size, (512, 256))
        self.assertEqual(duplicate.read_bytes(), Path(paths[0]).read_bytes())

    def test_tile_snap_sample_sizes_and_disabled_mode(self):
        for size, margin, expected in (((900, 443), 100, (900, 443)),
                                       ((1314, 646), 100, (1314, 646)),
                                       ((515, 1119), 100, (471, 1024)),
                                       ((1314, 646), 150, (1024, 503)),
                                       ((515, 1119), 0, (515, 1119)),
                                       ((512, 1024), 100, (512, 1024))):
            path = self.root / 'snap.png'
            Image.new('RGB', size).save(path)
            info = snap_to_tile_boundary(path, margin)
            with Image.open(path) as image:
                self.assertEqual(image.size, expected)
            if expected != size:
                self.assertLess(info['estimated_tiles_after'], info['estimated_tiles_before'])
            self.assertEqual(info['extra_saved_pixels'], size[0] * size[1] - expected[0] * expected[1])
        self.assertEqual(tile_count(515, 1119), 6)
        self.assertEqual(tile_count(471, 1024), 2)

    def test_tiny_aspect_ratio_and_image_caps(self):
        path = self.root / 'thin.png'
        Image.new('RGB', (1, 1000)).save(path)
        width, height = resize_to_area(path, 10)
        self.assertLessEqual(width * height, 10)
        paths = self.frames()
        asset = self.root / 'asset.png'
        Image.new('RGB', (1000, 1000)).save(asset)
        result = core.prepare_images_in_temp_folder([str(asset)], paths, str(self.root / 'cap'),
                                                   {'max_single_image_size': 40}, 1)
        with Image.open(result[0]) as image:
            self.assertEqual(image.size, (40, 40))
        with Image.open(result[1]) as image:
            self.assertEqual(image.size, (600, 300))

    def test_generation_backs_up_budget(self):
        source = self.frames(2)
        checkpoint_dir = self.root / 'data' / '1' / 'checkpoints'
        checkpoint_dir.mkdir(parents=True)
        for i, path in enumerate(source):
            (checkpoint_dir / f'step_{i}.png').write_bytes(Path(path).read_bytes())
        for setting in (1, 2):
            process_task({'id': 1}, str(self.root / 'data'), str(self.root / 'outputs'), 'mock', setting=setting)
            out = self.root / 'outputs' / '1' / f'mock_setting{setting}'
            self.assertTrue((out / 'compression_budget.json').exists())
            self.assertEqual(len(list((out / 'backups').glob('*/compression_budget.json'))), 1)


if __name__ == '__main__':
    unittest.main()
