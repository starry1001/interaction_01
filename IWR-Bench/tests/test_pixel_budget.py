import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from generate import process_task
from utils import core
from utils.pixel_budget import (estimate_image_tokens, patch_count, resize_to_token_budget,
                                token_budget_size, validate_original_size)


class TokenBudgetTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def frames(self, count=5, size=(600, 300)):
        paths = []
        for i in range(count):
            path = self.root / f'step_{i}.png'
            Image.new('RGB', size, (30 + i*25, 60, 100)).save(path)
            paths.append(str(path))
        return paths

    def test_patch_formula_and_proportional_token_target(self):
        self.assertEqual(patch_count(1920, 1080), 2040)
        self.assertEqual(estimate_image_tokens(1920, 1080), 2448)
        for width, height in ((1920, 1080), (1080, 1920), (641, 333), (1, 10000), (10000, 1), (3840, 13152)):
            size, target, effective = token_budget_size(width, height)
            self.assertEqual(target, estimate_image_tokens(width, height)*3//5)
            self.assertLessEqual(estimate_image_tokens(*size), effective)
            self.assertLessEqual(abs(size[0]/width-size[1]/height), max(1/width, 1/height))
            self.assertLessEqual(size[0], width)
            self.assertLessEqual(size[1], height)
            validate_original_size(*size)

    def test_small_image_budget_is_largest_feasible_proportional_size(self):
        width, height = 321, 219
        size, target, _ = token_budget_size(width, height)
        candidates = []
        for numerator, denominator in ([(x, width) for x in range(1, width+1)]
                                       + [(y, height) for y in range(1, height+1)]):
            w, h = max(1, width*numerator//denominator), max(1, height*numerator//denominator)
            if estimate_image_tokens(w, h) <= target:
                candidates.append(w*h)
        self.assertEqual(size[0]*size[1], max(candidates))

    def test_minimum_patch_and_original_limits(self):
        path = self.root/'tiny.png'
        Image.new('RGB', (1, 1)).save(path)
        r = resize_to_token_budget(path)
        self.assertEqual(r['target_estimated_tokens'], 1)
        self.assertEqual(r['output_estimated_tokens'], 2)
        self.assertTrue(r['minimum_patch_limited'])
        with self.assertRaises(ValueError):
            validate_original_size(65536, 1)
        with self.assertRaises(ValueError):
            validate_original_size(6400, 6400)
        size, _, _ = token_budget_size(16000, 16000)
        validate_original_size(*size)

    def test_setting1_scales_every_frame_including_first_without_setting3_reference(self):
        paths = self.frames()
        original = [Path(p).read_bytes() for p in paths]
        out = self.root/'setting1'
        with patch.object(core, 'prepare_delta_screenshots', side_effect=AssertionError('unexpected Setting 3 reference')):
            result = core.prepare_images_in_temp_folder([], paths, str(out), setting=1)
        budget = json.loads((out/'compression_budget.json').read_text())
        self.assertEqual(len(result), 5)
        self.assertEqual(budget['profile']['detail'], 'original')
        self.assertTrue(budget['profile']['assumption_not_officially_verified'])
        for index, record in enumerate(budget['allocations']):
            self.assertEqual(record['index'], index)
            self.assertLessEqual(record['retained_token_fraction'], .6)
            self.assertNotEqual(record['output_size'], [600, 300])
        self.assertNotEqual(Path(result[0]).read_bytes(), original[0])
        self.assertEqual([Path(p).read_bytes() for p in paths], original)

    def test_setting2_stacks_landscape_vertically_and_portrait_horizontally(self):
        for size, layout, expected in (((600, 300), 'vertical', [[600, 996], [600, 664]]),
                                        ((300, 600), 'horizontal', [[900, 632], [600, 632]])):
            paths = self.frames(size=size)
            originals = [Path(p).read_bytes() for p in paths]
            out = self.root/layout
            outputs = core.prepare_images_in_temp_folder([], paths, str(out), setting=2)
            budget = json.loads((out/'compression_budget.json').read_text())
            self.assertEqual(len(outputs), 2)
            records = budget['allocations']
            self.assertEqual([r['screenshot_count'] for r in records], [3, 2])
            self.assertEqual([r['source_size'] for r in records], expected)
            self.assertEqual([i for r in records for i in r['source_indices']], list(range(5)))
            for record, output in zip(records, outputs):
                self.assertEqual(record['layout'], layout)
                reference = estimate_image_tokens(*size) * record['screenshot_count']
                self.assertEqual(record['reference_estimated_tokens'], reference)
                self.assertEqual(record['target_estimated_tokens'], reference*3//5)
                self.assertLessEqual(record['output_estimated_tokens'], reference*3//5)
                self.assertLessEqual(record['retained_token_fraction'], .6)
                with Image.open(output) as image:
                    for region in record['source_regions']:
                        x1, y1, x2, y2 = region['output_sheet_box']
                        self.assertEqual(image.getpixel(((x1+x2)//2, (y1+y2)//2)),
                                         (30+region['index']*25, 60, 100))
            self.assertEqual([Path(p).read_bytes() for p in paths], originals)

    def test_setting2_added_labels_and_padding_do_not_increase_budget(self):
        paths = self.frames(3, size=(32, 64))
        out = self.root/'large_padding'
        core.prepare_images_in_temp_folder([], paths, str(out), setting=2)
        budget = json.loads((out/'compression_budget.json').read_text())
        record = budget['allocations'][0]
        reference = sum(estimate_image_tokens(32, 64) for _ in paths)
        self.assertGreater(record['source_pixels'], record['original_screenshot_pixels'])
        self.assertGreater(record['source_estimated_tokens'], reference)
        self.assertEqual(record['reference_estimated_tokens'], reference)
        self.assertEqual(record['target_estimated_tokens'], reference*3//5)
        self.assertLessEqual(record['output_estimated_tokens'], reference*3//5)
        self.assertEqual(budget['reference_estimated_tokens'], reference)
        self.assertEqual(budget['retained_token_fraction'], record['output_estimated_tokens']/reference)

    def test_groups_and_mixed_orientation_keep_all_frames_in_order(self):
        self.assertEqual(core.screenshot_group_sizes(5), [3, 2])
        self.assertEqual(core.screenshot_group_sizes(7), [4, 3])
        paths = self.frames(7)
        Image.new('RGB', (300, 600)).save(paths[1])
        out = self.root/'mixed'
        core.prepare_images_in_temp_folder([], paths, str(out), setting=2)
        records = json.loads((out/'compression_budget.json').read_text())['allocations']
        self.assertEqual([i for r in records for i in r['source_indices']], list(range(7)))
        self.assertEqual(records[0]['layout'], 'vertical')

    def test_payload_sets_original_for_luna_without_adding_text_or_changing_assets(self):
        paths = self.frames(2)
        asset = self.root/'asset.png'
        Image.new('RGB', (100, 80)).save(asset)
        for model in ('gpt-6-luna', 'gpt-5.6-luna', 'gpt-5'):
            for setting in (0, 1, 2, 3):
                with patch.object(core, 'call_mllm', return_value='<html></html>') as call:
                    core.call_mllm_html(model, [str(asset)], paths,
                                        str(self.root/f'{model}_{setting}'), '600x300', setting=setting)
                payload = call.call_args.args[1]
                self.assertEqual(sum(p['type']=='text' for p in payload), 1)
                self.assertEqual(base64.b64decode(payload[1]['image_url']['url'].split(',')[1]), asset.read_bytes())
                for part in payload[1:]:
                    self.assertEqual(part['type'], 'image_url')
                    if model == 'gpt-5':
                        self.assertNotIn('detail', part['image_url'])
                    else:
                        self.assertEqual(part['image_url']['detail'], 'original')

    def test_direct_setting_preserves_source_size_and_assets_untouched(self):
        paths = self.frames(1)
        asset = self.root/'asset.png'
        Image.new('RGB', (1200, 1200)).save(asset)
        for setting in (0, 1, 2, 3):
            result = core.prepare_images_in_temp_folder([str(asset)], paths,
                                                       str(self.root/f'assets{setting}'), setting=setting)
            self.assertEqual(Path(result[0]).read_bytes(), asset.read_bytes())
            if setting == 0:
                self.assertEqual(Path(result[1]).read_bytes(), Path(paths[0]).read_bytes())

    def test_generation_saves_token_budget_without_backup(self):
        source = self.frames(2)
        checkpoints = self.root/'data/1/checkpoints'
        checkpoints.mkdir(parents=True)
        for i, path in enumerate(source):
            (checkpoints/f'step_{i}.png').write_bytes(Path(path).read_bytes())
        for setting in (1, 2):
            process_task({'id': 1}, str(self.root/'data'), str(self.root/'out'), 'mock', setting=setting)
            out = self.root/f'out/1/mock_setting{setting}'
            budget = json.loads((out/'compression_budget.json').read_text())
            self.assertEqual(budget['unit'], 'estimated image input tokens')
            self.assertTrue((out/'index.html').exists())
            self.assertTrue((out/'raw_response.txt').exists())
            self.assertFalse((out/'backups').exists())


if __name__ == '__main__':
    unittest.main()
