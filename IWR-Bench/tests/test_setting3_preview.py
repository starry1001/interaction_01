import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from setting3 import analyze_pair, prepare_delta_screenshots, _trim_blank_sides


class Setting3PreviewTests(unittest.TestCase):
    def test_thin_progress_bar_and_small_text_edit_survive(self):
        before = np.full((300, 500, 3), 240, np.uint8)
        for x1, y1, x2, y2 in ((200, 200, 260, 202), (210, 80, 213, 89)):
            after = before.copy()
            after[y1:y2, x1:x2] = 0
            result = analyze_pair(before, after)
            a, b, c, d = result['boxes'][0]
            self.assertLessEqual(a, x1)
            self.assertLessEqual(b, y1)
            self.assertGreaterEqual(c, x2)
            self.assertGreaterEqual(d, y2)
            self.assertLess((c-a)*(d-b), 300*500)

    def test_scrollbar_is_ignored_without_verified_content_scroll(self):
        for background, thumb in ((250, 130), (30, 140)):
            with self.subTest(background=background):
                a = np.full((800, 1000, 3), background, np.uint8)
                b = a.copy()
                a[100:220, -12:-3] = thumb
                b[300:420, -12:-3] = thumb
                b[350:390, 180:240] = (20, 80, 220)
                with patch('setting3.detect_scroll', return_value=None):
                    result = analyze_pair(a, b)
                self.assertGreaterEqual(result['metrics']['ignored_scrollbar_width'], 12)
                self.assertLess(result['boxes'][0][2], 400)
                self.assertTrue(self.covers(result['boxes'], 200, 370))

    def test_colored_right_edge_control_is_preserved(self):
        a = np.full((800, 1000, 3), 250, np.uint8)
        b = a.copy()
        a[100:220, -12:-3] = (20, 80, 220)
        b[300:420, -12:-3] = (20, 80, 220)
        result = analyze_pair(a, b)
        self.assertEqual(result['metrics']['ignored_scrollbar_width'], 0)
        self.assertTrue(self.covers(result['boxes'], 990, 350))

    def test_side_backgrounds_trim_without_erasing_deleted_content(self):
        a = np.full((600, 1000, 3), 250, np.uint8)
        b = a.copy()
        a[200:300, 200:600] = 20
        b[200:300, 400:600] = 20
        box = _trim_blank_sides([0, 0, 1000, 600], a, b, 0, 24)
        self.assertEqual(box, [176, 0, 624, 600])

    def page(self):
        page = np.full((1800, 1000, 3), 250, np.uint8)
        rng = np.random.default_rng(42)
        for y in range(25, 1750, 55):
            text = ''.join(rng.choice(list('ABCDEFGH123456789'), 30))
            cv2.putText(page, text, (130, y), cv2.FONT_HERSHEY_SIMPLEX, .65, (30, 30, 30), 2)
        return page

    def covers(self, boxes, x, y):
        return any(a <= x < c and b <= y < d for a, b, c, d in boxes)

    def test_down_and_up_scroll_with_stationary_sidebar(self):
        page = self.page()
        a, b = page[:800].copy(), page[350:1150].copy()
        sidebar = np.full((600, 200, 3), 180, np.uint8)
        a[100:700, 760:960] = sidebar
        b[100:700, 760:960] = sidebar
        for before, after, direction, dy, novel_y in ((a, b, 'down', -350, 750), (b, a, 'up', 350, 25)):
            result = analyze_pair(before, after)
            self.assertIsNotNone(result['scroll'])
            self.assertEqual(result['scroll']['direction'], direction)
            self.assertAlmostEqual(result['scroll']['dy'], dy, delta=2)
            self.assertFalse(self.covers(result['boxes'], 850, 400))
            self.assertTrue(self.covers(result['boxes'], 200, novel_y))
            self.assertLess(sum((c-a)*(d-b) for a,b,c,d in result['boxes']), 800*1000*.8)

    def test_changed_sidebar_is_preserved_during_scroll(self):
        page = self.page()
        a, b = page[:800].copy(), page[350:1150].copy()
        a[100:700, 760:960] = 180
        b[100:700, 760:960] = 180
        b[300:345, 810:910] = (0, 0, 200)
        result = analyze_pair(a, b)
        self.assertIsNotNone(result['scroll'])
        self.assertTrue(self.covers(result['boxes'], 850, 320))

    def test_small_new_edit_in_the_scroll_overlap_is_preserved(self):
        page = self.page()
        a, b = page[:800].copy(), page[350:1150].copy()
        b[170:182, 650:654] = 0
        result = analyze_pair(a, b)
        self.assertIsNotNone(result['scroll'])
        self.assertTrue(self.covers(result['boxes'], 652, 175))

    def test_changed_digit_keeps_the_complete_text_line(self):
        a = np.full((500, 1000, 3), 250, np.uint8)
        b = a.copy()
        for frame, text in ((a, 'Total price 120'), (b, 'Total price 121')):
            cv2.putText(frame, text, (160, 230), cv2.FONT_HERSHEY_SIMPLEX, .9, (20, 20, 20), 2)
        result = analyze_pair(a, b)
        self.assertTrue(self.covers(result['boxes'], 165, 220))

    def test_distant_edits_share_one_enclosing_box(self):
        a = np.full((1000, 1400, 3), 250, np.uint8)
        b = a.copy()
        b[90:140, 150:250] = 0
        b[800:850, 1150:1250] = 0
        result = analyze_pair(a, b)
        self.assertIsNone(result['scroll'])
        self.assertEqual(len(result['boxes']), 1)
        self.assertTrue(self.covers(result['boxes'], 200, 110))
        self.assertTrue(self.covers(result['boxes'], 1200, 820))
        self.assertTrue(self.covers(result['boxes'], 700, 500))

    def test_new_page_and_dimensions_are_not_invented_scrolls(self):
        a = self.page()[:800].copy()
        b = np.random.default_rng(99).integers(0, 256, a.shape, dtype=np.uint8)
        self.assertIsNone(analyze_pair(a, b)['scroll'])
        self.assertEqual(analyze_pair(a, b[:600])['boxes'], [[0, 0, 1000, 600]])

    def test_single_crop_has_exact_source_pixels_and_first_frame_is_untouched(self):
        a = np.full((1000, 1400, 3), 250, np.uint8)
        b = a.copy()
        b[90:140, 150:250] = 0
        b[800:850, 1150:1250] = 0
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = []
            for i, frame in enumerate((a, b)):
                path = root / f'step_{i}.png'
                Image.fromarray(frame).save(path)
                paths.append(str(path))
            original_bytes = [Path(p).read_bytes() for p in paths]
            output = root / 'out'
            prepared = prepare_delta_screenshots(paths, output)
            self.assertEqual(Path(prepared[0]).read_bytes(), original_bytes[0])
            self.assertEqual([Path(p).read_bytes() for p in paths], original_bytes)
            manifest = json.loads((output / 'screenshot_manifest.json').read_text())
            self.assertFalse(manifest['preview_only'])
            record = manifest['screenshots'][1]
            self.assertEqual(record['mode'], 'crop')
            self.assertEqual(len(record['boxes']), 1)
            with Image.open(prepared[1]) as image:
                x1, y1, x2, y2 = record['box']
                self.assertEqual(image.size, (x2 - x1, y2 - y1))
                np.testing.assert_array_equal(np.array(image), b[y1:y2, x1:x2])


if __name__ == '__main__':
    unittest.main()
