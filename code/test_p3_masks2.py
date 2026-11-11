"""Run: python -m unittest -v test_p3_masks2.py"""
import unittest
import torch
from p3_masks2 import build_masks, masks_from_batch, LengthCollator


class MasksTest(unittest.TestCase):
    def test_tail_and_invalid_audio_invariance(self):
        w = torch.ones(2, 8 * 320)
        args = dict(src_lengths=[2 * 320, 7 * 320], mix_length=6 * 320, n_frames=8)
        a = build_masks(w, **args)
        w[0, 2 * 320:] = 1e6
        w[1, 6 * 320:] = 1e6
        b = build_masks(w, **args)
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)
        rec = a['source_active'][1] & (torch.arange(8) >= 2)
        self.assertEqual(int(rec.sum()), 4)
        self.assertTrue(a['union_active'][rec].all())
        self.assertFalse(a['overlap_active'][rec].any())

    def test_partial_and_empty(self):
        m = build_masks(torch.ones(2, 1000), [319, 641], 639, n_frames=4)
        self.assertEqual(int(m['union_active'].sum()), 1)
        self.assertEqual(int(m['overlap_active'].sum()), 0)
        m = build_masks(torch.ones(2, 1000), [0, 0], 0, n_frames=4)
        self.assertFalse(m['union_active'].any())

    def test_bad_lengths(self):
        for lens in ([-1, 320], [641, 320], [1.5, 320], [320]):
            with self.assertRaises(ValueError):
                build_masks(torch.ones(2, 640), lens, 640, n_frames=2)
        with self.assertRaises(ValueError):
            build_masks(torch.ones(2, 640), [320, 320], -1, n_frames=2)

    def test_collator_to_masks(self):
        items = [dict(clean_lengths=[320, 960], mix_length=960),
                 dict(clean_lengths=[640, 320], mix_length=640)]
        def base(rows):
            # Deliberate mutation proves lengths were cloned before base call.
            rows[0]['clean_lengths'][0] = 0
            return {'clean_audios': torch.ones(2, 2, 1280)}
        batch = LengthCollator(base)(items)
        self.assertEqual(int(batch['clean_lengths'][0, 0]), 320)
        masks = masks_from_batch(batch, n_frames=4)
        self.assertEqual(masks['union_active'].sum(1).tolist(), [3, 2])
        self.assertEqual(masks['overlap_active'].sum(1).tolist(), [1, 1])

    def test_missing_metadata_fails_before_base(self):
        def base(_):
            self.fail('base must not run with incomplete metadata')
        with self.assertRaises(ValueError):
            LengthCollator(base)([dict(clean_lengths=[320, 320], mix_length=320), {}])

    def test_explicit_feature_axis(self):
        with self.assertRaises(TypeError):
            masks_from_batch({})

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda(self):
        w = torch.ones(2, 960, device='cuda')
        m = build_masks(w, [320, 960], 960, n_frames=3)
        self.assertEqual(m['union_active'].device.type, 'cuda')
        self.assertEqual(int(m['overlap_active'].sum()), 1)


if __name__ == '__main__':
    unittest.main()
