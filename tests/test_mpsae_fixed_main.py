"""Image-specified MPSAEv2: fixed main/reconstruction K, linear auxiliary OT K."""
import contextlib
import copy
import io
import math
import unittest
from unittest import mock

import torch

import mmrsae_imagenet as runner
import mpsae_repair as core
from mmrsae import MMRSAE, MMRSAEConfig, linear_ot_annealed_k
from repair_runtime import named_parameters, pack_adam, restore_adam


class FixedMainMPSAETests(unittest.TestCase):
    def fixture(self, variant="mpsaev2", **overrides):
        settings = dict(input_dim=3, hidden_dim=64, k=2, anneal_start_k=8,
                        variant=variant, microbatch=3, k_aux=4, dead_steps=2)
        settings.update(overrides)
        model = MMRSAE(MMRSAEConfig(**settings))
        with torch.no_grad():
            model.sae.decoder.abs_()
            model.sae.inactive_steps.fill_(50)
        return model, torch.tensor([[.9, .4, .2], [.3, .5, 1.], [.6, .9, .1]])

    def test_linear_schedule_matches_the_image_at_each_step_and_boundary(self):
        for total, start, target, fraction in ((10, 8, 2, .7), (500, 64, 8, .7),
                (333, 63, 7, .37), (10, 8, 2, 1.), (1, 8, 2, .7), (5, 2, 2, .7)):
            duration = max(1, math.ceil(total * fraction))
            for step in range(total + 2):
                expected = round(start - (start - target) * step / duration) if step < duration else target
                self.assertEqual(linear_ot_annealed_k(step, total, start, target, fraction), expected)
            self.assertEqual(linear_ot_annealed_k(0, total, start, target, fraction), start)
            self.assertEqual(linear_ot_annealed_k(duration, total, start, target, fraction), target)
        for arguments in ((0, 0, 8, 2, .7), (-1, 10, 8, 2, .7),
                          (0, 10, 1, 2, .7), (0, 10, 8, 2, 0)):
            with self.assertRaises(ValueError):
                linear_ot_annealed_k(*arguments)

    def test_actual_views_and_full_objective_match_independent_fixed_main_definition(self):
        model, x = self.fixture()
        self.assertFalse(model.needs_dictionary or model.needs_bank or model.needs_context)
        self.assertFalse(model.config.mask)
        for step in (0, 3, 7, 9):
            with self.subTest(step=step), \
                    mock.patch.object(model.sae, "preactivations", wraps=model.sae.preactivations) as encoder, \
                    mock.patch.object(core, "relational_kl", side_effect=AssertionError("KL used")):
                loss, stats, views = model.loss(x, step=step, total_steps=10, update_activity=False)
            self.assertEqual(encoder.call_count, 1)
            ka = round(8 - 6 * step / 7) if step < 7 else 2
            self.assertEqual(stats["training_k"], 2)
            self.assertEqual(stats["annealing_k"], ka)
            self.assertEqual(stats["auxiliary_budget"], 8)
            self.assertEqual((stats["ot_view_k"], stats["ot_view_2k"], stats["ot_view_4k"]), (2, 2*ka, 4*ka))
            expected = [model.encode(x, k) for k in (2, 2*ka, 4*ka)]
            for name, code in zip(("final_k", "ot_2k", "ot_4k"), expected):
                torch.testing.assert_close(views[name], code, rtol=0., atol=0.)
            torch.testing.assert_close(views["raw_auxiliary"], model.encode(x, 8))
            reconstruction, _, _ = core.reconstruction_terms(model.sae, x, 2, 8,
                update_activity=False, k_aux=4)
            gap, _, _ = core.mixed_matching_gap(*expected, microbatch=3)
            torch.testing.assert_close(loss, reconstruction + 2 * gap, rtol=0., atol=0.)
            self.assertGreater(float(stats["loss_dead"]), 0.)

    def test_reconstruction_dead_loss_and_activity_stay_fixed_across_annealing(self):
        v2, x = self.fixture()
        fixed, _ = self.fixture("mpsae")
        reference = fixed.loss(x, update_activity=True)
        for step in (0, 3, 7, 9):
            current, _ = self.fixture()
            _, stats, views = current.loss(x, step=step, total_steps=10)
            for field in ("mse_final_K", "mse_raw_aux", "loss_dead", "dead_fraction"):
                torch.testing.assert_close(stats[field], reference[1][field], rtol=0., atol=0.)
            for view in ("raw_k", "raw_2k", "raw_4k", "raw_auxiliary", "final_k"):
                torch.testing.assert_close(views[view], reference[2][view], rtol=0., atol=0.)
            torch.testing.assert_close(current.sae.inactive_steps, fixed.sae.inactive_steps, rtol=0., atol=0.)

    def test_endpoint_loss_and_parameter_gradients_equal_fixed_mpsae(self):
        fixed, x = self.fixture("mpsae")
        v2, _ = self.fixture()
        a, _, _ = fixed.loss(x, update_activity=False)
        b, _, _ = v2.loss(x, step=7, total_steps=10, update_activity=False)
        torch.testing.assert_close(a, b, rtol=0., atol=0.)
        a.backward()
        b.backward()
        for left, right in zip(fixed.parameters(), v2.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0., atol=0.)

    def test_diagnostics_follow_annealed_ot_views_and_width_caps(self):
        model, x = self.fixture(hidden_dim=16, anneal_start_k=16)
        before = model.sae.inactive_steps.clone()
        loss, stats, views = model.loss(x, step=0, total_steps=10,
                                      diagnostics=True, update_activity=False)
        self.assertEqual(stats["ot_view_2k"], 16)
        self.assertEqual(stats["ot_view_4k"], 16)
        self.assertEqual(stats["auxiliary_budget"], 8)
        self.assertIn("gradient_old_ot_ot_2k_norm", stats)
        self.assertGreater(float(stats["gradient_old_ot_ot_2k_norm"]), 0.)
        torch.testing.assert_close(model.sae.inactive_steps, before)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_adam_resume_matches_uninterrupted_training_through_linear_boundary(self):
        model, x = self.fixture()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        def train(current, adam, first, stop):
            trace = []
            for step in range(first, stop):
                adam.zero_grad(set_to_none=True)
                loss, stats, _ = current.loss(x, step=step, total_steps=10)
                loss.backward()
                adam.step()
                trace.append((stats["training_k"], stats["annealing_k"], loss.detach().clone()))
            return trace

        train(model, optimizer, 0, 4)
        state = copy.deepcopy(model.state_dict())
        moments = copy.deepcopy(pack_adam(optimizer, named_parameters(model.sae, model.repair)))
        restored, _ = self.fixture()
        restored.load_state_dict(state)
        resumed = torch.optim.Adam(restored.parameters(), lr=1e-3)
        restore_adam(resumed, named_parameters(restored.sae, restored.repair), moments)
        a, b = train(model, optimizer, 4, 10), train(restored, resumed, 4, 10)
        for left, right in zip(a, b):
            self.assertEqual(left[:2], right[:2])
            torch.testing.assert_close(left[2], right[2], rtol=0., atol=0.)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0., atol=0.)

    def test_preflight_mpsaev2_has_no_dictionary_or_context_memory(self):
        with mock.patch("mmrsae_gpu.preflight", return_value={"gpu_search_probe": "passed"}) as preflight, \
                mock.patch("mmrsae_data.prepare_imagenet_features", side_effect=AssertionError("data opened")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(["--preflight-only", "--variant", "mpsaev2"]), 0)
        self.assertFalse(preflight.call_args.args[1]["model"]["landmark_dictionary"])
        self.assertEqual(preflight.call_args.args[1]["model"]["context_reference_count"], 0)


if __name__ == "__main__":
    unittest.main()
