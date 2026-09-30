"""Check decision alignment, noncrossing quantiles, and causality."""
import numpy as np
import torch
from models.od.pdr_hurdle import PDRHurdleQuantile, hurdle_median, positive_pinball


def test_hurdle_median_queries_probability_dependent_quantile():
    levels = torch.tensor([0., .5, 1.])
    knots = torch.tensor([[1., 3., 5.]]).expand(3, -1)
    q = torch.tensor([.4, .75, 1.])
    torch.testing.assert_close(hurdle_median(q, knots, levels), torch.tensor([0., 7 / 3, 3.]))
    empty = torch.zeros(3)
    assert positive_pinball(knots, empty, levels) == 0


def test_hurdle_shapes_noncrossing_and_causal_auxiliary():
    torch.set_num_threads(2)
    model = PDRHurdleQuantile(seq_len=4, input_dim=3, hid_dim=8, node_num=3,
                             layer_nums=2, dropout=0., feature=2, horizon=2, output_dim=3,
                             activity_prior=np.full((3, 3, 2), .7), data_min=[0., 0.],
                             data_span=[3., 3.], use_log1p=True, quantile_count=5).eval()
    x = {'od': torch.rand(2, 4, 3, 3, 2), 'prev_od': torch.rand(2, 4, 3, 3, 2),
         'prev_y_od': torch.rand(2, 2, 3, 3, 2)}
    point, previous, logits, knots, past_logits, past_knots = model(x, return_aux=True)
    assert point.shape == previous.shape == logits.shape == (2, 2, 3, 3, 2)
    assert knots.shape == past_knots.shape == (2, 2, 3, 3, 2, 5)
    assert (knots.diff(dim=-1) >= 0).all() and (knots >= 1).all()
    torch.testing.assert_close(point, model({**x, 'prev_y_od': x['prev_y_od'] + 100}))
    loss = positive_pinball(knots, torch.ones_like(logits) * 3, model.levels)
    loss.backward()
    assert model.positive[-1].weight.grad.abs().sum() > 0
