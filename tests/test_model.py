import datetime as dt

import torch

from forecasting_earthquakes_hopefully import normal
from forecasting_earthquakes_hopefully.model import Normal


def test_shape_and_that_a_hidden_station_does_not_matter():
    torch.manual_seed(0)
    model = Normal(stations=4, bands=5, width=32).eval()
    x = torch.randn(2, 4, 6, 86400)
    present = torch.tensor([[False, True, True, False], [True, False, True, True]])
    target, weekday = torch.tensor([0, 1]), torch.tensor([3, 6])
    with torch.no_grad():
        a = model(x, present, target, weekday)
        x[0, 0], x[0, 3] = 99.0, -99.0                     # the target and an absent station of the first sample
        b = model(x, present, target, weekday)
    assert a.shape == (2, 1440, 10)
    assert torch.allclose(a, b, atol=1e-5)


def test_roles_around_an_earthquake():
    e = [dt.date(2020, 1, 24)]
    role = lambda d: normal.role_of(d, e, "elazig", 0, 5)
    assert role(dt.date(2020, 1, 24)) is None and role(dt.date(2020, 2, 23)) is None
    assert role(dt.date(2020, 1, 23)) == "before" and role(dt.date(2019, 12, 27)) == "before"
    assert role(dt.date(2019, 12, 26)) in ("train", "val") and role(dt.date(2020, 2, 24)) in ("train", "val")
