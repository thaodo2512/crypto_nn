from app.runtime.throttle import Throttler


def test_throttle_with_tau_rule():
    thr = Throttler(thresh_temp=70, thresh_gpu=80)
    # Weak signal below tau+0.02 should be throttled when hot
    assert thr.should_throttle(75.0, 50.0) is True
    assert thr.should_throttle(60.0, 85.0) is True
    assert thr.should_throttle(60.0, 50.0) is False

