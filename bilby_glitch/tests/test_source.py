import numpy as np
import pytest
from bilby_glitch.source import antiglitch


@pytest.fixture
def base_inputs():
    """Provides a standard set of physical inputs.
    """
    freqs = np.linspace(10, 1000, 100)
    return {
        "frequency_array": freqs,
        "psd": np.ones_like(freqs),
        "tstart": 0.0,
        "A": 2.5,
        "f": 100.0,
        "phi": np.pi / 4,
        "tc": 10.0
    }


def test_gamma_explicitly_provided(base_inputs):
    """Test that the function computes correctly when 'gamma' is passed."""
    result = antiglitch(**base_inputs, gamma=5.0)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(base_inputs["frequency_array"])
    assert np.iscomplexobj(result)


def test_log_gamma_kwargs_provided(base_inputs):
    """Test that 'log_gamma' calculates gamma correctly via kwargs."""
    # log10(5) should evaluate to gamma=5.0
    result_log = antiglitch(**base_inputs, log_gamma=np.log10(5.0))
    result_gamma = antiglitch(**base_inputs, gamma=5.0)
    np.testing.assert_allclose(result_log, result_gamma, rtol=1e-7)


def test_log_gamma_overrides_gamma(base_inputs):
    """Test the priority logic: log_gamma overrides an explicit gamma if both
    are passed.
    """
    # log_gamma=1.0 translates to gamma=10.0
    result_both = antiglitch(**base_inputs, gamma=999.0, log_gamma=1.0)
    result_expected = antiglitch(**base_inputs, gamma=10.0) 
    np.testing.assert_allclose(result_both, result_expected, rtol=1e-7)


def test_missing_gamma_raises_error(base_inputs):
    """Ensure the correct ValueError is raised if both gamma parameters are
    missing.
    """
    error_msg = "Either gamma or log_gamma must be provided"
    with pytest.raises(ValueError, match=error_msg):
        antiglitch(**base_inputs)


def test_amplitude_scaling(base_inputs):
    """Check that scaling the amplitude 'A' scales the entire output linearly."""
    result_base = antiglitch(**base_inputs, gamma=2.0)
    inputs_doubled = base_inputs.copy()
    inputs_doubled["A"] = base_inputs["A"] * 2.0
    result_doubled = antiglitch(**inputs_doubled, gamma=2.0)
    np.testing.assert_allclose(result_doubled, result_base * 2.0, rtol=1e-7)


def test_zero_time_shift(base_inputs):
    """Test phase evolution when dt (tc - tstart) is zero."""
    inputs_no_dt = base_inputs.copy()
    inputs_no_dt["tstart"] = 5.0
    inputs_no_dt["tc"] = 5.0
    result = antiglitch(**inputs_no_dt, gamma=2.0)
    expected_phase = 2 * inputs_no_dt["phi"]
    phases = np.angle(result)
    expected_phase_wrapped = (expected_phase + np.pi) % (2 * np.pi) - np.pi
    np.testing.assert_allclose(phases, expected_phase_wrapped, atol=1e-7)
