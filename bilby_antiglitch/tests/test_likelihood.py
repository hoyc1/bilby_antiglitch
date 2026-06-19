import numpy as np
import pytest
from types import SimpleNamespace
from bilby_antiglitch import likelihood as module
from bilby_antiglitch.likelihood import Glitch, GravitationalWaveTransientPlusGlitch


class DummyCalculatedSNRs(object):
    """Dummy class to simulate bilby's SNR object"""
    def __init__(
        self,
        d_inner_h=0j,
        optimal_snr_squared=0.,
        complex_matched_filter_snr=0j,
        d_inner_h_array=None,
        optimal_snr_squared_array=None
    ):
        self.d_inner_h = d_inner_h
        self.optimal_snr_squared = optimal_snr_squared
        self.complex_matched_filter_snr = complex_matched_filter_snr
        self.d_inner_h_array = d_inner_h_array
        self.optimal_snr_squared_array = optimal_snr_squared_array

    def __iadd__(self, other):
        self.d_inner_h += other.d_inner_h
        self.optimal_snr_squared += other.optimal_snr_squared

        if self.d_inner_h_array is None:
            self.d_inner_h_array = other.d_inner_h_array
        elif other.d_inner_h_array is not None:
            self.d_inner_h_array = self.d_inner_h_array + other.d_inner_h_array

        if self.optimal_snr_squared_array is None:
            self.optimal_snr_squared_array = other.optimal_snr_squared_array
        elif other.optimal_snr_squared_array is not None:
            self.optimal_snr_squared_array = (
                self.optimal_snr_squared_array +
                other.optimal_snr_squared_array
            )

        if self.optimal_snr_squared != 0:
            self.complex_matched_filter_snr = (
                self.d_inner_h / np.sqrt(self.optimal_snr_squared)
            )
        else:
            self.complex_matched_filter_snr = 0j
        return self


class FakeInterferometer(object):
    """Dummy class to simulate bilby's Interferometer class"""
    def __init__(self, name):
        self.name = name
        self.frequency_mask = np.array([False, True, True, False])
        self.power_spectral_density_array = np.array([1., 2., 2., 1.])
        self.frequency_domain_strain = np.array(
            [0. + 0.j, 1. + 1.j, 2. + 0.j, 0. + 0.j]
        )

    def inner_product(self, signal):
        # simplified inner product to make the tests easier to check against
        # ground truth
        return np.sum(signal[self.frequency_mask])

    def optimal_snr_squared(self, signal):
        # Simplified optimal snr calculation to make the tests easier to check
        # agsinst
        return np.sum(np.abs(signal[self.frequency_mask]) ** 2)


class FakeWaveformGenerator(object):
    """Dummy class to simulate bilby's WaveformGenerator container in tests."""
    def __init__(self):
        self.start_time = 100.
        self.frequency_array = np.array([10., 20., 30., 40.])
        self.duration = 2.
        self.last_parameters = None

    def frequency_domain_strain(self, parameters):
        self.last_parameters = dict(parameters)
        return {"plus": np.ones(4, dtype=complex)}


def glitch_model(
    A, f, phi, tc, gamma=None, frequency_array=None, psd=None,
    tstart=None, **kwargs
):
    """Simple glitch model used throughout the test suite."""
    return np.ones_like(frequency_array, dtype=complex) * A


def unit_glitch_model(
    A=None, f=None, phi=None, tc=None, gamma=None, log_gamma=None,
    frequency_array=None, psd=None, tstart=None, **kwargs
):
    """Glitch model that always returns a unit strain."""
    return np.ones_like(frequency_array, dtype=complex)


def _dummy_parent_init(self, *args, **kwargs):
    """Dummy parent init used to isolate Glitch.__init__ tests."""
    self._interferometers = [
        SimpleNamespace(name="H1"),
        SimpleNamespace(name="L1"),
    ]
    self.waveform_generator = SimpleNamespace(
        start_time=100.,
        frequency_array=np.array([10., 20., 30., 40.]),
        duration=2.,
    )
    self.parameters = {}


class BaseTest(object):
    """Helper class for the glitch likelihood tests."""
    @pytest.fixture(autouse=True)
    def setup_test(self, monkeypatch):
        self.ifos = [FakeInterferometer("H1"), FakeInterferometer("L1")]
        self.waveform_generator = FakeWaveformGenerator()

        monkeypatch.setattr(
            module, "_fallback_to_parameters",
            lambda self, parameters:
            self.parameters if parameters is None else parameters
        )

    def _patch_parent_init(self, monkeypatch, cls):
        """Patch the actual parent used in the MRO for init-only tests."""
        parent = cls.__mro__[1]

        monkeypatch.setattr(parent, "__init__", _dummy_parent_init)
        monkeypatch.setattr(
            parent,
            "interferometers",
            property(
                lambda self: getattr(self, "_interferometers", []),
                lambda self, value:
                object.__setattr__(self, "_interferometers", value),
            ),
        )

    def _base_glitch_likelihood(self):
        likelihood = Glitch.__new__(Glitch)
        likelihood._interferometers = self.ifos
        likelihood.waveform_generator = self.waveform_generator
        likelihood._glitch_model = glitch_model
        likelihood._glitch_params = ["A", "f", "phi", "tc", "gamma"]
        likelihood.detectors_glitch = [ifo.name for ifo in self.ifos]
        likelihood._CalculatedSNRs = DummyCalculatedSNRs
        likelihood.parameters = {}
        likelihood.calibration_draws = {
            "H1": np.array([[2., 2.]]),
            "L1": np.array([[2., 2.]]),
        }
        likelihood.calibration_marginalization = False
        return likelihood

    def _base_plus_glitch_likelihood(self):
        likelihood = GravitationalWaveTransientPlusGlitch.__new__(
            GravitationalWaveTransientPlusGlitch
        )
        likelihood._interferometers = self.ifos
        likelihood.waveform_generator = self.waveform_generator
        likelihood._glitch_model = glitch_model
        likelihood._glitch_params = ["A", "f", "phi", "tc", "gamma"]
        likelihood.detectors_glitch = [ifo.name for ifo in self.ifos]
        likelihood._CalculatedSNRs = DummyCalculatedSNRs
        likelihood.parameters = {}
        likelihood.calibration_draws = {
            "H1": np.array([[1., 1.], [2., 2.]]),
            "L1": np.array([[1., 1.], [2., 2.]]),
        }
        likelihood.calibration_abs_draws = {
            "H1": np.array([[1., 1.], [3., 3.]]),
            "L1": np.array([[1., 1.], [3., 3.]]),
        }
        likelihood.calibration_marginalization = False
        likelihood.get_sky_frame_parameters = (
            lambda params: {"zenith": 1.23, "azimuth": 4.56}
        )
        return likelihood

    def _base_standard_gw_likelihood(self):
        likelihood = module.GravitationalWaveTransient.__new__(
            module.GravitationalWaveTransient
        )
        likelihood._interferometers = self.ifos
        likelihood.waveform_generator = self.waveform_generator
        likelihood._CalculatedSNRs = DummyCalculatedSNRs
        likelihood.parameters = {}
        likelihood.calibration_draws = {
            "H1": np.array([[1., 1.], [2., 2.]]),
            "L1": np.array([[1., 1.], [2., 2.]]),
        }
        likelihood.calibration_abs_draws = {
            "H1": np.array([[1., 1.], [3., 3.]]),
            "L1": np.array([[1., 1.], [3., 3.]]),
        }
        likelihood.calibration_marginalization = False
        likelihood.time_marginalization = False
        likelihood.distance_marginalization = False
        likelihood.phase_marginalization = False
        likelihood.get_sky_frame_parameters = lambda params: {}
        return likelihood

    def _network_glitch_snrs(self, likelihood, signal, parameters=None):
        total = likelihood._CalculatedSNRs()
        _parameters = {} if parameters is None else parameters

        for interferometer in self.ifos:
            total += likelihood.calculate_snrs(
                glitch_strain=signal.copy(),
                interferometer=interferometer,
                parameters=_parameters
            )
        return total

    def _network_standard_snrs(self, likelihood, signal, parameters=None):
        total = likelihood._CalculatedSNRs()
        _parameters = {} if parameters is None else parameters

        for interferometer in self.ifos:
            total += likelihood.calculate_snrs(
                {"plus": signal.copy()},
                interferometer,
                parameters=_parameters
            )
        return total

    def _network_plus_glitch_snrs(
        self, likelihood, waveform_polarizations, glitch_signal, parameters=None
    ):
        total = likelihood._CalculatedSNRs()
        _parameters = {} if parameters is None else parameters

        for interferometer in self.ifos:
            total += likelihood.calculate_snrs(
                waveform_polarizations=waveform_polarizations,
                interferometer=interferometer,
                glitch_strain=glitch_signal.copy(),
                parameters=_parameters
            )
        return total


class TestGlitch(BaseTest):
    """Tests for the Glitch likelihood."""

    @pytest.mark.parametrize(
        "kwargs,msg",
        [
            (
                {"time_marginalization": True},
                "Unable to use time marginalization"
            ),
            (
                {"distance_marginalization": True},
                "Unable to use distance marginalization"
            ),
            (
                {"phase_marginalization": True},
                "Unable to use phase marginalization"
            ),
        ]
    )
    def test_init_raises_for_unsupported_marginalizations(
        self, monkeypatch, kwargs, msg
    ):
        self._patch_parent_init(monkeypatch, Glitch)

        with pytest.raises(ValueError, match=msg):
            Glitch(glitch_model=glitch_model, **kwargs)

    def test_init_uses_default_glitch_model(self, monkeypatch):
        self._patch_parent_init(monkeypatch, Glitch)
        monkeypatch.setattr(
            Glitch, "_default_glitch_model", property(lambda self: glitch_model)
        )

        likelihood = Glitch()

        assert likelihood._glitch_model is glitch_model
        assert likelihood._glitch_params == ["A", "f", "phi", "tc", "gamma"]
        assert likelihood.detectors_glitch == ["H1", "L1"]

    def test_init_respects_detectors_glitch(self, monkeypatch):
        self._patch_parent_init(monkeypatch, Glitch)

        likelihood = Glitch(
            glitch_model=glitch_model,
            detectors_glitch=["H1"]
        )

        assert likelihood.detectors_glitch == ["H1"]

    def test_get_ifo_glitch_params_prefers_detector_specific_values(self):
        likelihood = self._base_glitch_likelihood()
        params = {
            "A": 1.,
            "f": 20.,
            "phi": 0.1,
            "tc": 0.2,
            "gamma": 0.3,
            "log_gamma": -4.,
            "H1_A": 9.,
            "H1_log_gamma": -1.,
        }

        result = likelihood._get_ifo_glitch_params(self.ifos[0], params)
        assert result["A"] == 9.
        assert result["f"] == 20.
        assert result["phi"] == 0.1
        assert result["tc"] == 0.2
        assert result["gamma"] == 0.3
        assert result["log_gamma"] == -1.
        assert result["tstart"] == self.waveform_generator.start_time

        np.testing.assert_array_equal(
            result["frequency_array"],
            self.waveform_generator.frequency_array
        )
        np.testing.assert_array_equal(
            result["psd"],
            self.ifos[0].power_spectral_density_array
        )

        result = likelihood._get_ifo_glitch_params(self.ifos[1], params)
        assert result["A"] == 1.
        assert result["f"] == 20.
        assert result["phi"] == 0.1
        assert result["tc"] == 0.2
        assert result["gamma"] == 0.3
        assert result["log_gamma"] == -4.
        assert result["tstart"] == self.waveform_generator.start_time

    def test_get_ifo_glitch_params_falls_back_to_global_values(self):
        likelihood = self._base_glitch_likelihood()
        params = {
            "A": 1.,
            "f": 20.,
            "phi": 0.1,
            "tc": 0.2,
            "log_gamma": -4.,
        }

        result = likelihood._get_ifo_glitch_params(self.ifos[1], params)

        assert result["A"] == 1.
        assert result["f"] == 20.
        assert result["phi"] == 0.1
        assert result["tc"] == 0.2
        assert result["log_gamma"] == -4.

    def test_calculate_snrs(self):
        likelihood = self._base_glitch_likelihood()
        signal = np.array([0., 1. + 0.j, 2. + 0.j, 0.])

        snrs = likelihood.calculate_snrs(
            glitch_strain=signal.copy(),
            interferometer=self.ifos[0],
            parameters={}
        )

        assert snrs.d_inner_h == 3. + 0.j
        assert snrs.optimal_snr_squared == 5.
        assert snrs.complex_matched_filter_snr == pytest.approx(
            (3. + 0.j) / np.sqrt(5.)
        )
        assert snrs.d_inner_h_array is None
        assert snrs.optimal_snr_squared_array is None

    def test_compute_log_likelihood_from_snrs(self):
        likelihood = self._base_glitch_likelihood()
        snrs = DummyCalculatedSNRs(
            d_inner_h=10. + 1.j,
            optimal_snr_squared=4.
        )

        assert likelihood.compute_log_likelihood_from_snrs(snrs) == 8.

    def test_log_likelihood_ratio_calls_glitch_model_when_conditions_met(
        self, monkeypatch
    ):
        likelihood = self._base_glitch_likelihood()
        likelihood.detectors_glitch = ["H1"]
        calls = []

        def _glitch_model(**kwargs):
            calls.append(kwargs)
            return np.ones_like(kwargs["frequency_array"], dtype=complex)

        def _calculate_snrs(
            self, glitch_strain, interferometer, return_array=True,
            parameters=None
        ):
            return DummyCalculatedSNRs(
                d_inner_h=np.sum(glitch_strain),
                optimal_snr_squared=float(np.sum(np.abs(glitch_strain) ** 2))
            )

        likelihood._glitch_model = _glitch_model
        monkeypatch.setattr(Glitch, "calculate_snrs", _calculate_snrs)

        log_l = likelihood.log_likelihood_ratio(
            {
                "A": 1.,
                "f": 30.,
                "phi": 0.,
                "tc": 0.1,
                "gamma": 0.2,
            }
        )

        assert len(calls) == 1
        assert calls[0]["A"] == 1.
        assert log_l == 2.

    def test_log_likelihood_ratio_uses_zero_strain_for_incomplete_glitch(
        self, monkeypatch
    ):
        likelihood = self._base_glitch_likelihood()
        calls = []

        def _glitch_model(**kwargs):
            calls.append(kwargs)
            return np.ones_like(kwargs["frequency_array"], dtype=complex)

        def _calculate_snrs(
            self, glitch_strain, interferometer, return_array=True,
            parameters=None
        ):
            return DummyCalculatedSNRs(
                d_inner_h=np.sum(glitch_strain),
                optimal_snr_squared=float(np.sum(np.abs(glitch_strain) ** 2))
            )

        likelihood._glitch_model = _glitch_model
        monkeypatch.setattr(Glitch, "calculate_snrs", _calculate_snrs)

        log_l = likelihood.log_likelihood_ratio(
            {
                "A": 1.,
                "f": 30.,
                "phi": 0.,
                "gamma": 0.2,
            }
        )

        assert calls == []
        assert log_l == 0.

    def test_log_likelihood_ratio_uses_zero_strain_for_detector_not_included(
        self, monkeypatch
    ):
        likelihood = self._base_glitch_likelihood()
        likelihood.detectors_glitch = []
        calls = []

        def _glitch_model(**kwargs):
            calls.append(kwargs)
            return np.ones_like(kwargs["frequency_array"], dtype=complex)

        def _calculate_snrs(
            self, glitch_strain, interferometer, return_array=True,
            parameters=None
        ):
            return DummyCalculatedSNRs(
                d_inner_h=np.sum(glitch_strain),
                optimal_snr_squared=float(np.sum(np.abs(glitch_strain) ** 2))
            )

        likelihood._glitch_model = _glitch_model
        monkeypatch.setattr(Glitch, "calculate_snrs", _calculate_snrs)

        log_l = likelihood.log_likelihood_ratio(
            {
                "A": 1.,
                "f": 30.,
                "phi": 0.,
                "tc": 0.1,
                "gamma": 0.2,
            }
        )

        assert calls == []
        assert log_l == 0.

    def test_log_likelihood_ratio_accepts_log_gamma_without_gamma(
        self, monkeypatch
    ):
        likelihood = self._base_glitch_likelihood()
        calls = []

        def _glitch_model(**kwargs):
            calls.append(kwargs)
            return np.ones_like(kwargs["frequency_array"], dtype=complex)

        def _calculate_snrs(
            self, glitch_strain, interferometer, return_array=True,
            parameters=None
        ):
            return DummyCalculatedSNRs(
                d_inner_h=np.sum(glitch_strain),
                optimal_snr_squared=float(np.sum(np.abs(glitch_strain) ** 2))
            )

        likelihood._glitch_model = _glitch_model
        monkeypatch.setattr(Glitch, "calculate_snrs", _calculate_snrs)

        log_l = likelihood.log_likelihood_ratio(
            {
                "A": 1.,
                "f": 30.,
                "phi": 0.,
                "tc": 0.1,
                "log_gamma": -2.,
            }
        )

        assert len(calls) == 2
        assert log_l == 4.

    def test_log_likelihood_ratio_uses_self_parameters_if_none_are_passed(
        self, monkeypatch
    ):
        likelihood = self._base_glitch_likelihood()
        likelihood.parameters = {
            "A": 1.,
            "f": 30.,
            "phi": 0.,
            "tc": 0.1,
            "gamma": 0.2,
        }

        def _calculate_snrs(
            self, glitch_strain, interferometer, return_array=True,
            parameters=None
        ):
            return DummyCalculatedSNRs(
                d_inner_h=np.sum(glitch_strain),
                optimal_snr_squared=float(np.sum(np.abs(glitch_strain) ** 2))
            )

        monkeypatch.setattr(Glitch, "calculate_snrs", _calculate_snrs)

        log_l = likelihood.log_likelihood_ratio()

        assert log_l == 4.

    def test_log_likelihood_ratio_does_not_mutate_parameters(self, monkeypatch):
        likelihood = self._base_glitch_likelihood()
        params = {
            "A": 1.,
            "f": 30.,
            "phi": 0.,
            "tc": 0.1,
            "gamma": 0.2,
        }

        def _calculate_snrs(
            self, glitch_strain, interferometer, return_array=True,
            parameters=None
        ):
            parameters["new_key"] = "value"
            return DummyCalculatedSNRs()

        monkeypatch.setattr(Glitch, "calculate_snrs", _calculate_snrs)

        likelihood.log_likelihood_ratio(params)

        assert "new_key" not in params


class TestGravitationalWaveTransientPlusGlitch(BaseTest):
    """Tests for the CBC + glitch likelihood."""

    def test_log_likelihood_ratio_filters_out_glitch_parameters(
        self, monkeypatch
    ):
        likelihood = self._base_plus_glitch_likelihood()
        capture = {}

        def _frequency_domain_strain(params):
            capture["waveform_parameters"] = dict(params)
            return {"plus": np.ones(4, dtype=complex)}

        def _get_sky_frame_parameters(params):
            capture["sky_parameters"] = dict(params)
            return {"zenith": 0.1, "azimuth": 0.2}

        def _calculate_snrs(
            self, waveform_polarizations, interferometer, glitch_strain,
            return_array=True, parameters=None
        ):
            return DummyCalculatedSNRs(
                d_inner_h=3.,
                optimal_snr_squared=2.
            )

        self.waveform_generator.frequency_domain_strain = (
            _frequency_domain_strain
        )
        likelihood.get_sky_frame_parameters = _get_sky_frame_parameters
        monkeypatch.setattr(
            GravitationalWaveTransientPlusGlitch,
            "calculate_snrs",
            _calculate_snrs
        )

        log_l = likelihood.log_likelihood_ratio(
            {
                "mass_1": 30.,
                "mass_2": 20.,
                "luminosity_distance": 100.,
                "A": 1.,
                "f": 40.,
                "phi": 0.1,
                "tc": 0.2,
                "gamma": 0.3,
                "H1_A": 9.,
                "H1_log_gamma": -2.,
            }
        )

        assert capture["waveform_parameters"] == {
            "mass_1": 30.,
            "mass_2": 20.,
            "luminosity_distance": 100.,
        }
        assert capture["sky_parameters"] == {
            "mass_1": 30.,
            "mass_2": 20.,
            "luminosity_distance": 100.,
        }
        assert log_l == 4.

    def test_log_likelihood_ratio_returns_negative_infinity_when_waveform_is_none(
        self
    ):
        likelihood = self._base_plus_glitch_likelihood()
        self.waveform_generator.frequency_domain_strain = lambda params: None

        result = likelihood.log_likelihood_ratio(
            {
                "mass_1": 30.,
                "mass_2": 20.,
                "A": 1.,
                "f": 40.,
                "phi": 0.1,
                "tc": 0.2,
                "gamma": 0.3,
            }
        )

        assert result == np.nan_to_num(-np.inf)

    def test_calculate_snrs_combines_cbc_and_glitch(self):
        likelihood = self._base_plus_glitch_likelihood()
        likelihood._compute_full_waveform = (
            lambda signal_polarizations, interferometer, parameters:
            np.array([0., 1. + 0.j, 2. + 0.j, 0.])
        )
        glitch = np.array([0., 0.5 + 0.j, 0.5 + 0.j, 0.])

        snrs = likelihood.calculate_snrs(
            waveform_polarizations={"plus": np.ones(4, dtype=complex)},
            interferometer=self.ifos[0],
            glitch_strain=glitch,
            parameters={}
        )

        assert snrs.d_inner_h == 4. + 0.j
        assert snrs.optimal_snr_squared == 1.5 ** 2 + 2.5 ** 2
        assert snrs.complex_matched_filter_snr == pytest.approx(
            (4. + 0.j) / np.sqrt(1.5 ** 2 + 2.5 ** 2)
        )

    def test_calculate_snrs_returns_calibration_arrays_when_needed(self):
        likelihood = self._base_plus_glitch_likelihood()
        likelihood.calibration_marginalization = True
        likelihood._compute_full_waveform = (
            lambda signal_polarizations, interferometer, parameters:
            np.array([0., 1. + 0.j, 2. + 0.j, 0.])
        )
        glitch = np.array([0., 0.5 + 0.j, 0.5 + 0.j, 0.])

        snrs = likelihood.calculate_snrs(
            waveform_polarizations={"plus": np.ones(4, dtype=complex)},
            interferometer=self.ifos[0],
            glitch_strain=glitch,
            parameters={}
        )

        signal = np.array([0., 1.5 + 0.j, 2.5 + 0.j, 0.])
        mask = self.ifos[0].frequency_mask
        psd = self.ifos[0].power_spectral_density_array
        data = self.ifos[0].frequency_domain_strain
        normalization = 4. / self.waveform_generator.duration

        d_inner_h_integrand = (
            normalization * data.conjugate() * signal / psd
        )
        expected_d_inner_h_array = np.dot(
            d_inner_h_integrand[mask],
            likelihood.calibration_draws["H1"].T
        )

        optimal_integrand = normalization * np.abs(signal) ** 2 / psd
        expected_optimal_array = np.dot(
            optimal_integrand[mask],
            likelihood.calibration_abs_draws["H1"].T
        )

        np.testing.assert_allclose(
            snrs.d_inner_h_array,
            expected_d_inner_h_array
        )
        np.testing.assert_allclose(
            snrs.optimal_snr_squared_array,
            expected_optimal_array
        )

    def test_calculate_snrs_return_array_false(self):
        likelihood = self._base_plus_glitch_likelihood()
        likelihood.calibration_marginalization = True
        likelihood._compute_full_waveform = (
            lambda signal_polarizations, interferometer, parameters:
            np.array([0., 1. + 0.j, 2. + 0.j, 0.])
        )
        glitch = np.array([0., 0.5 + 0.j, 0.5 + 0.j, 0.])

        snrs = likelihood.calculate_snrs(
            waveform_polarizations={"plus": np.ones(4, dtype=complex)},
            interferometer=self.ifos[0],
            glitch_strain=glitch,
            parameters={},
            return_array=False
        )

        assert snrs.d_inner_h_array is None
        assert snrs.optimal_snr_squared_array is None

    def test_log_likelihood_ratio_does_not_mutate_parameters(self, monkeypatch):
        likelihood = self._base_plus_glitch_likelihood()
        params = {
            "mass_1": 30.,
            "mass_2": 20.,
            "A": 1.,
            "f": 40.,
            "phi": 0.1,
            "tc": 0.2,
            "gamma": 0.3,
        }

        self.waveform_generator.frequency_domain_strain = (
            lambda params: {"plus": np.ones(4, dtype=complex)}
        )

        def _calculate_snrs(
            self, waveform_polarizations, interferometer, glitch_strain,
            return_array=True, parameters=None
        ):
            parameters["new_key"] = "value"
            return DummyCalculatedSNRs()

        monkeypatch.setattr(
            GravitationalWaveTransientPlusGlitch,
            "calculate_snrs",
            _calculate_snrs
        )

        likelihood.log_likelihood_ratio(params)

        assert "new_key" not in params


class TestAgreementWithStandardGravitationalWaveTransient(BaseTest):
    """Agreement tests with the standard bilby GW transient likelihood."""
    def test_plus_glitch_calculate_snrs_matches_standard_gw_for_zero_cbc_and_unit_glitch(
        self
    ):
        plus_glitch_likelihood = self._base_plus_glitch_likelihood()
        standard_likelihood = self._base_standard_gw_likelihood()

        unit_signal = np.ones(4, dtype=complex)

        plus_glitch_likelihood._compute_full_waveform = (
            lambda signal_polarizations, interferometer, parameters:
            np.zeros(4, dtype=complex)
        )
        standard_likelihood._compute_full_waveform = (
            lambda signal_polarizations, interferometer, parameters:
            unit_signal.copy()
        )

        plus_glitch_snrs = plus_glitch_likelihood.calculate_snrs(
            waveform_polarizations={"plus": unit_signal.copy()},
            interferometer=self.ifos[0],
            glitch_strain=unit_signal.copy(),
            parameters={}
        )
        standard_snrs = standard_likelihood.calculate_snrs(
            {"plus": unit_signal.copy()},
            self.ifos[0],
            parameters={}
        )

        assert plus_glitch_snrs.d_inner_h == standard_snrs.d_inner_h
        assert plus_glitch_snrs.optimal_snr_squared == (
            standard_snrs.optimal_snr_squared
        )
        assert plus_glitch_snrs.complex_matched_filter_snr == pytest.approx(
            standard_snrs.complex_matched_filter_snr
        )

    def test_plus_glitch_compute_log_likelihood_matches_standard_gw_for_zero_cbc_and_unit_glitch(
        self
    ):
        plus_glitch_likelihood = self._base_plus_glitch_likelihood()
        standard_likelihood = self._base_standard_gw_likelihood()

        plus_glitch_likelihood._compute_full_waveform = (
            lambda signal_polarizations, interferometer, parameters:
            np.zeros(4, dtype=complex)
        )
        standard_likelihood._compute_full_waveform = (
            lambda signal_polarizations, interferometer, parameters:
            np.ones(4, dtype=complex)
        )

        plus_glitch_total = self._network_plus_glitch_snrs(
            plus_glitch_likelihood,
            waveform_polarizations={"plus": np.ones(4, dtype=complex)},
            glitch_signal=np.ones(4, dtype=complex),
            parameters={}
        )
        standard_total = self._network_standard_snrs(
            standard_likelihood,
            np.ones(4, dtype=complex),
            parameters={}
        )

        plus_glitch_log_l = (
            plus_glitch_likelihood.compute_log_likelihood_from_snrs(
                plus_glitch_total,
                parameters={}
            )
        )
        standard_log_l = standard_likelihood.compute_log_likelihood_from_snrs(
            standard_total,
            parameters={}
        )

        assert plus_glitch_total.d_inner_h == standard_total.d_inner_h
        assert plus_glitch_total.optimal_snr_squared == (
            standard_total.optimal_snr_squared
        )
        assert plus_glitch_log_l == pytest.approx(standard_log_l)

    def test_plus_glitch_log_likelihood_ratio_matches_standard_gw_for_zero_cbc_and_unit_glitch(
        self
    ):
        plus_glitch_likelihood = self._base_plus_glitch_likelihood()
        standard_likelihood = self._base_standard_gw_likelihood()

        plus_glitch_likelihood._glitch_model = unit_glitch_model
        plus_glitch_likelihood._glitch_params = ["A", "f", "phi", "tc", "gamma"]
        plus_glitch_likelihood._compute_full_waveform = (
            lambda signal_polarizations, interferometer, parameters:
            np.zeros(4, dtype=complex)
        )
        standard_likelihood._compute_full_waveform = (
            lambda signal_polarizations, interferometer, parameters:
            np.ones(4, dtype=complex)
        )

        self.waveform_generator.frequency_domain_strain = (
            lambda parameters: {"plus": np.ones(4, dtype=complex)}
        )

        params = {
            "mass_1": 30.,
            "mass_2": 20.,
            "luminosity_distance": 100.,
            "A": 1.,
            "f": 40.,
            "phi": 0.1,
            "tc": 0.2,
            "gamma": 0.3,
        }

        plus_glitch_log_l = plus_glitch_likelihood.log_likelihood_ratio(params)
        standard_total = self._network_standard_snrs(
            standard_likelihood,
            np.ones(4, dtype=complex),
            parameters={}
        )
        standard_log_l = standard_likelihood.compute_log_likelihood_from_snrs(
            standard_total,
            parameters={}
        )

        assert plus_glitch_log_l == pytest.approx(standard_log_l)

    def test_plus_glitch_log_likelihood_ratio_matches_single_detector_standard_gw_limit(
        self
    ):
        plus_glitch_likelihood = self._base_plus_glitch_likelihood()
        standard_likelihood = self._base_standard_gw_likelihood()

        plus_glitch_likelihood._glitch_model = unit_glitch_model
        plus_glitch_likelihood._glitch_params = ["A", "f", "phi", "tc", "gamma"]
        plus_glitch_likelihood.detectors_glitch = ["H1"]
        plus_glitch_likelihood._compute_full_waveform = (
            lambda signal_polarizations, interferometer, parameters:
            np.zeros(4, dtype=complex)
        )

        def _standard_waveform(signal_polarizations, interferometer, parameters):
            if interferometer.name == "H1":
                return np.ones(4, dtype=complex)
            return np.zeros(4, dtype=complex)

        standard_likelihood._compute_full_waveform = _standard_waveform
        self.waveform_generator.frequency_domain_strain = (
            lambda parameters: {"plus": np.ones(4, dtype=complex)}
        )

        params = {
            "mass_1": 30.,
            "mass_2": 20.,
            "luminosity_distance": 100.,
            "A": 1.,
            "f": 40.,
            "phi": 0.1,
            "tc": 0.2,
            "gamma": 0.3,
        }

        plus_glitch_log_l = plus_glitch_likelihood.log_likelihood_ratio(params)

        standard_total = standard_likelihood._CalculatedSNRs()
        standard_total += standard_likelihood.calculate_snrs(
            {"plus": np.ones(4, dtype=complex)},
            self.ifos[0],
            parameters={}
        )
        standard_total += standard_likelihood.calculate_snrs(
            {"plus": np.zeros(4, dtype=complex)},
            self.ifos[1],
            parameters={}
        )
        standard_log_l = standard_likelihood.compute_log_likelihood_from_snrs(
            standard_total,
            parameters={}
        )

        assert plus_glitch_log_l == pytest.approx(standard_log_l)
