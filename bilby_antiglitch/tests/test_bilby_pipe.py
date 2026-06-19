import sys
import types
import copy
import numpy as np
import pandas as pd
import pytest

from bilby_antiglitch import bilby_pipe as module
from bilby_antiglitch.bilby_pipe import (
    Input,
    MainInput,
    DataGenerationInput,
    DataAnalysisInput,
    create_parser,
)


class DummyPrior(object):
    """Simple prior object used for testing prior transformations."""

    def __init__(self, name):
        self.name = name


class DummyLikelihood(object):
    """Simple likelihood class used to capture constructor kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyStrainData(object):
    def __init__(self, frequency_domain_strain):
        self.frequency_domain_strain = frequency_domain_strain


class DummyInterferometer(object):
    def __init__(self, name):
        self.name = name
        self.start_time = 100.
        self.frequency_array = np.array([10., 20., 30., 40.])
        self.power_spectral_density_array = np.array([1., 2., 2., 1.])
        self.strain_data = DummyStrainData(
            np.zeros(4, dtype=complex)
        )


class DummyLogger(object):
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


class DummyParserGroup(object):
    def __init__(self, name):
        self.name = name
        self.arguments = []

    def add_argument(self, *args, **kwargs):
        self.arguments.append((args, kwargs))


class DummyParser(object):
    def __init__(self):
        self.defaults = {}
        self.groups = []
        self.top_level = None

    def set_defaults(self, **kwargs):
        self.defaults.update(kwargs)

    def add_argument_group(self, name):
        group = DummyParserGroup(name)
        self.groups.append(group)
        return group


def glitch_model(
    A=None, f=None, phi=None, tc=None, gamma=None,
    frequency_array=None, psd=None, tstart=None, **kwargs
):
    return np.ones_like(frequency_array, dtype=complex)


def other_glitch_model(
    A=None, f=None, frequency_array=None, psd=None, tstart=None, **kwargs
):
    return np.ones_like(frequency_array, dtype=complex) * 2.


def _dummy_input_parent_init(self, *args, **kwargs):
    """Dummy parent __init__ used to isolate input initialisation tests."""
    self._interferometers = [
        DummyInterferometer("H1"),
        DummyInterferometer("L1"),
    ]
    self.detectors = ["H1", "L1"]
    self.priors = {}
    self.waveform_generator = object()
    self.phase_marginalization = False
    self.distance_marginalization = False
    self.time_marginalization = False
    self.likelihood_type = "GravitationalWaveTransient"
    self._likelihood = "parent-likelihood"


class BaseTest(object):
    """Shared helpers for the bilby_pipe input tests."""

    @pytest.fixture(autouse=True)
    def setup_test(self, monkeypatch):
        self.logger = DummyLogger()
        monkeypatch.setattr(module, "logger", self.logger, raising=False)

        if not hasattr(module, "BilbyPipeError"):
            class BilbyPipeError(Exception):
                pass

            monkeypatch.setattr(
                module, "BilbyPipeError", BilbyPipeError, raising=False
            )

    def _patch_direct_parent_init(self, monkeypatch, cls):
        """Patch the next __init__ reached by super() for init tests."""
        parent = cls.__mro__[1]
        monkeypatch.setattr(parent, "__init__", _dummy_input_parent_init)

    def _base_input(self):
        obj = Input.__new__(Input)
        obj.detectors = ["H1", "L1"]
        obj._detectors_glitch = ["H1", "L1"]
        obj._glitch_frequency_domain_source_model = None
        obj._injection_glitch_frequency_domain_source_model = None
        obj.priors = {}
        obj.interferometers = [
            DummyInterferometer("H1"),
            DummyInterferometer("L1"),
        ]
        obj.waveform_generator = object()
        obj.phase_marginalization = False
        obj.distance_marginalization = False
        obj.time_marginalization = False
        obj.likelihood_type = "GravitationalWaveTransient"
        obj._likelihood = None
        return obj

    def _base_data_generation_input(self):
        obj = DataGenerationInput.__new__(DataGenerationInput)
        obj.detectors = ["H1", "L1"]
        obj._detectors_glitch = ["H1", "L1"]
        obj._glitch_frequency_domain_source_model = glitch_model
        obj._injection_glitch_frequency_domain_source_model = glitch_model
        obj.priors = {}
        obj.interferometers = [
            DummyInterferometer("H1"),
            DummyInterferometer("L1"),
        ]
        obj.idx = 0
        obj.injection_df = pd.DataFrame(
            [
                {
                    "mass_1": 30.,
                    "mass_2": 20.,
                    "A": 1.,
                    "f": 40.,
                    "phi": 0.1,
                    "tc": 0.2,
                    "gamma": 0.3,
                }
            ]
        )
        obj.injection_parameters = {}
        return obj

    def _install_fake_convert_detectors_input(self, monkeypatch, return_value):
        utils_module = types.ModuleType("bilby_pipe.utils")
        utils_module.convert_detectors_input = lambda value: return_value
        monkeypatch.setitem(sys.modules, "bilby_pipe.utils", utils_module)

    def _install_fake_parser_module(self, monkeypatch, parser):
        parser_module = types.ModuleType("bilby_pipe.parser")
        parser_module.create_parser = lambda top_level=False: parser
        monkeypatch.setitem(sys.modules, "bilby_pipe.parser", parser_module)

    def _install_fake_likelihood_module(
        self, monkeypatch, glitch_cls=None, plus_glitch_cls=None
    ):
        package = module.__name__.rsplit(".", 1)[0]
        likelihood_module = types.ModuleType(f"{package}.likelihood")
        likelihood_module.Glitch = (
            DummyLikelihood if glitch_cls is None else glitch_cls
        )
        likelihood_module.GravitationalWaveTransientPlusGlitch = (
            DummyLikelihood if plus_glitch_cls is None else plus_glitch_cls
        )
        monkeypatch.setitem(
            sys.modules, f"{package}.likelihood", likelihood_module
        )


class TestInputProperties(BaseTest):
    """Tests for Input property setters/getters."""

    def test_glitch_frequency_domain_source_model_getter(self):
        obj = self._base_input()
        obj._glitch_frequency_domain_source_model = glitch_model

        assert obj.glitch_frequency_domain_source_model is glitch_model

    def test_glitch_frequency_domain_source_model_setter_accepts_none_string(
        self
    ):
        obj = self._base_input()
        obj.glitch_frequency_domain_source_model = "None"

        assert obj._glitch_frequency_domain_source_model is None

    def test_glitch_frequency_domain_source_model_setter_resolves_string_path(
        self, monkeypatch
    ):
        obj = self._base_input()

        calls = []

        def _resolver(path):
            calls.append(path)
            return glitch_model

        monkeypatch.setattr(module, "get_function_from_string_path", _resolver)

        obj.glitch_frequency_domain_source_model = "a.b.c"

        assert calls == ["a.b.c"]
        assert obj._glitch_frequency_domain_source_model is glitch_model

    def test_glitch_frequency_domain_source_model_setter_accepts_callable(self):
        obj = self._base_input()
        obj.glitch_frequency_domain_source_model = glitch_model

        assert obj._glitch_frequency_domain_source_model is glitch_model

    def test_injection_glitch_frequency_domain_source_model_getter(self):
        obj = self._base_input()
        obj._injection_glitch_frequency_domain_source_model = other_glitch_model

        assert obj.injection_glitch_frequency_domain_source_model is (
            other_glitch_model
        )

    def test_injection_glitch_frequency_domain_source_model_defaults_to_glitch_model_with_none(
        self
    ):
        obj = self._base_input()
        obj._glitch_frequency_domain_source_model = glitch_model

        obj.injection_glitch_frequency_domain_source_model = None

        assert obj._injection_glitch_frequency_domain_source_model is glitch_model

    def test_injection_glitch_frequency_domain_source_model_defaults_to_glitch_model_with_none_string(
        self
    ):
        obj = self._base_input()
        obj._glitch_frequency_domain_source_model = glitch_model

        obj.injection_glitch_frequency_domain_source_model = "None"

        assert obj._injection_glitch_frequency_domain_source_model is glitch_model

    def test_injection_glitch_frequency_domain_source_model_resolves_string(
        self, monkeypatch
    ):
        obj = self._base_input()
        calls = []

        def _resolver(path):
            calls.append(path)
            return other_glitch_model

        monkeypatch.setattr(module, "get_function_from_string_path", _resolver)

        obj.injection_glitch_frequency_domain_source_model = "x.y.z"

        assert calls == ["x.y.z"]
        assert obj._injection_glitch_frequency_domain_source_model is (
            other_glitch_model
        )

    def test_detectors_glitch_defaults_to_detectors(self):
        obj = self._base_input()
        obj.detectors = ["H1", "L1", "V1"]

        obj.detectors_glitch = None

        assert obj._detectors_glitch == ["H1", "L1", "V1"]

    def test_detectors_glitch_uses_convert_detectors_input(self, monkeypatch):
        obj = self._base_input()
        self._install_fake_convert_detectors_input(
            monkeypatch, return_value=["H1", "K1"]
        )

        obj.detectors_glitch = ["H1", "K1"]

        assert obj._detectors_glitch == ["H1", "K1"]


class TestInputGetPriors(BaseTest):
    """Tests for Input._get_priors."""

    def test_get_priors_returns_parent_priors_if_glitch_model_is_none(
        self, monkeypatch
    ):
        obj = self._base_input()
        obj._glitch_frequency_domain_source_model = None

        base_priors = {
            "mass_1": DummyPrior("mass_1"),
            "A": DummyPrior("A"),
        }

        parent = Input.__mro__[1]
        monkeypatch.setattr(
            parent, "_get_priors", lambda self, **kwargs: base_priors
        )

        result = obj._get_priors()

        assert result is base_priors
        assert "A" in result
        assert result["A"].name == "A"

    def test_get_priors_returns_parent_priors_if_no_interferometers(
        self, monkeypatch
    ):
        obj = self._base_input()
        obj._glitch_frequency_domain_source_model = glitch_model
        del obj.interferometers

        base_priors = {
            "A": DummyPrior("A"),
            "f": DummyPrior("f"),
        }

        parent = Input.__mro__[1]
        monkeypatch.setattr(
            parent, "_get_priors", lambda self, **kwargs: base_priors
        )

        result = obj._get_priors()

        assert result is base_priors
        assert set(result.keys()) == {"A", "f"}

    def test_get_priors_transforms_glitch_priors_to_detector_specific(
        self, monkeypatch
    ):
        obj = self._base_input()
        obj._glitch_frequency_domain_source_model = glitch_model
        obj._detectors_glitch = ["H1", "L1"]

        base_priors = {
            "mass_1": DummyPrior("mass_1"),
            "A": DummyPrior("A"),
            "f": DummyPrior("f"),
            "gamma": DummyPrior("gamma"),
            "phi": DummyPrior("phi"),
            "tc": DummyPrior("tc"),
            "log_A": DummyPrior("log_A"),
        }

        parent = Input.__mro__[1]
        monkeypatch.setattr(
            parent, "_get_priors",
            lambda self, **kwargs: copy.deepcopy(base_priors)
        )

        result = obj._get_priors()

        assert "mass_1" in result
        assert "A" not in result
        assert "f" not in result
        assert "gamma" not in result
        assert "phi" not in result
        assert "tc" not in result
        assert "log_A" not in result

        for key in [
            "H1_A", "L1_A", "H1_f", "L1_f", "H1_gamma", "L1_gamma",
            "H1_phi", "L1_phi", "H1_tc", "L1_tc", "H1_log_A", "L1_log_A"
        ]:
            assert key in result
            assert result[key].name == key

        assert obj._glitch_params == ["A", "f", "phi", "tc", "gamma", "log_A"]

    def test_get_priors_only_creates_priors_for_requested_glitch_detectors(
        self, monkeypatch
    ):
        obj = self._base_input()
        obj._glitch_frequency_domain_source_model = glitch_model
        obj._detectors_glitch = ["H1"]

        base_priors = {
            "A": DummyPrior("A"),
            "f": DummyPrior("f"),
        }

        parent = Input.__mro__[1]
        monkeypatch.setattr(
            parent, "_get_priors",
            lambda self, **kwargs: copy.deepcopy(base_priors)
        )

        result = obj._get_priors()

        assert "H1_A" in result
        assert "H1_f" in result
        assert "L1_A" not in result
        assert "L1_f" not in result

    def test_get_priors_does_not_overwrite_existing_detector_specific_prior(
        self, monkeypatch
    ):
        obj = self._base_input()
        obj._glitch_frequency_domain_source_model = glitch_model
        obj._detectors_glitch = ["H1", "L1"]

        h1_prior = DummyPrior("H1_A")
        base_priors = {
            "A": DummyPrior("A"),
            "H1_A": h1_prior,
        }

        parent = Input.__mro__[1]
        monkeypatch.setattr(
            parent, "_get_priors",
            lambda self, **kwargs: copy.deepcopy(base_priors)
        )

        result = obj._get_priors()

        assert "A" not in result
        assert "H1_A" in result
        assert "L1_A" in result
        assert result["H1_A"].name == "H1_A"
        assert result["L1_A"].name == "L1_A"


class TestInputLikelihood(BaseTest):
    """Tests for Input.likelihood."""

    def test_likelihood_delegates_to_parent_if_no_glitch_model(
        self, monkeypatch
    ):
        obj = self._base_input()
        obj._glitch_frequency_domain_source_model = None
        obj.priors = {"mass_1": DummyPrior("mass_1")}

        sentinel = object()

        parent = Input.__mro__[1]
        monkeypatch.setattr(
            parent,
            "likelihood",
            property(lambda self: sentinel),
        )

        result = obj.likelihood

        assert list(obj.search_priors.keys()) == list(obj.priors.keys())

    def test_likelihood_builds_glitch_likelihood(self, monkeypatch):
        obj = self._base_input()
        obj._glitch_frequency_domain_source_model = glitch_model
        obj._detectors_glitch = ["H1"]
        obj.likelihood_type = "Glitch"
        obj.priors = {"A": DummyPrior("A")}

        self._install_fake_likelihood_module(
            monkeypatch,
            glitch_cls=DummyLikelihood,
            plus_glitch_cls=DummyLikelihood,
        )

        result = obj.likelihood

        assert isinstance(result, DummyLikelihood)
        assert result.kwargs["interferometers"] == obj.interferometers
        assert result.kwargs["waveform_generator"] is obj.waveform_generator
        assert result.kwargs["priors"] == obj.priors
        assert result.kwargs["phase_marginalization"] is False
        assert result.kwargs["distance_marginalization"] is False
        assert result.kwargs["time_marginalization"] is False
        assert result.kwargs["glitch_model"] is glitch_model
        assert result.kwargs["detectors_glitch"] == ["H1"]

    def test_likelihood_builds_plus_glitch_likelihood(self, monkeypatch):
        obj = self._base_input()
        obj._glitch_frequency_domain_source_model = glitch_model
        obj._detectors_glitch = ["H1", "L1"]
        obj.likelihood_type = "GravitationalWaveTransientPlusGlitch"

        class DummyPlusGlitchLikelihood(DummyLikelihood):
            pass

        self._install_fake_likelihood_module(
            monkeypatch,
            glitch_cls=DummyLikelihood,
            plus_glitch_cls=DummyPlusGlitchLikelihood,
        )

        result = obj.likelihood

        assert isinstance(result, DummyPlusGlitchLikelihood)
        assert result.kwargs["glitch_model"] is glitch_model
        assert result.kwargs["detectors_glitch"] == ["H1", "L1"]

    def test_likelihood_raises_for_invalid_likelihood_type_with_glitch_model(
        self, monkeypatch
    ):
        obj = self._base_input()
        obj._glitch_frequency_domain_source_model = glitch_model
        obj.likelihood_type = "GravitationalWaveTransient"

        self._install_fake_likelihood_module(monkeypatch)

        with pytest.raises(module.BilbyPipeError, match="glitch likelihood"):
            obj.likelihood


class TestMainInputInit(BaseTest):
    """Tests for MainInput.__init__."""

    def test_init_copies_glitch_attributes_from_first_argument(
        self, monkeypatch
    ):
        self._patch_direct_parent_init(monkeypatch, MainInput)

        source = types.SimpleNamespace(
            glitch_frequency_domain_source_model=glitch_model,
            injection_glitch_frequency_domain_source_model=other_glitch_model,
            detectors_glitch=["H1"],
        )

        obj = MainInput(source)

        assert obj.glitch_frequency_domain_source_model is glitch_model
        assert obj.injection_glitch_frequency_domain_source_model is (
            other_glitch_model
        )
        assert obj.detectors_glitch == ["H1"]


class TestDataAnalysisInputInit(BaseTest):
    """Tests for DataAnalysisInput.__init__."""

    def test_init_copies_glitch_attributes_from_first_argument(
        self, monkeypatch
    ):
        self._patch_direct_parent_init(monkeypatch, DataAnalysisInput)

        source = types.SimpleNamespace(
            glitch_frequency_domain_source_model=glitch_model,
            injection_glitch_frequency_domain_source_model=other_glitch_model,
            detectors_glitch=["H1", "L1"],
        )

        obj = DataAnalysisInput(source)

        assert obj.glitch_frequency_domain_source_model is glitch_model
        assert obj.injection_glitch_frequency_domain_source_model is (
            other_glitch_model
        )
        assert obj.detectors_glitch == ["H1", "L1"]


class TestDataGenerationInputInit(BaseTest):
    """Tests for DataGenerationInput.__init__."""

    def test_init_copies_glitch_attributes_from_first_argument(
        self, monkeypatch
    ):
        self._patch_direct_parent_init(monkeypatch, DataGenerationInput)

        source = types.SimpleNamespace(
            glitch_frequency_domain_source_model=glitch_model,
            injection_glitch_frequency_domain_source_model=other_glitch_model,
            detectors_glitch=["H1"],
        )

        obj = DataGenerationInput(source)

        assert obj.glitch_frequency_domain_source_model is glitch_model
        assert obj.injection_glitch_frequency_domain_source_model is (
            other_glitch_model
        )
        assert obj.detectors_glitch == ["H1"]


class TestDataGenerationInputSetInterferometers(BaseTest):
    """Tests for DataGenerationInput._set_interferometers_from_injection_in_gaussian_noise."""

    def test_set_interferometers_from_injection_in_gaussian_noise_injects_glitch(
        self, monkeypatch
    ):
        obj = self._base_data_generation_input()
        parent_capture = {}

        def _parent_method(self):
            parent_capture["injection_parameters"] = dict(self.injection_parameters)
            parent_capture["injection_df_columns"] = list(self.injection_df.columns)

        monkeypatch.setattr(
            Input,
            "_set_interferometers_from_injection_in_gaussian_noise",
            _parent_method,
            raising=False,
        )

        obj._set_interferometers_from_injection_in_gaussian_noise()

        assert parent_capture["injection_parameters"] == {
            "mass_1": 30.,
            "mass_2": 20.,
        }
        assert parent_capture["injection_df_columns"] == ["mass_1", "mass_2"]

        for ifo in obj.interferometers:
            np.testing.assert_array_equal(
                ifo.strain_data.frequency_domain_strain,
                np.ones(4, dtype=complex)
            )

        assert list(obj.injection_df.columns) == [
            "mass_1", "mass_2", "A", "f", "phi", "tc", "gamma"
        ]

    def test_set_interferometers_from_injection_in_gaussian_noise_only_injects_for_requested_detectors(
        self, monkeypatch
    ):
        obj = self._base_data_generation_input()
        obj._detectors_glitch = ["H1"]

        monkeypatch.setattr(
            Input,
            "_set_interferometers_from_injection_in_gaussian_noise",
            lambda self: None,
            raising=False,
        )

        obj._set_interferometers_from_injection_in_gaussian_noise()

        np.testing.assert_array_equal(
            obj.interferometers[0].strain_data.frequency_domain_strain,
            np.ones(4, dtype=complex)
        )
        np.testing.assert_array_equal(
            obj.interferometers[1].strain_data.frequency_domain_strain,
            np.zeros(4, dtype=complex)
        )

    def test_set_interferometers_from_injection_in_gaussian_noise_prefers_detector_specific_glitch_parameters(
        self, monkeypatch
    ):
        obj = self._base_data_generation_input()
        obj.injection_df = pd.DataFrame(
            [
                {
                    "mass_1": 30.,
                    "mass_2": 20.,
                    "A": 1.,
                    "f": 40.,
                    "phi": 0.1,
                    "tc": 0.2,
                    "gamma": 0.3,
                }
            ]
        )

        calls = []

        def _glitch_model(**kwargs):
            calls.append(kwargs)
            return np.ones_like(kwargs["frequency_array"], dtype=complex)

        obj._injection_glitch_frequency_domain_source_model = _glitch_model

        monkeypatch.setattr(
            Input,
            "_set_interferometers_from_injection_in_gaussian_noise",
            lambda self: None,
            raising=False,
        )

        obj._set_interferometers_from_injection_in_gaussian_noise()

        assert len(calls) == 2
        for call in calls:
            assert call["A"] == 1.
            assert call["f"] == 40.
            assert call["phi"] == 0.1
            assert call["tc"] == 0.2
            assert call["gamma"] == 0.3
            assert "tstart" in call
            assert "frequency_array" in call
            assert "psd" in call

    def test_set_interferometers_from_injection_in_gaussian_noise_logs_injected_parameters(
        self, monkeypatch
    ):
        obj = self._base_data_generation_input()

        monkeypatch.setattr(
            Input,
            "_set_interferometers_from_injection_in_gaussian_noise",
            lambda self: None,
            raising=False,
        )

        obj._set_interferometers_from_injection_in_gaussian_noise()

        assert any("Injected glitch in H1" in msg for msg in self.logger.messages)
        assert any("Injected glitch in L1" in msg for msg in self.logger.messages)
        assert any("A = 1.0" in msg for msg in self.logger.messages)
        assert any("f = 40.0" in msg for msg in self.logger.messages)

    def test_set_interferometers_from_injection_in_gaussian_noise_restores_injection_df(
        self, monkeypatch
    ):
        obj = self._base_data_generation_input()
        original = obj.injection_df.copy()

        monkeypatch.setattr(
            Input,
            "_set_interferometers_from_injection_in_gaussian_noise",
            lambda self: None,
            raising=False,
        )

        obj._set_interferometers_from_injection_in_gaussian_noise()

        pd.testing.assert_frame_equal(obj.injection_df, original)


class TestCreateParser(BaseTest):
    """Tests for create_parser."""

    def test_create_parser_sets_defaults_and_adds_glitch_arguments(
        self, monkeypatch
    ):
        parser = DummyParser()
        self._install_fake_parser_module(monkeypatch, parser)

        result = create_parser(top_level=True)

        assert result is parser
        assert parser.defaults == {
            "main_input_class": "bilby_antiglitch.bilby_pipe.MainInput",
            "analysis_input_class": (
                "bilby_antiglitch.bilby_pipe.DataAnalysisInput"
            ),
            "generation_executable_parser": (
                "bilby_antiglitch.bilby_pipe.create_parser"
            ),
            "generation_input_class": (
                "bilby_antiglitch.bilby_pipe.DataGenerationInput"
            ),
        }

        assert len(parser.groups) == 1
        group = parser.groups[0]
        assert group.name == "Glitch models"
        assert len(group.arguments) == 3

        argument_names = [args[0][0] for args in group.arguments]
        assert "--glitch-frequency-domain-source-model" in argument_names
        assert "--injection-glitch-frequency-domain-source-model" in (
            argument_names
        )
        assert "--detectors-glitch" in argument_names

    def test_create_parser_adds_detectors_glitch_with_append_action(
        self, monkeypatch
    ):
        parser = DummyParser()
        self._install_fake_parser_module(monkeypatch, parser)

        create_parser(top_level=False)

        group = parser.groups[0]
        detectors_argument = None
        for args, kwargs in group.arguments:
            if args[0] == "--detectors-glitch":
                detectors_argument = kwargs
                break

        assert detectors_argument is not None
        assert detectors_argument["action"] == "append"
