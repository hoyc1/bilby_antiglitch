# Licensed under an MIT style license -- see LICENSE.md

from bilby_pipe.input import Input as _Input
from bilby_pipe.main import MainInput as _MainInput, main
from bilby_pipe.data_analysis import DataAnalysisInput as _DAInput
from bilby_pipe.data_generation import DataGenerationInput as _DGInput
from bilby_pipe.utils import logger, BilbyPipeError, get_function_from_string_path
from bilby_pipe.parser import BilbyArgParser as _BilbyArgParser
from bilby.core.utils import logger
import inspect
import copy

__author__ = ["Charlie Hoy <charlie.hoy@port.ac.uk>"]


class Input(_Input):
    """Superclass of input handlers inherited from bilby_pipe.input.Input
    """
    @property
    def glitch_frequency_domain_source_model(self):
        """String of which frequency domain source model to use"""
        return self._glitch_frequency_domain_source_model

    @glitch_frequency_domain_source_model.setter
    def glitch_frequency_domain_source_model(self, glitch_frequency_domain_source_model):
        if glitch_frequency_domain_source_model == "None":
            self.glitch_frequency_domain_source_model = None
        elif isinstance(glitch_frequency_domain_source_model, str):
            self._glitch_frequency_domain_source_model = \
                get_function_from_string_path(
                    glitch_frequency_domain_source_model
                )
        else:
            self._glitch_frequency_domain_source_model = \
                glitch_frequency_domain_source_model

    @property
    def injection_glitch_frequency_domain_source_model(self):
        return self._injection_glitch_frequency_domain_source_model

    @injection_glitch_frequency_domain_source_model.setter
    def injection_glitch_frequency_domain_source_model(
        self, injection_glitch_frequency_domain_source_model
    ):
        if injection_glitch_frequency_domain_source_model is None:
            self._injection_glitch_frequency_domain_source_model = (
                self.glitch_frequency_domain_source_model
            )
        elif injection_glitch_frequency_domain_source_model == "None":
            self._injection_glitch_frequency_domain_source_model = (
                self.glitch_frequency_domain_source_model
            )
        else:
            self._injection_glitch_frequency_domain_source_model = (
                get_function_from_string_path(
                    injection_glitch_frequency_domain_source_model
                )
            )

    @property
    def detectors_glitch(self):
        """The glitch detectors"""
        return self._detectors_glitch

    @detectors_glitch.setter
    def detectors_glitch(self, detectors_glitch):
        if detectors_glitch is None:
            self._detectors_glitch = self.detectors
            return
            
        from bilby_pipe.utils import convert_detectors_input
        self._detectors_glitch = convert_detectors_input(detectors_glitch)
            
    def _get_priors(self, **kwargs):
        """Transform non-detector specific glitch priors to detector specific ones"""
        priors = super()._get_priors(**kwargs)
        if self.glitch_frequency_domain_source_model is None:
            return priors
            
        glitch_params = list(
            inspect.signature(
                self.glitch_frequency_domain_source_model
            ).parameters.keys()
        )
        log_params = [f"log_{p}" for p in glitch_params]
        glitch_params.extend(log_params)
        ifos = getattr(self, "interferometers", None)
        if ifos is None:
            return priors
            
        ifo_names = [ifo.name for ifo in ifos if ifo.name in self.detectors_glitch]
        keys_to_remove = []
        new_priors = {}
        for key, prior in priors.items():
            if key in glitch_params:
                for ifo in ifo_names:
                    ifo_key = f"{ifo}_{key}"
                    if ifo_key not in priors:
                        new_prior = copy.deepcopy(prior)
                        new_prior.name = ifo_key
                        new_priors[ifo_key] = new_prior
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del priors[key]
        for key, prior in new_priors.items():
            priors[key] = prior

        self._glitch_params = keys_to_remove
        return priors

    @property
    def likelihood(self):
        """ The likelihood function """
        self.search_priors = self.priors.copy()
        if self.glitch_frequency_domain_source_model is not None:
            from .likelihood import GravitationalWaveTransientPlusGlitch, Glitch
            
            if self.likelihood_type == "GravitationalWaveTransientPlusGlitch":
                LikelihoodClass = GravitationalWaveTransientPlusGlitch
            elif self.likelihood_type == "Glitch":
                LikelihoodClass = Glitch
            else:
                raise BilbyPipeError(
                    f"A glitch-frequency-domain-source-model is provided, "
                    f"but likelihood-type '{self.likelihood_type}' is not a "
                    f"glitch likelihood. Please use 'Glitch' or "
                    f"'GravitationalWaveTransientPlusGlitch'."
                )
            
            self._likelihood = LikelihoodClass(
                interferometers=self.interferometers,
                waveform_generator=self.waveform_generator,
                priors=self.priors,
                phase_marginalization=self.phase_marginalization,
                distance_marginalization=self.distance_marginalization,
                time_marginalization=self.time_marginalization,
                reference_frame=self.reference_frame,
                time_reference=self.time_reference,
                glitch_model=self.glitch_frequency_domain_source_model,
                detectors_glitch=self.detectors_glitch
            )
            return self._likelihood
        else:
            return super().likelihood


class MainInput(Input, _MainInput):
    """An object to hold all the inputs to bilby_pipe. Inherited from
    bilby_pipe.main.MainInput
    """
    def __init__(self, *args, **kwargs):
        self.glitch_frequency_domain_source_model = args[0].glitch_frequency_domain_source_model
        self.injection_glitch_frequency_domain_source_model = args[0].injection_glitch_frequency_domain_source_model
        self.detectors_glitch = args[0].detectors_glitch
        super().__init__(*args, **kwargs)


class DataGenerationInput(Input, _DGInput):
    def __init__(self, *args, **kwargs):
        self.glitch_frequency_domain_source_model = args[0].glitch_frequency_domain_source_model
        self.injection_glitch_frequency_domain_source_model = args[0].injection_glitch_frequency_domain_source_model
        self.detectors_glitch = args[0].detectors_glitch
        super().__init__(*args, **kwargs)

    def _set_interferometers_from_injection_in_gaussian_noise(self):
        """Extended method to additionally inject a glitch"""
        self.injection_parameters = self.injection_df.iloc[self.idx].to_dict()
        cbc_injection_parameters = {}
        glitch_injection_parameters = {}
        glitch_params = list(
            inspect.signature(
                self.injection_glitch_frequency_domain_source_model
            ).parameters.keys()
        )
        for key, item in self.injection_parameters.items():
            if key in glitch_params or any(f"_{key}" in p for p in glitch_params):
                glitch_injection_parameters[key] = item
            else:
                cbc_injection_parameters[key] = item
        self.injection_parameters = cbc_injection_parameters
        injection_df_copy = self.injection_df.copy()
        self.injection_df = self.injection_df[cbc_injection_parameters.keys()]
        super()._set_interferometers_from_injection_in_gaussian_noise()
        self.injection_df = injection_df_copy
        for ifo in self.interferometers:
            if ifo.name in self.detectors_glitch:
                _ifo_glitch_params = {}
                for param in glitch_params:
                    if f"{ifo}_{param}" in glitch_injection_parameters:
                        _ifo_glitch_params[param] = glitch_injection_parameters[
                            f"{ifo}_{param}"
                        ]
                    elif param in glitch_injection_parameters:
                        _ifo_glitch_params[param] = glitch_injection_parameters[
                            param
                        ]

                logger.info("Injected glitch in {}:".format(ifo.name))
                for key in _ifo_glitch_params:
                    logger.info('  {} = {}'.format(key, _ifo_glitch_params[key]))

                _ifo_glitch_params["tstart"] = ifo.start_time
                _ifo_glitch_params["frequency_array"] = ifo.frequency_array
                _ifo_glitch_params["psd"] = ifo.power_spectral_density_array
                glitch = self.injection_glitch_frequency_domain_source_model(
                    **_ifo_glitch_params
                )
                ifo.strain_data.frequency_domain_strain += glitch


class DataAnalysisInput(Input, _DAInput):
    """Handles user-input for the data analysis script. Inherited from
    bilby_pipe.data_analysis.DataAnalysisInput
    """
    def __init__(self, *args, **kwargs):
        self.glitch_frequency_domain_source_model = args[0].glitch_frequency_domain_source_model
        self.injection_glitch_frequency_domain_source_model = args[0].injection_glitch_frequency_domain_source_model
        self.detectors_glitch = args[0].detectors_glitch
        super().__init__(*args, **kwargs)


def create_parser(top_level=False):
    """Extends the BilbyArgParser for bilby_pipe to include additional
    options for sampling over multiple models

    Parameters
    ----------
    top_level:
        If true, parser is to be used at the top-level with requirement
        checking etc, else it is an internal call and will be ignored.
    """
    from bilby_pipe.parser import create_parser
    parser = create_parser(top_level=top_level)
    parser.set_defaults(
        main_input_class="bilby_antiglitch.bilby_pipe.MainInput",
        analysis_input_class="bilby_antiglitch.bilby_pipe.DataAnalysisInput",
        generation_executable_parser="bilby_antiglitch.bilby_pipe.create_parser",
        generation_input_class="bilby_antiglitch.bilby_pipe.DataGenerationInput"
    )
    glitch_group = parser.add_argument_group("Glitch models")
    glitch_group.add_argument(
        "--glitch-frequency-domain-source-model",
        type=str,
        default=None,
        help="Name of the frequency domain source model. Can be one of"
        " [antiglitch] or the python path to a source equation e.g."
        " bilby_antiglitch.source.antiglitch",
    )
    glitch_group.add_argument(
        "--injection-glitch-frequency-domain-source-model",
        type=str,
        default=None,
        help="Name of the frequency domain source model. Can be one of"
        " [antiglitch] or the python path to a source equation e.g."
        " bilby_antiglitch.source.antiglitch",
    )
    glitch_group.add_argument(
        "--detectors-glitch",
        action="append",
        help="The names of detectors to apply the glitch model to. If given in"
        " the ini file, detectors are specified by `detectors-glitch=[H1, L1]`."
        " If given at the command line, as `--detectors-glitch H1"
        " --detectors-glitch L1`. Defaults to the value of `--detectors`.",
    )
    return parser
