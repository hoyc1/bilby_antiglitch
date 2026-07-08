# Licensed under an MIT style license -- see LICENSE.md

import inspect
import copy
import numpy as np
from bilby.core.likelihood import _fallback_to_parameters
from bilby.gw.likelihood import GravitationalWaveTransient

__author__ = ["Charlie Hoy <charlie.hoy@port.ac.uk>"]


class Glitch(GravitationalWaveTransient):
    """Likelihood for a Glitch model.
    """
    def __init__(
        self, *args, glitch_model=None, time_marginalization=False,
        distance_marginalization=False, phase_marginalization=False,
        detectors_glitch=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if time_marginalization:
            raise ValueError(
                "Unable to use time marginalization in this likelihood"
            )
        if distance_marginalization:
            raise ValueError(
                "Unable to use distance marginalization in this likelihood"
            )
        if phase_marginalization:
            raise ValueError(
                "Unable to use phase marginalization in this likelihood"
            )
        if glitch_model is None:
            glitch_model = self._default_glitch_model

        self._glitch_model = glitch_model
        self._glitch_params = list(
            inspect.signature(glitch_model).parameters.keys()
        )
        if "frequency_array" in self._glitch_params:
            self._glitch_params.remove("frequency_array")
        if "psd" in self._glitch_params:
            self._glitch_params.remove("psd")
        if "tstart" in self._glitch_params:
            self._glitch_params.remove("tstart")
        if "kwargs" in self._glitch_params:
            self._glitch_params.remove("kwargs")

        if detectors_glitch is None:
            self.detectors_glitch = [ifo.name for ifo in self.interferometers]
        else:
            self.detectors_glitch = detectors_glitch

    @property
    def _default_glitch_model(self):
        from .source import antiglitch
        return antiglitch

    def _get_ifo_glitch_params(self, interferometer, parameters):
        """
        Extract glitch parameters for a specific interferometer.
        It looks for `<ifo>_<param>`, if not found, it falls back to `<param>`.
        """
        ifo = interferometer.name
        glitch_params = {}
        for param in self._glitch_params:
            ifo_param = f"{ifo}_{param}"
            if ifo_param in parameters:
                glitch_params[param] = parameters[ifo_param]
            elif param in parameters:
                glitch_params[param] = parameters[param]
                
        ifo_log_gamma = f"{ifo}_log_gamma"
        if ifo_log_gamma in parameters:
            glitch_params["log_gamma"] = parameters[ifo_log_gamma]
        elif "log_gamma" in parameters:
            glitch_params["log_gamma"] = parameters["log_gamma"]
            
        glitch_params["tstart"] = self.waveform_generator.start_time
        glitch_params["frequency_array"] = self.waveform_generator.frequency_array
        glitch_params["psd"] = interferometer.power_spectral_density_array
        return glitch_params

    def log_likelihood_ratio(self, parameters=None):
        if parameters is not None:
            parameters = copy.deepcopy(parameters)
        else:
            parameters = _fallback_to_parameters(self, parameters)

        total_snrs = self._CalculatedSNRs()
        for interferometer in self.interferometers:
            glitch_params = self._get_ifo_glitch_params(
                interferometer, parameters
            )
            required_params = ["A", "f", "phi", "tc"]
            cond1 = all(p in glitch_params for p in required_params)
            cond2 = ("gamma" in glitch_params or "log_gamma" in glitch_params)
            if interferometer.name in self.detectors_glitch and cond1 and cond2:
                h = self._glitch_model(**glitch_params)
            else:
                h = np.zeros_like(glitch_params["frequency_array"])
        
            per_detector_snr = self.calculate_snrs(
                glitch_strain=h,
                interferometer=interferometer,
                parameters=parameters,
            )
            total_snrs += per_detector_snr

        log_l = self.compute_log_likelihood_from_snrs(
            total_snrs, parameters=parameters
        )
        return float(log_l.real)

    def calculate_snrs(
        self, glitch_strain, interferometer, return_array=True, parameters=None
    ):
        parameters = _fallback_to_parameters(self, parameters)
        signal = glitch_strain
        _mask = interferometer.frequency_mask
        if 'recalib_index' in parameters:
            signal[_mask] *= self.calibration_draws[interferometer.name][
                int(parameters['recalib_index'])
            ]

        d_inner_h = interferometer.inner_product(signal=signal)
        optimal_snr_squared = interferometer.optimal_snr_squared(signal=signal)
        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared**0.5)

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h,
            optimal_snr_squared=optimal_snr_squared.real,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_array=None,
            optimal_snr_squared_array=None,
        )

    def compute_log_likelihood_from_snrs(self, snrs, parameters=None):
        log_l = np.real(snrs.d_inner_h) - snrs.optimal_snr_squared / 2
        return float(log_l.real)


class GravitationalWaveTransientPlusGlitch(Glitch, GravitationalWaveTransient):
    """Likelihood for CBC + Glitch model.
    """
    def log_likelihood_ratio(self, parameters=None):
        if parameters is not None:
            parameters = copy.deepcopy(parameters)
        else:
            parameters = _fallback_to_parameters(self, parameters)

        cbc_parameters = {}
        for key, item in parameters.items():
            is_glitch_param = False
            for p in self._glitch_params + ["log_gamma"]:
                cond = any(
                    key == f"{ifo.name}_{p}" for ifo in self.interferometers
                )
                if key == p or cond:
                    is_glitch_param = True
                    break
            if not is_glitch_param:
                cbc_parameters[key] = item

        waveform_polarizations = \
            self.waveform_generator.frequency_domain_strain(cbc_parameters)
        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        sky_frame_params = self.get_sky_frame_parameters(cbc_parameters)
        parameters = {**parameters, **sky_frame_params}
        total_snrs = self._CalculatedSNRs()
        for interferometer in self.interferometers:
            glitch_params = self._get_ifo_glitch_params(interferometer, parameters)
            required_params = ["A", "f", "phi", "tc"]
            cond1 = all(p in glitch_params for p in required_params)
            cond2 = ("gamma" in glitch_params or "log_gamma" in glitch_params)
            if interferometer.name in self.detectors_glitch and cond1 and cond2:
                h = self._glitch_model(**glitch_params)
            else:
                h = np.zeros_like(glitch_params["frequency_array"])

            per_detector_snr = self.calculate_snrs(
                waveform_polarizations=waveform_polarizations,
                interferometer=interferometer,
                glitch_strain=h,
                parameters=parameters,
            )

            total_snrs += per_detector_snr

        log_l = self.compute_log_likelihood_from_snrs(
            total_snrs, parameters=parameters
        )

        return float(log_l.real)

    def calculate_snrs(
        self, waveform_polarizations, interferometer,
        glitch_strain=None, return_array=True, parameters=None
    ):
        parameters = _fallback_to_parameters(self, parameters)
        signal = self._compute_full_waveform(
            signal_polarizations=waveform_polarizations,
            interferometer=interferometer,
            parameters=parameters,
        )
        if glitch_strain is not None:
            signal += glitch_strain
        _mask = interferometer.frequency_mask

        if 'recalib_index' in parameters:
            signal[_mask] *= self.calibration_draws[interferometer.name][
                int(parameters['recalib_index'])
            ]

        d_inner_h = interferometer.inner_product(signal=signal)
        optimal_snr_squared = interferometer.optimal_snr_squared(signal=signal)
        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared**0.5)

        d_inner_h_array = None
        optimal_snr_squared_array = None

        normalization = 4 / self.waveform_generator.duration

        if return_array is False:
            d_inner_h_array = None
            optimal_snr_squared_array = None
        elif self.calibration_marginalization and ('recalib_index' not in parameters):
            d_inner_h_integrand = (
                normalization *
                interferometer.frequency_domain_strain.conjugate() * signal
                / interferometer.power_spectral_density_array
            )
            d_inner_h_array = np.dot(
                d_inner_h_integrand[_mask],
                self.calibration_draws[interferometer.name].T
            )
            optimal_snr_squared_integrand = (
                normalization * np.abs(signal)**2 /
                interferometer.power_spectral_density_array
            )
            optimal_snr_squared_array = np.dot(
                optimal_snr_squared_integrand[_mask],
                self.calibration_abs_draws[interferometer.name].T
            )

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h,
            optimal_snr_squared=optimal_snr_squared.real,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_array=d_inner_h_array,
            optimal_snr_squared_array=optimal_snr_squared_array,
        )
