# bilby glitch

[![PyPI version](https://img.shields.io/pypi/v/bilby_antiglitch)](https://img.shields.io/pypi/v/bilby_antiglitch)

[![Coverage report](https://hoyc1.github.io/bilby_antiglitch/coverage-badge.svg)](https://hoyc1.github.io/bilby_antiglitch/coverage.xml) [![Pipeline Status](https://github.com/hoyc1/bilby_antiglitch/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/hoyc1/bilby_antiglitch/actions/workflows/test.yml)

This Python package extends the functionality in the bilby to additionally sample over glitch realisations

## Installation

`bilby_antiglitch` is currently available via PyPI and can be installed with:

```bash
$ pip install bilby_antiglitch
```

Once `bilby_antiglitch` has been installed, a custom version of `bilby_pipe` needs to
be installed with:

```bash
$ pip install 'bilby_pipe @ git+https://git.ligo.org/charlie.hoy/bilby_pipe.git@input_class'
```

This version needs to be installed because we are waiting for required code to
be merged into the main `bilby_pipe` code base. Please see the following merge
request for details:

    * `bilby_pipe!583 <https://git.ligo.org/lscsoft/bilby_pipe/-/merge_requests/583>`_

For full installation instructions, see [our documentation](https://hoyc1.github.io/bilby_antiglitch/installation.html).

## Usage in bilby_pipe

The functionality in `bilby_antiglitch` can be used with `bilby_pipe` as you would with any other frequency domain source model. It simply requires the following options to be specified in your configuration file:

```ini
analysis_executable_parser=bilby_antiglitch.bilby_pipe.create_parser
likelihood-type=GravitationalWaveTransientPlusGlitch
glitch-frequency-domain-source-model = bilby_antiglitch.source.antiglitch
detectors-glitch = ['L1']
```

If you wish to perform an injection with a specific glitch model, this can be
done with the following option in your configuration file:

```ini
injection-glitch-frequency-domain-source-model = bilby_antiglitch.source.antiglitch
```

## Citing

If you find `bilby_antiglitch` useful in your work please cite the following papers:

```bibtex

```
