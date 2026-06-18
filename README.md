# bilby glitch

[![PyPI version](https://img.shields.io/pypi/v/bilby_glitch)](https://img.shields.io/pypi/v/bilby_glitch)

[![Coverage report](https://hoyc1.github.io/bilby_glitch/coverage-badge.svg)](https://hoyc1.github.io/bilby_glitch/coverage.xml) [![Pipeline Status](https://github.com/hoyc1/bilby_glitch/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/hoyc1/bilby_glitch/actions/workflows/test.yml)

This Python package extends the functionality in the bilby to additional sample over glitch realisations

## Installation

`bilby_glitch` is currently available via PyPI and can be installed with:

```bash
$ pip install bilby_glitch
```

Once `bilby_glitch` has been installed, a custom version of `bilby_pipe` needs to
be installed with:

```bash
$ pip install 'bilby_pipe @ git+https://git.ligo.org/charlie.hoy/bilby_pipe.git@input_class'
```

This version needs to be installed because we are waiting for required code to
be merged into the main `bilby_pipe` code base. Please see the following merge
request for details:

    * `bilby_pipe!583 <https://git.ligo.org/lscsoft/bilby_pipe/-/merge_requests/583>`_

For full installation instructions, see [our documentation](https://hoyc1.github.io/bilby_glitch/installation.html).

## Usage in bilby_pipe

The functionality in `bilby_glitch` can be used with `bilby_pipe` as you would with any other frequency domain source model. It simply requires the following options to be specified in your configuration file:

```ini
analysis_executable_parser=bilby_glitch.bilby_pipe.create_parser
likelihood-type=GravitationalWaveTransientPlusGlitch
glitch-frequency-domain-source-model = bilby_glitch.source.antiglitch
detectors-glitch = ['L1']
```

## Citing

If you find `bilby_glitch` useful in your work please cite the following papers:

```bibtex

```
