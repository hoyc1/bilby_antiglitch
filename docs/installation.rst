Installation instructions
=========================

:code:`bilby_antiglitch` can be installed through a variety of methods, see below.
Independent of the method chosen, we recommend installing :code:`bilby_antiglitch`
within a conda environment. For speed, we recommend creating an environment with
`mamba <https://mamba.readthedocs.io/en/latest/>`_. :code:`bilby_antiglitch` can
be installed with,

.. tabs::

    .. tab:: mamba and pip

        .. code-block::

             $ mamba env create --name bilby-antiglitch python=3.11
             $ mamba activate bilby-antiglitch
             $ python -m pip install bilby_antiglitch
             $ python -m pip install 'bilby_pipe @ git+https://git.ligo.org/charlie.hoy/bilby_pipe.git@input_class'

        .. warning::

            As part of this installation, a non-released version of
            :code:`bilby_pipe` is installed. This is because we are waiting
            for required code to be merged into the main :code:`bilby_pipe`
            code base. Please see the following merge request for details:

                * `bilby_pipe!583 <https://git.ligo.org/lscsoft/bilby_pipe/-/merge_requests/583>`_

            The non-released version :code:`bilby_pipe` is rebased onto the
            following tag:

                * :code:`bilby_pipe`: `v1.8.0 <https://git.ligo.org/lscsoft/bilby_pipe/-/tags/v1.8.0>`_

    .. tab:: From source

        If installing from source, an enviroment must first be created. An
        environment with all required dependencies can be created with
        :code:`mamba` by running,

        .. code-block:: console

            $ mamba create --name bilby-antiglitch python=3.11

        ``bilby_antiglitch`` can then be installed with,

        .. code-block::

            $ git clone git@github.com:hoyc1/bilby_antiglitch.git
            $ cd bilby_antiglitch
            $ python -m pip install .

        .. warning::

            Once ``bilby_antiglitch`` has been installed, a non-released version of
            :code:`bilby_pipe` is installed. This is because we are waiting
            for required code to be merged into the main :code:`bilby_pipe`
            code base. Please see the following merge request for details:

                * `bilby_pipe!583 <https://git.ligo.org/lscsoft/bilby_pipe/-/merge_requests/583>`_

            The non-released version :code:`bilby_pipe` is rebased onto the
            following tag:

                * :code:`bilby_pipe`: `v1.8.0 <https://git.ligo.org/lscsoft/bilby_pipe/-/tags/v1.8.0>`_
