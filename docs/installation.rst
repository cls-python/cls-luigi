Installation
============


Stable release
--------------

To install CLS, run this command in your terminal:

.. code-block:: console

    $ pip install cls_luigi

This is the preferred method to install CLS-Luigi, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for CLS-Luigi can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone https://github.com/cls-python/cls-luigi.git

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/cls-python/cls-luigi/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install

CLS-Luigi on Windows
--------------------

In its current state, CLS-Luigi has not yet been tested to run fully on Windows, as it has been developed mainly on Linux.

There will undoubtedly be restrictions since `Luigi`_ itself already defines restrictions for use under Windows.
To learn more, please read the luigi documentation [`Luigi on Windows`_] on the subject.

There are also general limitations due to the usage of the `multiprocessing package`_  under Windows, which is used by both `Luigi`_ and `CLS-Python`_.
For more information refere to the multiprocessing documentation [`docs-multiprocessing`_, `contexts-and-start-methods`_] on the subject.

CLS-Luigi in a Jupyter Notebook
-------------------------------

At the time of writing, this hasn't been truly tested yet.


.. _Github repo: https://github.com/cls-python/cls-luigi
.. _tarball: https://github.com/cls-python/cls-luigi/tarball/master
.. _multiprocessing package: https://docs.python.org/3/library/multiprocessing.html
.. _Luigi on Windows: https://luigi.readthedocs.io/en/stable/running_luigi.html?highlight=windows#luigi-on-windows
.. _Luigi: https://luigi.readthedocs.io
.. _CLS-Python: https://cls-python.github.io/cls-python/readme.html
.. _contexts-and-start-methods: https://docs.python.org/dev/library/multiprocessing.html#contexts-and-start-methods
.. _docs-multiprocessing: https://docs.python.org/3/library/multiprocessing.html
