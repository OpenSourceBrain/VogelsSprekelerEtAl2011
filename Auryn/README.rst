Vogels et al (2011) - Auryn implementation
------------------------------------------

This implementation uses Auryn_ which is a new neural simulator written in C++. 

Compiling the code
==================
In order to compile the vogels.cpp file, you will need to have Auryn built along with an open MPI implementation of your choice. Please modify the Makefile to point to the location of the Auryn installation. An autotoolised version of Auryn is available here_.

Inputs
======
The program takes "pattern" and "stimulus" files as inputs. These are also provided here. The pattern files are a list of neurons that make up a pattern. The stimulus files are a subset of the patterns that need to be stimulated to recall the pattern.

.. _Auryn: http://www.fzenke.net/auryn/doku.php
.. _here: https://github.com/sanjayankur31/auryn/tree/autotoolize

