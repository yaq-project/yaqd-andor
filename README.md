# yaqd-andor

[![PyPI](https://img.shields.io/pypi/v/yaqd-andor)](https://pypi.org/project/yaqd-andor)
[![Conda](https://img.shields.io/conda/vn/conda-forge/yaqd-andor)](https://anaconda.org/conda-forge/yaqd-andor)
[![yaq](https://img.shields.io/badge/framework-yaq-orange)](https://yaq.fyi/)
[![black](https://img.shields.io/badge/code--style-black-black)](https://black.readthedocs.io/)
[![ver](https://img.shields.io/badge/calver-YYYY.0M.MICRO-blue)](https://calver.org/)
[![log](https://img.shields.io/badge/change-log-informational)](https://github.com/yaq-project/yaqd-andor)

Daemons for [Andor](https://andor.oxinst.com/) cameras

This package contains the following daemon(s):

- https://yaq.fyi/daemons/andor-neo
- https://yaq.fyi/daemons/andor-simcam

To run these daemons, you must also have access to the ANDOR sdk3 driver files:

- atcore.h
- atcore.lib
- atblklx.dll
- actl_bitflow.dll
- atcore.dll
- atdevregcam.dll (andor-neo daemon)
- atdevsimcam.dll (andor-simcam daemon)
- atusb_libusb10.dll

`andor-simcam` uses Andor's virtual camera, which is useful for remote development.
For Windows, these libraries require [Microsoft Build tools for Visual Studio](https://visualstudio.microsoft.com/downloads/) (specifically, the Windows SDK and C++ x64/x86 build tools are needed).

## maintainers

[Dan Kohler](https://github.com/ddkohler)
