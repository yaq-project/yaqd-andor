# yaqd-andor

[![PyPI](https://img.shields.io/pypi/v/yaqd-andor)](https://pypi.org/project/yaqd-andor)
[![Conda](https://img.shields.io/conda/vn/conda-forge/yaqd-andor)](https://anaconda.org/conda-forge/yaqd-andor)
[![yaq](https://img.shields.io/badge/framework-yaq-orange)](https://yaq.fyi/)
[![black](https://img.shields.io/badge/code--style-black-black)](https://black.readthedocs.io/)
[![ver](https://img.shields.io/badge/calver-YYYY.0M.MICRO-blue)](https://calver.org/)
[![log](https://img.shields.io/badge/change-log-informational)](https://gitlab.com/yaq/yaqd-andor/-/blob/master/CHANGELOG.md)

Daemons for [Andor](https://andor.oxinst.com/?gclid=CjwKCAiA4rGCBhAQEiwAelVtiwSRE1kz3nD1g-x6c1ni5svwLkqg7OMvJE5n0CIB8shS2Nnnvrgy4BoCdJcQAvD_BwE) cameras

This package contains the following daemon(s):

- https://yaq.fyi/daemons/neo-triggered

To run this daemon, you must also have access to the ANDOR sdk3 driver files:

- atcore.h
- atcore.lib
- atblklx.dll
- actl_bitflow.dll
- atcore.dll
- atdevregcam.dll
- atdevsimcam.dll (optional)
- atusb_libusb10.dll

atdevsimcam.dll can allow use of Andor's virtual camera, which is useful for remote development.  
For Windows, these libraries require [Microsoft Build tools for Visual Studio](https://visualstudio.microsoft.com/downloads/) (specifically, the Windows SDK and C++ x64/x86 build tools are needed).

