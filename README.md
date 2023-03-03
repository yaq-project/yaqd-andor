# yaqd-andor

[![PyPI](https://img.shields.io/pypi/v/yaqd-andor)](https://pypi.org/project/yaqd-andor)
[![Conda](https://img.shields.io/conda/vn/conda-forge/yaqd-andor)](https://anaconda.org/conda-forge/yaqd-andor)
[![yaq](https://img.shields.io/badge/framework-yaq-orange)](https://yaq.fyi/)
[![black](https://img.shields.io/badge/code--style-black-black)](https://black.readthedocs.io/)
[![ver](https://img.shields.io/badge/calver-YYYY.0M.MICRO-blue)](https://calver.org/)
[![log](https://img.shields.io/badge/change-log-informational)](https://github.com/yaq-project/yaqd-andor)

Daemons for [Andor](https://andor.oxinst.com/) cameras.

This package contains the following daemon(s):

- https://yaq.fyi/daemons/andor-neo
- https://yaq.fyi/daemons/andor-simcam
- https://yaq.fyi/daemons/andor-sona
- https://yaq.fyi/daemons/andorsdk2-ixon


# Installation Details
## Daemons using Andor SDK3

The following daemons use Andor Software Development Kit v3 (SDK3):
* `andor-neo`
* `andor-sona`
* `andor-simcam`

To run, you must also have access to the SKD3 driver files (not provided here):

- atcore.h
- atcore.lib
- atblklx.dll
- actl_bitflow.dll
- atcore.dll
- atdevregcam.dll (andor-neo daemon)
- atdevsimcam.dll (andor-simcam daemon)
- atusb_libusb10.dll

Place these files in the `yaqd-andor` package source directory.


### Simcam needs Visual Studio
`andor-simcam` uses Andor's virtual camera, which is useful for remote development.
For Windows, these libraries require [Microsoft Build tools for Visual Studio](https://visualstudio.microsoft.com/downloads/) (specifically, the Windows SDK and C++ x64/x86 build tools are needed).

## Daemons using Andor SDK2

The following daemons use Andor Software Development Kit v2 (SDK2)
* `andorsdk2-ixon`

To run, you must also have access to the SKD2 driver files:

`/usr/local/lib/libandor.so` (Linux) OR  `atmcd64d.dll` /`atmcd32d.dll`  (Windows (64/32-bit))

## maintainers
[Dan Kohler](https://github.com/ddkohler)

