__all__ = ["NeoTriggered"]

import asyncio
from re import L
import numpy as np

from yaqd_core import IsDaemon, IsSensor, HasMeasureTrigger, HasMapping
from typing import Dict, Any, List
from . import atcore 
from . import features
from collections import namedtuple

import os
os.chdir(os.path.dirname(__file__))

ATCore = atcore.ATCore
ATCoreException = atcore.ATCoreException


class NeoTriggered(HasMapping, HasMeasureTrigger, IsSensor, IsDaemon):
    _kind = "neo-triggered"

    def __init__(self, name, config, config_filepath):
        super().__init__(name, config, config_filepath)
        self.sdk3 = ATCore() # Initialise SDK3
        # find devices
        device_count = self.sdk3.get_int(self.sdk3.AT_HNDL_SYSTEM, "DeviceCount")
        if device_count == 0:
            raise ConnectionError("No devices found.")
        i = 0
        # select device
        while i < device_count:
            temp = self.sdk3.open(i)
            serial = self.sdk3.get_string(temp, "SerialNumber")
            if serial == self._config["serial_number"]:
                self.hndl = temp
                print("    Serial No   : ",serial)
                break
            i += 1
        else:
            raise ConnectionError(
                r"device with serial number {0} not found".format(self._config["serial_number"])
            )

        if not self._config["is_virtual"]:
            self.sdk3.set_enumerated_string(
                self.hndl,
                "SimplePreAmpGainControl",
                "16-bit (low noise & high well capacity)"
            )

        # set trigger mode to software
        self.sdk3.set_enumerated_string(
            self.hndl, "TriggerMode", "Advanced" if self._config["is_virtual"] else "Software"
        )
        # set cycle mode to fixed
        self.sdk3.set_enumerated_string(
            self.hndl, "CycleMode", "Fixed"
        )
        # enable metadata
        self.sdk3.set_enumerated_string(
            self.hndl, "CycleMode", "Fixed"
        )
        self.sdk3.set_bool(self.hndl, "MetadataEnable", True)

    async def _measure(self):
        # queue buffer
        imageSizeBytes = self.sdk3.get_int(self.hndl, "ImageSizeBytes")
        buf = np.empty((imageSizeBytes,), dtype='B')
        # todo: queue buffer for n frames
        self.sdk3.queue_buffer(self.hndl, buf.ctypes.data, imageSizeBytes)
        # acquire frame
        # self.sdk3.command(self.hndl, "AcquisitionStart")
        self.sdk3.command(self.hndl, "SoftwareTrigger")
        (returnedBuf, returnedSize) = await self._loop.run_in_executor(
            None, self.sdk3.wait_buffer(self.hndl)
        )
        # parse readout
        pixels = buf.view(dtype='H')
        # self.sdk3.command(self.hndl,"AcquisitionStop")

        return {"image": pixels}

    def get_sensor_info(self):
        pass

    def interrupt(self):
        """stop measure before exposure time is reached.
        """
        pass

    def close(self):
        self.sdk3.close(self.hndl)

    def _get_feature(self, feature):
        call = type_to_call["int"].format("get")
        if self.sdk3.is_implemented(self.hndl, call):
            if self.sdk3.is_readable(self.hndl, call):
                return self.sdk3.__getattribute__(call)(self.hndl, feature.name)
            else:
                raise TypeError(f"call {call} is not readable")
        else:
            raise ValueError(f"call {call} is not implemented")

    def _set_feature(self, feature):
        call = type_to_call["int"].format("set")
        if self.sdk3.is_readonly(self.hndl, call):
            return TypeError(f"Cannot write.  Feature {call} is read only")
        if self.sdk3.is_implemented(self.hndl, call):
            while not self.sdk3.is_writable(self.hndl, call):
                return self.sdk3.__getattribute__(call)(self.hndl, feature.name, feature.value)
        else:
            raise ValueError(f"call {call} is not implemented")




