__all__ = ["NeoTriggered"]

import asyncio
import numpy as np

from yaqd_core import IsDaemon, IsSensor, HasMeasureTrigger, HasMapping
from typing import Dict, Any, List
from . import atcore 

import os
os.chdir(os.path.dirname(__file__))

ATCore = atcore.ATCore
ATCoreException = atcore.ATCoreException


class NeoTriggered(HasMapping, HasMeasureTrigger, IsSensor, IsDaemon):
    _kind = "neo-triggered"

    def __init__(self, name, config, config_filepath):
        super().__init__(name, config, config_filepath)
        print("Intialising SDK3")
        import os
        print(os.getcwd())
        self.sdk3 = ATCore() # Initialise SDK3
        device_count = self.sdk3.get_int(self.sdk3.AT_HNDL_SYSTEM, "DeviceCount")
        i = 0
        while i < device_count:
            temp = self.sdk3.open(i)
            serial = self.sdk3.get_string(temp, "SerialNumber")
            if serial == self._config["serial_number"]:
                self.hndl = temp
                print("    Serial No   : ",serial)
                break
        else:
            print("no devices found")
        

        self.hndl = self.sdk3.open(0)
        if False:
            self.sdk3.set_enumerated_string(
                self.hndl,
                "SimplePreAmpGainControl",
                "16-bit (low noise & high well capacity)"
            )
            # set trigger mode to software
            self.sdk3.set_enumerated_string(
                self.hndl, "TriggerMode", "Software"
            )
            # set cycle mode to fixed
            self.sdk3.set_enumerated_string(
                self.hndl, "CycleMode", "Fixed"
            )
            # enable metadata

    async def _measure(self):
        # queue buffer
        imageSizeBytes = self.sdk3.get_int(self.hndl, "ImageSizeBytes")
        buf = np.empty((imageSizeBytes,), dtype='B')
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

    def close(self):
        self.sdk3.close(self.hndl)
        # ddk: need to call sdk3.__del__ ever?

