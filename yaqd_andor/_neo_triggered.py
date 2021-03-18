__all__ = ["NeoTriggered"]

import asyncio
import numpy as np

from yaqd_core import IsDaemon, IsSensor, HasMeasureTrigger, HasMapping
from typing import Dict, Any, List
from atcore import ATCore, ATCoreException


class NeoTriggered(HasMapping, HasMeasureTrigger, IsSensor, IsDaemon):
    _kind = "neo-triggered"

    def __init__(self, name, config, config_filepath):
        super().__init__(name, config, config_filepath)
        self.sdk3 = ATCore() # Initialise SDK3
        # deviceCount = sdk3.get_int(sdk3.AT_HNDL_SYSTEM,"DeviceCount")
        self.hndl = self.sdk3.open(0)
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

