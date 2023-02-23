#!/usr/bin/env python
from yaqd_andor import atcore
import numpy as np


def main():
    print("Single Scan Example")

    print("Intialising SDK3")
    sdk3 = ATCore()  # Initialise SDK3
    deviceCount = sdk3.get_int(sdk3.AT_HNDL_SYSTEM, "DeviceCount")

    print("Found : ", deviceCount, " device(s)")

    if deviceCount > 0:
        try:
            print("  Opening camera ")
            hndl = sdk3.open(0)

            print("    Configuring Acquisition")
            sdk3.set_enumerated_string(
                hndl, "SimplePreAmpGainControl", "16-bit (low noise & high well capacity)"
            )

            print("    Queuing Buffer")
            imageSizeBytes = sdk3.get_int(hndl, "ImageSizeBytes")
            buf = np.empty((imageSizeBytes,), dtype="B")
            sdk3.queue_buffer(hndl, buf.ctypes.data, imageSizeBytes)

            print("    Acquiring Frame")
            sdk3.command(hndl, "AcquisitionStart")
            (returnedBuf, returnedSize) = sdk3.wait_buffer(hndl)

            print("    Frame Returned, first 10 pixels")
            pixels = buf.view(dtype="H")
            for i in range(0, 10):
                print("      Pixel ", i, " value ", pixels[i])

            sdk3.command(hndl, "AcquisitionStop")

        except ATCoreException as err:
            print("     SDK3 Error {0}".format(err))
        print("  Closing camera")
        sdk3.close(hndl)
    else:
        print("Could not connect to camera")


main()
