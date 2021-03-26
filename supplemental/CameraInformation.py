#!/usr/bin/env python
import yaqd_andor
atcore = yaqd_andor.atcore

print("Camera Information Example")
print("Intialising SDK3")

sdk3 = atcore.ATCore() # Initialise SDK3
deviceCount = sdk3.get_int(sdk3.AT_HNDL_SYSTEM,"DeviceCount")

print("Found : ",deviceCount," device(s)")

for i in range(deviceCount):
    try:
        print("  Opening device ",(i+1))
        hndl = sdk3.open(i)
        
        serial = sdk3.get_string(hndl,"SerialNumber")
        print("    Serial No   : ",serial)

        model = sdk3.get_string(hndl,"CameraModel")
        print("    Model No    : ",model)

    except atcore.ATCoreException as err :
        print("     SDK3 Error {0}".format(err))
    print("  Closing device ",(i+1))
    sdk3.close(hndl)
