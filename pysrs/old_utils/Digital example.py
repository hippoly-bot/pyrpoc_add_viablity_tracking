import nidaqmx
# https://forums.ni.com/t5/Digital-I-O/Multiple-Digital-Output-Channels-with-Python/td-p/4364138
# Define 2 functions. The first 'initializes' the device. The second actually aquires the data. Otherwise the data is wrong. 
# This seems to depend on the specific device and driver version? and my not be required. I am using a PXI-6508 and DAQmx 18.5 and it is required. 

def digital_task():

# Write to DO lines 
     
    with nidaqmx.Task() as task:
        task.do_channels.add_do_chan("test/port0/line0")  
        task.do_channels.add_do_chan("test/port0/line1")
        task.do_channels.add_do_chan("test/port0/line2")
        level = True
        level_0 = False
        task.write([level, level_0, level])
    
# Read DI lines 

    with nidaqmx.Task() as task:
        task.di_channels.add_di_chan("test/port1/line0")
        task.di_channels.add_di_chan("test/port1/line1")
        task.di_channels.add_di_chan("test/port1/line2")
        data = task.read()
    
    print(data)

digital_task()

# If this doesn't give the correct values you may need to define a new function and run both functions. 
# Running the same function twice may work but I have seen where it doesn't. This example seems to work fine but if you had 8 ports 8 lines each it may cause a problem. 
# time.sleep(), task.close() or task.stop() doesn't seem to fix this. At least for my device and driver version. 
# You may only need to run this once.
