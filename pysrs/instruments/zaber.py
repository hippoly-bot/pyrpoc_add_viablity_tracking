import numpy as np
from zaber_motion import Units
from zaber_motion.ascii import Connection
import concurrent.futures

class ZaberStage:
    def __init__(self, port='COM3'):
        self.port = port
        self.connection = None
        self.device = None
        self.axis = None

    def connect(self, timeout=10):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor: # failed attempt to get it to not crash if it cant connect
            future = executor.submit(self._connect)
            future.result(timeout=timeout)

    def _connect(self):
        if self.connection is not None:
            return  
        self.connection = Connection.open_serial_port(self.port)
        self.connection.enable_alerts()
        devices = self.connection.detect_devices()
        if not devices:
            raise RuntimeError("No Zaber devices found.")
        self.device = devices[0]
        self.axis = self.device.get_axis(1)
        if not self.axis.is_homed():
            print("Homing the stage...")
            self.axis.home()

    def move_absolute_um(self, position_um):
        if self.axis is None:
            self.connect()
        position_mm = position_um * 1e-3
        self.axis.move_absolute(position_mm, Units.LENGTH_MILLIMETRES)
        self.axis.wait_until_idle()

    def is_connected(self):
        if self.connection is not None:
            return True
        else: 
            return False

    def disconnect(self):
        if self.connection:
            self.connection.close()
            self.connection = None

if __name__ == '__main__':
    config = {
        'serial': 'COM3', 
        'num_shifts': 3,  
        'shift_size': 5,  # mm
        'shift_offset': 20,  # mm
    }
    stage = ZaberStage(config)
    stage.scan_range()