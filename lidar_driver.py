import serial
import time

class TF02Pro:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.connected = False
        if self.ser.is_open:
            self.connected = True
            print(f"LiDAR connected on {port}")

    def read_data(self):
        """
        Reads a single frame. Returns distance (cm) and strength.
        Protocol: 0x59 0x59 Dist_L Dist_H Strength_L Strength_H Temp_L Temp_H Checksum
        """
        while True:
            if self.ser.in_waiting >= 9:
                if self.ser.read(1) == b'\x59':
                    if self.ser.read(1) == b'\x59':
                        data = self.ser.read(7)
                        if len(data) == 7:
                            # Parse Data
                            dist = data[0] + (data[1] << 8)
                            strength = data[2] + (data[3] << 8)
                            # Checksum validation (optional but recommended)
                            checksum = 0x59 + 0x59 + sum(data[:6])
                            if (checksum & 0xFF) == data[6]:
                                return dist, strength
        return None, None

    def close(self):
        self.ser.close()