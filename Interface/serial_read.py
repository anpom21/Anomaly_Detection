import serial
import time


def readserial(ser):

    # 1/timeout is the frequency at which the port is read

    data = ser.readline().decode().strip()
    timestamp = time.strftime('%H:%M:%S')
    if data:
        return data, timestamp
    else:
        return False, timestamp


if __name__ == '__main__':

    comport = 'COM3'
    baudrate = 9600
    ser = serial.Serial(comport, baudrate, timeout=0.1)
    while True:
        data, timestamp = readserial(ser)
        if data:
            print(f'{timestamp} > {data}')
