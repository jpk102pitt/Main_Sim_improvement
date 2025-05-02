from numpy.linalg.lapack_lite import zgelsd

from bus import Bus

class Generator:

    def __init__(self, name: str, bus: Bus, voltage_setpoint: float, mw_setpoint: float, x1 , x2, x0, zg ):
        self.name = name
        self.bus = bus
        self.voltage_setpoint = voltage_setpoint
        self.mw_setpoint = mw_setpoint
        self.x1 = x1
        self.x2 = x2
        self.x0 = x0
        self.zg = zg
        self.y_bus_admittance = 1/(1j*x1)
        self.y2 = 1/(1j*x2)
        self.y0 = 1/(1j*x0+3*zg)

if __name__ == '__main__':
    bus = Bus("Bus 1", 20)
    generator = Generator("G1", "Bus1", 1.0, 0.0, 0.12, 0.14, 0.05, 0)

    print(generator.name, generator.bus, generator.voltage_setpoint, generator.mw_setpoint)
    print(type(generator.y_bus_admittance))
    print(type(generator.y2))
    print(type(generator.y0))