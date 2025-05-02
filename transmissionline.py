import numpy as np
from bus import Bus
from conductor import Conductor
from geometry import Geometry
from bundle import Bundle
from settings import s

class TransmissionLine:

    def __init__(self, name: str, bus1: Bus, bus2: Bus, bundle: Bundle, conductor: Conductor, geometry: Geometry, length: float):
        self.name = name
        self.bus1 = bus1
        self.bus2 = bus2
        self.bundle = bundle
        self.conductor = conductor
        self.geometry = geometry
        self.length = length
        self.rseries : float
        self.rpu: float
        self.xseries : float
        self.xpu: float
        self.zpu, self.ypu = self.calc_series()
        self.bpu : float
        self.bpu = self.calc_admittance()
        self.yprim = self.calc_matrix()

    def calc_series(self):
        self.rseries = (self.conductor.resistance/self.bundle.num_conductors)*self.length
        self.xseries = (2 * np.pi * s.frequency) * (2 * 10 ** -7) * np.log(self.geometry.Deq/self.bundle.DSL) * 1609.34 * self.length
        z_base = self.bus1.base_kv**2/s.base_power

        self.rpu = self.rseries/z_base
        self.xpu = self.xseries/z_base
        self.ypu = 1 / (self.rpu + (1j * self.xpu))
        self.zpu = self.rpu + 1j * self.xpu
        return self.zpu, self.ypu

    def calc_admittance(self):
        bshunt = (2 * np.pi * s.frequency) * ((2 * np.pi * 8.854 * 10 ** -12)/(np.log(self.geometry.Deq/self.bundle.DSC))) * 1609.34 * self.length
        y_base = s.base_power/self.bus1.base_kv**2
        self.bpu = bshunt/ y_base
        return self.bpu

    def calc_matrix(self):
        yshunt = 1j * self.bpu
        self.yprim = np.array([[self.ypu + yshunt/2, -1*self.ypu],
                            [-1*self.ypu, self.ypu + yshunt/2]])
        return self.yprim

if __name__ == '__main__':
    bus1 = Bus("Bus_1", 230)
    bus2 = Bus("Bus_2", 230)
    conductor1 = Conductor("Partridge", 0.642, 0.0217, 0.385, 460)
    bundle1 = Bundle("Bundle 1", 2, 1.5, conductor1)
    geometry1 = Geometry("Geometry 1", 0, 0, 18.5, 0, 37, 0)

    line1 = TransmissionLine("Line 1", bus1, bus2, bundle1, conductor1, geometry1, 10)

    print(
        f"Line: {line1.name}, bus1: {line1.bus1}, bus2: {line1.bus2}, bundle1 {line1.bundle}: geometry: {line1.geometry}, length: {line1.length}")
    print(line1.calc_matrix())