import numpy as np
import warnings
from conductor import Conductor

class Bundle:
    def __init__(self, name: str, num_conductors: int, spacing: float, conductor: Conductor):
        self.name = name
        self.num_conductors = num_conductors
        self.spacing = spacing
        self.conductor = conductor
        self.DSC, self.DSL = self.calc_radii()

    def calc_radii(self):
        if self.num_conductors == 1:
            self.DSC = self.conductor.radius
            self.DSL = self.conductor.GMR
        elif self.num_conductors == 2:
            self.DSC = np.sqrt(self.conductor.radius * self.spacing)
            self.DSL = np.sqrt(self.conductor.GMR * self.spacing)
        elif self.num_conductors == 3:
            self.DSC = np.cbrt(self.conductor.radius * self.spacing**2)
            self.DSL = np.cbrt(self.conductor.GMR * self.spacing**2)
        elif self.num_conductors == 4:
            self.DSC = 1.091 * np.power(self.conductor.radius * self.spacing**4, 1/4)
            self.DSL = 1.091 * np.power(self.conductor.GMR * self.spacing**4, 1/4)
        else:

            warnings.warn("Invalid number of conductors for bundle")
            self.DSC = 1
            self.DSL = 1

        return self.DSC, self.DSL

if __name__ == '__main__':
    conductor1 = Conductor("Partridge", 0.642, 0.0217, 0.385, 460)
    bundle1 = Bundle("Bundle 1", 2, 1.5, conductor1)

    print(bundle1.name, bundle1.num_conductors, bundle1.spacing, bundle1.conductor)
