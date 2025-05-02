import numpy as np

class Geometry:

    def __init__(self, name: str, xa: float, ya: float, xb: float, yb: float, xc: float, yc: float):
        self.name = name
        self.xa = xa
        self.ya = ya
        self.xb = xb
        self.yb = yb
        self.xc = xc
        self.yc = yc
        self.Deq = self.calc_Deq()

    # Source: https://www.cuemath.com/distance-formula/

    def calc_Deq(self):

        dab = np.sqrt((self.xb - self.xa)**2 + (self.yb - self.ya)**2)
        dbc = np.sqrt((self.xc - self.xb)**2 + (self.yc - self.yb)**2)
        dac = np.sqrt((self.xc - self.xa)**2 + (self.yc - self.ya)**2)

        return (dab + dbc + dac) / 3

if __name__ == '__main__':
    geometry1 = Geometry("Geometry 1", 0, 0, 18.5, 0, 37, 0)
    print(geometry1.name, geometry1.xa, geometry1.ya, geometry1.xb, geometry1.yb, geometry1.xc, geometry1.yc)
    print(geometry1.Deq)
