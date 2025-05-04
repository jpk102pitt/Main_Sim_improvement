from numpy.linalg.lapack_lite import zgelsd

from bus import Bus
class Solar_gen:

    def __init__(self, name: str, bus: Bus, y_pv : float, f_pv: float, gt: float, gt_stc: float, a_p: float, tc : float, tc_stc: float ):
        self.name = name
        self.bus = bus
        self.y_pv = y_pv
        self.f_pv = f_pv
        self.gt = gt
        self.gt_stc = gt_stc
        self.a_p = a_p * -1 #setting it to a negative value
        self.tc = tc
        self.tc_stc = tc_stc
        #find the real power added to the system from the solar panel
        self.p_pv = y_pv*f_pv*(gt/gt_stc)*(1+a_p*(tc-tc_stc))


        #input of solar irradiance
        #vary the amount of sun hitting the panel
        #tempurature of panel and enviroment affect ouput
        # try to find if there is dust or snow on panel

if __name__ == "__main__":
    # Create a dummy bus for the solar generator
    bus3 = Bus("Bus3", 13.8)  # Voltage level in kV is arbitrary here

    # Define the solar generator inputs
    name = "S1"
    y_pv = 100         # kW
    f_pv = 0.95        # derating factor
    gt = 0.8           # solar radiation [kW/m²]
    gt_stc = 1.0       # standard radiation [kW/m²]
    a_p = 0.0045       # temperature coefficient [%/°C]
    tc = 60            # cell temperature [°C]
    tc_stc = 25        # standard test cell temp [°C]

    # Create Solar_gen instance
    solar = Solar_gen(name, bus3, y_pv, f_pv, gt, gt_stc, a_p, tc, tc_stc)

    # Print the calculated PV power output
    print(f"Solar PV Output Power (p_pv) = {solar.p_pv:.2f} kW")
