from mlpdft.constants import ENERGY_OFFSET


class EnergyOffsetCalculator:
    def __init__(self, group: str, f_num: int, li_num: int, b_num: int):
        self.group = group
        self.li_num = li_num
        self.f_num = f_num
        self.b_num = b_num

    def calculate(self) -> float:
        energy = ENERGY_OFFSET[3] * self.li_num + ENERGY_OFFSET[9] * self.f_num
        return energy


LIF64_ISOLATED = EnergyOffsetCalculator("LIF64_ISOLATED", 32, 32, 0)
LIF64_KJPAW_V2 = EnergyOffsetCalculator("LIF64_KJPAW_V2", 32, 32, 0)
LIFINTERFACE_KJPAW_V1 = EnergyOffsetCalculator("LIFINTERFACE_KJPAW_V1", 36, 86, 0)
LIFINTERFACE_KJPAW_NPT = EnergyOffsetCalculator("LIFINTERFACE_KJPAW_NPT", 36, 86, 0)
LIFINTERFACE_KJPAW_NPT_V2 = EnergyOffsetCalculator(
    "LIFINTERFACE_KJPAW_NPT_V2", 36, 86, 0
)
LIWITHF_V3 = EnergyOffsetCalculator("LIWITHF_V3", 1, 53, 0)
LIWITHF_ISOLATED = EnergyOffsetCalculator("LIWITHF_ISOLATED", 1, 53, 0)
LIWITHF_NPT_FINAL = EnergyOffsetCalculator("LIWITHF_NPT_FINAL", 1, 53, 0)
BLI_V2 = EnergyOffsetCalculator("BLI_V2", 16, 16, 0)
LIBF4_V4 = EnergyOffsetCalculator("LIBF4_V4", 12, 3, 3)

for group, calculator in [
    ("LIF64_ISOLATED", LIF64_ISOLATED),
    ("LIF64_KJPAW_V2", LIF64_KJPAW_V2),
    ("LIFINTERFACE_KJPAW_V1", LIFINTERFACE_KJPAW_V1),
    ("LIFINTERFACE_KJPAW_NPT", LIFINTERFACE_KJPAW_NPT),
    ("LIFINTERFACE_KJPAW_NPT_V2", LIFINTERFACE_KJPAW_NPT_V2),
    ("LIWITHF_V3", LIWITHF_V3),
    ("LIWITHF_ISOLATED", LIWITHF_ISOLATED),
    ("LIWITHF_NPT_FINAL", LIWITHF_NPT_FINAL),
    ("BLI_V2", BLI_V2),
    ("LIBF4_V4", LIBF4_V4),
]:
    print(f"{group}: {calculator.calculate()}")
