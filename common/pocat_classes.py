# pocat_classes.py
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

@dataclass
class Battery:
    name: str; voltage_min: float; voltage_max: float; capacity_mah: int; vout: float = 0.0

@dataclass
class Load:
    name: str; voltage_req_min: float; voltage_req_max: float; voltage_typical: float
    current_active: float; current_sleep: float
    independent_rail_type: Optional[str] = None
    always_on_in_sleep: bool = False

@dataclass
class PowerIC:
    name: str; vin_min: float; vin_max: float; vout_min: float; vout_max: float; i_limit: float
    operating_current: float; quiescent_current: float; cost: float; theta_ja: float; t_junction_max: int
    shutdown_current: Optional[float] = None

    load_dump_rating_v: float = 0.0; vin: float = 0.0; vout: float = 0.0
    original_i_limit: float = 0.0
    def calculate_power_loss(self, vin: float, i_out: float) -> float: raise NotImplementedError
    def calculate_input_current(self, vin: float, i_out: float) -> float: raise NotImplementedError

@dataclass
class LDO(PowerIC):
    type: str = "LDO"; v_dropout: float = 0.0
    def calculate_power_loss(self, vin: float, i_out: float) -> float: return ((vin - self.vout) * i_out) + (vin * self.operating_current)
    def calculate_input_current(self, vin: float, i_out: float) -> float: return i_out + self.operating_current

@dataclass
class BuckConverter(PowerIC):
    type: str = "Buck"; efficiency: Dict[float, float] = field(default_factory=dict)
    def get_efficiency(self, i_out: float) -> float:
        if not self.efficiency or i_out <= 0: return 0.9
        currents = sorted(self.efficiency.keys()); efficiencies = [self.efficiency[c] for c in currents]
        return np.interp(i_out, currents, efficiencies)
    def calculate_power_loss(self, vin: float, i_out: float) -> float:
        p_out = self.vout * i_out; eff = self.get_efficiency(i_out)
        if eff == 0: return float('inf')
                # 💡 수정: (vin * self.operating_current) 항을 추가하여 IC 자체 손실을 반영합니다.
        conversion_loss = (p_out / eff) - p_out
        return conversion_loss + (vin * self.operating_current)
    def calculate_input_current(self, vin: float, i_out: float) -> float:
        if vin == 0: return float('inf')
        p_out = self.vout * i_out; eff = self.get_efficiency(i_out)
        if eff == 0: return float('inf')
        p_in = p_out / eff
        return (p_in / vin) + self.operating_current