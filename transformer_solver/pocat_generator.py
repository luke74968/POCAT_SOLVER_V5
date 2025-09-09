# transformer_solver/pocat_generator.py
import json
import torch
from tensordict import TensorDict
import copy # 💡 deepcopy를 위해 import
from typing import Dict, Any

from common.pocat_classes import PowerIC, LDO, BuckConverter, Load, Battery
from common.pocat_defs import (
    PocatConfig, NODE_TYPE_BATTERY, NODE_TYPE_IC, NODE_TYPE_LOAD,
    FEATURE_DIM, FEATURE_INDEX
)

def calculate_derated_current_limit(ic: PowerIC, constraints: Dict[str, Any]) -> float:
    """IC의 열(Thermal) 제약조건을 고려하여 실제 사용 가능한 전류 한계를 계산합니다."""
    ambient_temp = constraints.get('ambient_temperature', 25)
    thermal_margin_percent = constraints.get('thermal_margin_percent', 0)
    if ic.theta_ja == 0: return ic.i_limit
    temp_rise_allowed = ic.t_junction_max - ambient_temp
    if temp_rise_allowed <= 0: return 0.0
    
    p_loss_max = (temp_rise_allowed / (ic.theta_ja * (1 + thermal_margin_percent)))
    i_limit_based_temp = ic.i_limit
    
    if isinstance(ic, LDO):
        vin, vout = ic.vin, ic.vout
        op_current = ic.operating_current
        numerator = p_loss_max - (vin * op_current)
        denominator = vin - vout
        if denominator > 0 and numerator > 0:
            i_limit_based_temp = numerator / denominator
    elif isinstance(ic, BuckConverter):
        low, high = 0.0, ic.i_limit
        i_limit_based_temp = 0.0
        for _ in range(50): # 50번의 이진 탐색으로 충분한 정밀도 확보
            mid = (low + high) / 2
            if mid < 1e-6: break
            power_loss_at_mid = ic.calculate_power_loss(ic.vin, mid)
            if power_loss_at_mid <= p_loss_max:
                i_limit_based_temp = mid
                low = mid
            else:
                high = mid
                
    return min(ic.i_limit, i_limit_based_temp)

class PocatGenerator:
    """
    pocat_solver의 config.json 파일을 읽고, IC를 동적으로 복제한 뒤,
    Transformer 모델 학습에 필요한 TensorDict 형태의 데이터를 생성합니다.
    """
    def __init__(self, config_file_path: str):
        with open(config_file_path, "r") as f:
            config_data = json.load(f)
        
        original_config = PocatConfig(**config_data)
        
        # --- 💡 1. OR-Tools처럼 IC 인스턴스를 동적으로 복제 ---
        num_loads_count = len(original_config.loads)
        expanded_ics = []
        original_ics = original_config.available_ics
        
        for ic_template in original_ics:
            # 원본 IC는 그대로 추가
            expanded_ics.append(ic_template)
            # (부하의 개수 - 1) 만큼 복제본 생성
            for i in range(1, num_loads_count):
                ic_copy = copy.deepcopy(ic_template)
                ic_copy['name'] = f"{ic_template['name']}_copy{i}"
                expanded_ics.append(ic_copy)

        print(f"✅ 동적 복제 완료: {len(original_ics)}개의 원본 IC -> {len(expanded_ics)}개의 사용 가능 인스턴스")
        
        # 복제된 IC 목록으로 config 객체 재생성
        config_data['available_ics'] = expanded_ics
        self.config = PocatConfig(**config_data)
        # --- 수정 완료 ---
        
        self.num_nodes = len(self.config.node_names)
        self.num_loads = len(self.config.loads)

    def _create_feature_tensor(self) -> torch.Tensor:
        features = torch.zeros(self.num_nodes, FEATURE_DIM)
        
        # --- 💡 2. 열 마진 계산을 위해 객체를 미리 생성합니다. ---
        battery_obj = Battery(**self.config.battery)
        loads_obj = [Load(**ld) for ld in self.config.loads]
        
        # Battery Features
        battery_conf = self.config.battery
        features[0, FEATURE_INDEX["node_type"][0] + NODE_TYPE_BATTERY] = 1.0
        features[0, FEATURE_INDEX["vout_min"]] = battery_conf["voltage_min"]
        features[0, FEATURE_INDEX["vout_max"]] = battery_conf["voltage_max"]

        # IC Features
        start_idx = 1
        for i, ic_conf in enumerate(self.config.available_ics):
            idx = start_idx + i
            features[idx, FEATURE_INDEX["node_type"][0] + NODE_TYPE_IC] = 1.0
            features[idx, FEATURE_INDEX["cost"]] = ic_conf.get("cost", 0.0)
            features[idx, FEATURE_INDEX["vin_min"]] = ic_conf.get("vin_min", 0.0)
            features[idx, FEATURE_INDEX["vin_max"]] = ic_conf.get("vin_max", 100.0)
            features[idx, FEATURE_INDEX["vout_min"]] = ic_conf.get("vout_min", 0.0)
            features[idx, FEATURE_INDEX["vout_max"]] = ic_conf.get("vout_max", 100.0)
            
            # --- 💡 3. 열 마진이 적용된 전류 한계로 교체 ---
            # 가상 IC 객체를 만들어 계산에 활용 (vin, vout은 대표값 사용)
            ic_type = ic_conf.get('type')
            temp_ic_obj = None
            if ic_type == 'LDO': temp_ic_obj = LDO(**ic_conf)
            elif ic_type == 'Buck': temp_ic_obj = BuckConverter(**ic_conf)
            
            if temp_ic_obj:
                temp_ic_obj.vin = (temp_ic_obj.vin_min + temp_ic_obj.vin_max) / 2
                temp_ic_obj.vout = (temp_ic_obj.vout_min + temp_ic_obj.vout_max) / 2
                derated_limit = calculate_derated_current_limit(temp_ic_obj, self.config.constraints)
                features[idx, FEATURE_INDEX["i_limit"]] = derated_limit
            else:
                features[idx, FEATURE_INDEX["i_limit"]] = ic_conf.get("i_limit", 0.0)
            # --- 수정 끝 ---

        # Load Features
        start_idx += len(self.config.available_ics)
        for i, load_conf in enumerate(self.config.loads):
            idx = start_idx + i
            features[idx, FEATURE_INDEX["node_type"][0] + NODE_TYPE_LOAD] = 1.0
            features[idx, FEATURE_INDEX["vin_min"]] = load_conf["voltage_req_min"]
            features[idx, FEATURE_INDEX["vin_max"]] = load_conf["voltage_req_max"]
            features[idx, FEATURE_INDEX["current_active"]] = load_conf["current_active"]
            features[idx, FEATURE_INDEX["current_sleep"]] = load_conf["current_sleep"]

        return features
    def __call__(self, batch_size: int, instance_repeats: int = 1) -> TensorDict:
        node_features = self._create_feature_tensor()
        constraints = self.config.constraints
        prompt_features = torch.tensor(
            [
                constraints.get("ambient_temperature", 25.0),
                constraints.get("max_sleep_current", 0.0),
            ]
        )

        # Expand along batch dimension and create repeated clones
        node_features = node_features.unsqueeze(0).expand(batch_size, -1, -1)
        prompt_features = prompt_features.unsqueeze(0).expand(batch_size, -1)
        
        node_features = node_features.unsqueeze(1).expand(batch_size, instance_repeats, -1, -1)
        prompt_features = prompt_features.unsqueeze(1).expand(batch_size, instance_repeats, -1)

        # 💡 수정된 부분: static_info를 TensorDict에서 제거
        return TensorDict(
            {
                "nodes": node_features,
                "prompt_features": prompt_features,
            },
            batch_size=[batch_size, instance_repeats],
        )