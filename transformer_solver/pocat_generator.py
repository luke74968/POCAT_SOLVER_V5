# transformer_solver/pocat_generator.py
import json
import torch
from tensordict import TensorDict
import copy
from typing import Dict, Any, List, Tuple
from collections import defaultdict  # 💡 이 라인을 추가해주세요!


from dataclasses import asdict
from common.pocat_preprocess import prune_dominated_ic_instances
from common.pocat_classes import PowerIC, LDO, BuckConverter, Load, Battery
from common.pocat_defs import (
    PocatConfig, NODE_TYPE_BATTERY, NODE_TYPE_IC, NODE_TYPE_LOAD,
    FEATURE_DIM, FEATURE_INDEX, SCALAR_PROMPT_FEATURE_DIM
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
        for _ in range(50):
            mid = (low + high) / 2
            if mid < 1e-6: break
            power_loss_at_mid = ic.calculate_power_loss(ic.vin, mid)
            if power_loss_at_mid <= p_loss_max:
                i_limit_based_temp = mid
                low = mid
            else:
                high = mid
                
    return min(ic.i_limit, i_limit_based_temp)

def expand_ic_instances(available_ics: List[PowerIC], loads: List[Load], battery: Battery, constraints: Dict[str, Any]) -> List[PowerIC]:
    """
    모든 유효한 Vin/Vout 조합에 대해 IC 인스턴스를 확장하고 복제합니다.
    """
    potential_vout = sorted(list(set(load.voltage_typical for load in loads)))
    battery.vout = (battery.voltage_min + battery.voltage_max) / 2
    potential_vin = sorted(list(set([battery.vout] + potential_vout)))
    candidate_ics = []
    
    exclusive_loads_per_vout = defaultdict(int)
    for load in loads:
        if load.independent_rail_type in ['exclusive_path', 'exclusive_supplier']:
            exclusive_loads_per_vout[load.voltage_typical] += 1


    for template_ic in available_ics:
        for vin in potential_vin:
            for vout in potential_vout:
                if not (template_ic.vin_min <= vin <= template_ic.vin_max and 
                        template_ic.vout_min <= vout <= template_ic.vout_max):
                    continue
                
                if isinstance(template_ic, LDO) and vin < (vout + template_ic.v_dropout):
                    continue
                if isinstance(template_ic, BuckConverter) and vin <= vout:
                    continue
                
                num_potential_loads = sum(1 for load in loads if load.voltage_typical == vout)
                extra_instances = exclusive_loads_per_vout[vout]
                num_to_create = num_potential_loads + extra_instances


                group_key = f"{template_ic.name}@{vin:.1f}Vin_{vout:.1f}Vout"
                
                for i in range(num_to_create ):
                    concrete_ic = copy.deepcopy(template_ic)
                    concrete_ic.vin, concrete_ic.vout = vin, vout
                    concrete_ic.name = f"{group_key}_copy{i+1}"
                    
                    concrete_ic.original_i_limit = template_ic.i_limit
                    derated_limit = calculate_derated_current_limit(concrete_ic, constraints)
                    
                    if derated_limit > 0:
                        concrete_ic.i_limit = derated_limit
                        candidate_ics.append(concrete_ic)

    return candidate_ics


class PocatGenerator:
    def __init__(self, config_file_path: str):
        with open(config_file_path, "r", encoding='utf-8') as f:
            config_data = json.load(f)

        # --- 💡 2. 지능적인 인스턴스 확장 로직 실행 ---
        # 먼저 설정 파일로부터 원본 객체들을 생성
        battery_obj = Battery(**config_data['battery'])
        loads_obj = [Load(**ld) for ld in config_data['loads']]
        
        original_ics_obj = []
        for ic_data in config_data['available_ics']:
            ic_type = ic_data.get('type')
            if ic_type == 'LDO': original_ics_obj.append(LDO(**ic_data))
            elif ic_type == 'Buck': original_ics_obj.append(BuckConverter(**ic_data))
        
        # 1. 지능적 확장
        candidate_ic_objs = expand_ic_instances(original_ics_obj, loads_obj, battery_obj, config_data['constraints'])

        # --- 💡 2. Dominance Pruning 적용 ---     
        # 객체 리스트를 다시 dict 리스트로 변환
        candidate_ics_dicts = [asdict(ic) for ic in candidate_ic_objs]
        pruned_ics_dicts, _ = prune_dominated_ic_instances(candidate_ics_dicts)
        
        print(f"✅ 지능적 확장 완료: {len(original_ics_obj)}개 원본 IC -> {len(candidate_ic_objs)}개 특화 인스턴스")
        print(f"🔪 Dominance Pruning 완료: {len(candidate_ic_objs)}개 -> {len(pruned_ics_dicts)}개 최종 인스턴스")
        # --- 수정 완료 ---
        
        """
        # --- 💡 [핵심 수정] 최종 인스턴스 목록 로그 출력 ---
        print("\n--- ✅ 최종 후보 IC 인스턴스 목록 ---")
        for ic_dict in sorted(pruned_ics_dicts, key=lambda x: x['name']):
            print(f"   - {ic_dict['name']}")
        print("------------------------------------")
        # --- 수정 완료 ---
        """
        
        config_data['available_ics'] = pruned_ics_dicts # Pruning된 최종 목록 사용
        self.config = PocatConfig(**config_data)

        self.num_nodes = len(self.config.node_names)
        self.num_loads = len(self.config.loads)
        self.num_ics = len(self.config.available_ics)

        self._base_tensors = {}
        self._tensor_cache_by_device = {}
        self._initialize_base_tensors()


    def _initialize_base_tensors(self) -> None:
        """Pre-compute reusable tensors used to build batches."""
        node_features = self._create_feature_tensor()
        connectivity_matrix = self._create_connectivity_matrix(node_features)
        scalar_prompt_features, matrix_prompt_features = self._create_prompt_tensors()

        # Ensure tensors are detached and not tracked by autograd
        self._base_tensors = {
            "nodes": node_features.detach(),
            "connectivity_matrix": connectivity_matrix.detach(),
            "scalar_prompt_features": scalar_prompt_features.detach(),
            "matrix_prompt_features": matrix_prompt_features.detach(),
        }

        base_device = node_features.device
        self._tensor_cache_by_device[base_device] = self._base_tensors

    def _create_feature_tensor(self) -> torch.Tensor:
        features = torch.zeros(self.num_nodes, FEATURE_DIM)
        ambient_temp = self.config.constraints.get("ambient_temperature", 25.0)

        # 모든 노드의 초기 온도는 주변 온도로 설정
        features[:, FEATURE_INDEX["junction_temp"]] = ambient_temp
        
        battery_conf = self.config.battery
        features[0, FEATURE_INDEX["node_type"][0] + NODE_TYPE_BATTERY] = 1.0
        features[0, FEATURE_INDEX["vout_min"]] = battery_conf["voltage_min"]
        features[0, FEATURE_INDEX["vout_max"]] = battery_conf["voltage_max"]

        # --- 💡 3. 확장된 '특화' 인스턴스 정보로 피처 생성 ---
        start_idx = 1
        for i, ic_conf in enumerate(self.config.available_ics):
            idx = start_idx + i
            features[idx, FEATURE_INDEX["node_type"][0] + NODE_TYPE_IC] = 1.0
            features[idx, FEATURE_INDEX["cost"]] = ic_conf.get("cost", 0.0)
            
            # 특화된 vin, vout 값을 피처로 사용 (이제 범위가 아닌 고정값)
            features[idx, FEATURE_INDEX["vin_min"]] = ic_conf.get("vin", 0.0)
            features[idx, FEATURE_INDEX["vin_max"]] = ic_conf.get("vin", 0.0)
            features[idx, FEATURE_INDEX["vout_min"]] = ic_conf.get("vout", 0.0)
            features[idx, FEATURE_INDEX["vout_max"]] = ic_conf.get("vout", 0.0)
            
            # 열 마진이 적용된 i_limit
            features[idx, FEATURE_INDEX["i_limit"]] = ic_conf.get("i_limit", 0.0)
            features[idx, FEATURE_INDEX["theta_ja"]] = ic_conf.get("theta_ja", 999.0)
            features[idx, FEATURE_INDEX["t_junction_max"]] = ic_conf.get("t_junction_max", 125.0)
            # 💡 [수정] quiescent_current와 shutdown_current 값을 텐서에 추가
            features[idx, FEATURE_INDEX["quiescent_current"]] = ic_conf.get("quiescent_current", 0.0)
            features[idx, FEATURE_INDEX["shutdown_current"]] = ic_conf.get("shutdown_current", 0.0)

            ic_type = ic_conf.get("type")
            if ic_type == 'LDO':
                features[idx, FEATURE_INDEX["ic_type_idx"]] = 1.0
                features[idx, FEATURE_INDEX["op_current"]] = ic_conf.get("operating_current", 0.0)
            elif ic_type == 'Buck':
                features[idx, FEATURE_INDEX["ic_type_idx"]] = 2.0
                eff_params = ic_conf.get("efficiency_params", [0.0, 0.0, 0.0])
                s, e = FEATURE_INDEX["efficiency_params"]
                features[idx, s:e] = torch.tensor(eff_params)

        start_idx += len(self.config.available_ics)
        for i, load_conf in enumerate(self.config.loads):
            idx = start_idx + i
            features[idx, FEATURE_INDEX["node_type"][0] + NODE_TYPE_LOAD] = 1.0
            features[idx, FEATURE_INDEX["vin_min"]] = load_conf["voltage_req_min"]
            features[idx, FEATURE_INDEX["vin_max"]] = load_conf["voltage_req_max"]
            features[idx, FEATURE_INDEX["current_active"]] = load_conf["current_active"]
            features[idx, FEATURE_INDEX["current_sleep"]] = load_conf["current_sleep"]

        return features

    def _create_connectivity_matrix(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        노드 피처를 기반으로 물리적으로 연결 가능한 엣지를 나타내는 인접 행렬을 생성합니다.
        (전압 호환성 기반)
        """
        num_nodes = node_features.shape[0]
        node_types = node_features[:, FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
        
        is_parent = (node_types == NODE_TYPE_IC) | (node_types == NODE_TYPE_BATTERY)
        is_child = (node_types == NODE_TYPE_IC) | (node_types == NODE_TYPE_LOAD)
        
        parent_mask = is_parent.unsqueeze(1).expand(-1, num_nodes)
        child_mask = is_child.unsqueeze(0).expand(num_nodes, -1)
        
        parent_vout_min = node_features[:, FEATURE_INDEX["vout_min"]].unsqueeze(1)
        parent_vout_max = node_features[:, FEATURE_INDEX["vout_max"]].unsqueeze(1)
        child_vin_min = node_features[:, FEATURE_INDEX["vin_min"]].unsqueeze(0)
        child_vin_max = node_features[:, FEATURE_INDEX["vin_max"]].unsqueeze(0)
        
        voltage_compatible = (parent_vout_min <= child_vin_max) & (parent_vout_max >= child_vin_min)
        
        # adj_matrix[i, j] = 1  =>  i가 j의 부모가 될 수 있음 (i -> j)
        adj_matrix = parent_mask & child_mask & voltage_compatible
        adj_matrix.diagonal().fill_(False)
        return adj_matrix.to(torch.bool)

    def _create_prompt_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        constraints = self.config.constraints

        scalar_prompt_list = [
            constraints.get("ambient_temperature", 25.0),
            constraints.get("max_sleep_current", 0.0),
            constraints.get("current_margin", 0.0),
            constraints.get("thermal_margin_percent", 0.0)
        ]
        scalar_prompt_features = torch.tensor(scalar_prompt_list, dtype=torch.float32)

        matrix_prompt_features = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float32)
        node_name_to_idx = {name: i for i, name in enumerate(self.config.node_names)}

        for seq in constraints.get("power_sequences", []):
            if seq.get("f") == 1:
                j_idx = node_name_to_idx.get(seq['j'])
                k_idx = node_name_to_idx.get(seq['k'])
                if j_idx is not None and k_idx is not None:
                    matrix_prompt_features[j_idx, k_idx] = 1.0

        return scalar_prompt_features, matrix_prompt_features

    def _get_device_base_tensors(self, device: Any = None) -> Dict[str, torch.Tensor]:
        if device is None:
            # Use the original device used to build the base tensors
            return next(iter(self._tensor_cache_by_device.values()))

        device = torch.device(device)
        if device in self._tensor_cache_by_device:
            return self._tensor_cache_by_device[device]

        # Always move tensors from the initial cache (first entry)
        base_device, base_tensors = next(iter(self._tensor_cache_by_device.items()))
        if device == base_device:
            return base_tensors

        device_tensors = {
            name: tensor.to(device, non_blocking=True)
            for name, tensor in base_tensors.items()
        }
        self._tensor_cache_by_device[device] = device_tensors
        return device_tensors


    def __call__(self, batch_size: int, **kwargs) -> TensorDict:
        device = kwargs.get("device", None)
        base_tensors = self._get_device_base_tensors(device)

        nodes = base_tensors["nodes"].detach().unsqueeze(0).expand(batch_size, -1, -1).clone()
        scalar_prompt = base_tensors["scalar_prompt_features"].detach().unsqueeze(0).expand(batch_size, -1).clone()
        matrix_prompt = base_tensors["matrix_prompt_features"].detach().unsqueeze(0).expand(batch_size, -1, -1).clone()
        connectivity = base_tensors["connectivity_matrix"].detach().unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        return TensorDict({
            "nodes": nodes,
            "scalar_prompt_features": scalar_prompt,
            "matrix_prompt_features": matrix_prompt,
            "connectivity_matrix": connectivity
        }, batch_size=[batch_size])
