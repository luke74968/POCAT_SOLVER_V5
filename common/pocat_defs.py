# pocat_defs.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

# 노드 타입을 구분하기 위한 상수
NODE_TYPE_BATTERY = 0
NODE_TYPE_IC = 1
NODE_TYPE_LOAD = 2

# 노드 피처 텐서의 각 인덱스가 어떤 값을 의미하는지 정의
FEATURE_INDEX = {
    "node_type": (0, 3),        # One-hot: Battery, IC, Load
    "cost": 3,
    "vin_min": 4,
    "vin_max": 5,
    "vout_min": 6,
    "vout_max": 7,
    "i_limit": 8,
    "current_active": 9,
    "current_sleep": 10,
    "current_out": 11,          # (신규) IC의 현재 총 출력 전류 (동적)
    "ic_type_idx": 12,          # (신규) 0: N/A, 1: LDO, 2: Buck (정적)
    "op_current": 13,           # (신규) LDO의 동작 전류 (정적)
    "efficiency_params": (14, 17), # (신규) Buck 손실 계산 계수 a,b,c (정적)
    "theta_ja": 17,             # (신규) 열저항 (정적)
    "t_junction_max": 18,       # (신규) 최대 허용 정션 온도 (정적)
    "junction_temp": 19,        # (신규) 현재 정션 온도 (동적)
}

# 전체 피처 차원
FEATURE_DIM = 20
# 💡 수정: 기존 PROMPT_FEATURE_DIM을 SCALAR_PROMPT_FEATURE_DIM으로 변경
SCALAR_PROMPT_FEATURE_DIM = 4


@dataclass
class PocatConfig:
    """ config.json 파일의 내용을 담는 데이터 클래스 """
    battery: Dict[str, Any]
    available_ics: List[Dict[str, Any]]
    loads: List[Dict[str, Any]]
    constraints: Dict[str, Any]
    
    node_names: List[str] = field(default_factory=list)
    node_types: List[int] = field(default_factory=list)

    def __post_init__(self):
        # 초기 로드 시 한 번만 호출
        self.rebuild_node_lists()

    def rebuild_node_lists(self):
        """
        IC 목록이 변경되었을 때 node_names와 node_types 리스트를 다시 생성합니다.
        """
        self.node_names.clear()
        self.node_types.clear()
        
        self.node_names.append(self.battery['name'])
        self.node_types.append(NODE_TYPE_BATTERY)
        for ic in self.available_ics:
            self.node_names.append(ic['name'])
            self.node_types.append(NODE_TYPE_IC)
        for load in self.loads:
            self.node_names.append(load['name'])
            self.node_types.append(NODE_TYPE_LOAD)