# pocat_defs.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

# 노드 타입을 구분하기 위한 상수
NODE_TYPE_BATTERY = 0
NODE_TYPE_IC = 1
NODE_TYPE_LOAD = 2

# 각 노드의 피처 벡터에서 각 속성의 인덱스
# [0:3]=NodeType, [3]=Cost, [4]=V_in_min, [5]=V_in_max, [6]=V_out_min, [7]=V_out_max,
# [8]=I_limit, [9]=I_active, [10]=I_sleep
FEATURE_INDEX = {
    "node_type": (0, 3),
    "cost": 3,
    "vin_min": 4,
    "vin_max": 5,
    "vout_min": 6,
    "vout_max": 7,
    "i_limit": 8,
    "current_active": 9,
    "current_sleep": 10,
}
FEATURE_DIM = 11

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