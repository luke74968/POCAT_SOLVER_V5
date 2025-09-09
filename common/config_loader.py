# common/config_loader.py
import json
from typing import List, Dict, Tuple, Any

# 공용 데이터 클래스를 import 합니다.
from .pocat_classes import Battery, Load, PowerIC, LDO, BuckConverter

def load_configuration_from_json(config_string: str) -> Tuple[Battery, List[PowerIC], List[Load], Dict[str, Any]]:
    """
    JSON 설정 문자열을 파싱하여 Battery, IC, Load 객체 리스트와 제약조건을 반환합니다.
    """
    config = json.loads(config_string)
    battery = Battery(**config['battery'])
    
    available_ics = []
    for ic_data in config['available_ics']:
        ic_type = ic_data.pop('type')
        if ic_type == 'LDO':
            available_ics.append(LDO(**ic_data))
        elif ic_type == 'Buck':
            available_ics.append(BuckConverter(**ic_data))
        # 나중에 다른 코드에서 원본 type 정보를 사용할 수 있도록 다시 추가해줍니다.
        ic_data['type'] = ic_type

    loads = [Load(**load_data) for load_data in config['loads']]
    constraints = config['constraints']
    
    return battery, available_ics, loads, constraints

def load_configuration_from_file(filepath: str) -> Tuple[Battery, List[PowerIC], List[Load], Dict[str, Any]]:
    """
    JSON 파일 경로를 입력받아 설정 객체들을 로드합니다.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        json_config_string = f.read()
    print(f"✅ 설정 파일 '{filepath}' 로딩 완료!")
    return load_configuration_from_json(json_config_string)