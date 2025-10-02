# pocat_core.py
import json
import copy
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from ortools.sat.python import cp_model

from common.pocat_classes import Battery, Load, PowerIC, LDO, BuckConverter
# 순환 참조를 피하기 위해 함수를 직접 임포트하지 않고, main에서 넘겨받도록 구조 변경
# from pocat_visualizer import check_solution_validity, print_and_visualize_one_solution

SCALE = 1_000_000_000


# 솔버 콜백 클래스
class SolutionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, ic_is_used, edges):
        super().__init__()
        self.__solution_count = 0
        self.__ic_is_used = ic_is_used
        self.__edges = edges
        self.solutions = []
    def on_solution_callback(self):
        self.__solution_count += 1
        current_solution = {
            "score": self.ObjectiveValue(),
            "used_ic_names": {name for name, var in self.__ic_is_used.items() if self.Value(var)},
            "active_edges": [(p, c) for (p, c), var in self.__edges.items() if self.Value(var)]}
        self.solutions.append(current_solution)
    def solution_count(self): return self.__solution_count

class SolutionLogger(cp_model.CpSolverSolutionCallback):
    def __init__(self, ic_is_used, edges, limit=1):
        super().__init__()
        self.__solution_count = 0
        self.__ic_is_used = ic_is_used
        self.__edges = edges
        self.limit = limit
        self.solutions = []
    def on_solution_callback(self):
        if len(self.solutions) >= self.limit:
            self.StopSearch()
            return
        self.__solution_count += 1
        print(f"  -> 대표 솔루션 #{self.__solution_count} 발견!")
        current_solution = {
            "score": self.ObjectiveValue(),
            "used_ic_names": {name for name, var in self.__ic_is_used.items() if self.Value(var)},
            "active_edges": [(p, c) for (p, c), var in self.__edges.items() if self.Value(var)]
        }
        self.solutions.append(current_solution)

# 핵심 로직 함수들
def calculate_derated_current_limit(ic: PowerIC, constraints: Dict[str, Any]) -> float:
    ambient_temp = constraints.get('ambient_temperature', 25)
    thermal_margin_percent = constraints.get('thermal_margin_percent', 0)
    if ic.theta_ja == 0: return ic.i_limit
    temp_rise_allowed = ic.t_junction_max - ambient_temp
    if temp_rise_allowed <= 0: return 0
    p_loss_max = (temp_rise_allowed / (ic.theta_ja * (1 + thermal_margin_percent)))
    i_limit_based_temp = ic.i_limit
    if isinstance(ic, LDO):
        vin, vout = ic.vin, ic.vout; op_current = ic.operating_current
        numerator = p_loss_max - (vin * op_current); denominator = vin - vout
        if denominator > 0 and numerator > 0: i_limit_based_temp = numerator / denominator
    elif isinstance(ic, BuckConverter):
        # --- 💡 수정: 선형 스캔을 이진 탐색으로 변경 ---
        low = 0.0
        high = ic.i_limit
        i_limit_based_temp = 0.0
        
        # 100회 반복으로 충분히 높은 정밀도를 얻을 수 있습니다 (2^100)
        for _ in range(100): 
            mid = (low + high) / 2
            if mid < 1e-6: # 전류가 매우 작으면 탐색 중단
                break
                
            # mid 전류에서의 전력 손실 계산
            power_loss_at_mid = ic.calculate_power_loss(ic.vin, mid)
            
            if power_loss_at_mid <= p_loss_max:
                # 허용 손실보다 작거나 같으면, 이 전류값은 유효함
                # 더 높은 전류도 가능한지 탐색하기 위해 low를 mid로 이동
                i_limit_based_temp = mid
                low = mid
            else:
                # 허용 손실보다 크면, 전류를 낮춰야 함
                high = mid
        # --- 수정 끝 ---
    return min(ic.i_limit, i_limit_based_temp)

def load_configuration(config_string: str) -> Tuple[Battery, List[PowerIC], List[Load], Dict[str, Any]]:
    config = json.loads(config_string); battery = Battery(**config['battery']); available_ics = []
    for ic_data in config['available_ics']:
        ic_type = ic_data.pop('type')
        if ic_type == 'LDO': available_ics.append(LDO(**ic_data))
        elif ic_type == 'Buck': available_ics.append(BuckConverter(**ic_data))
    loads = [Load(**load_data) for load_data in config['loads']]; constraints = config['constraints']
    print("✅ 설정 파일 로딩 완료!")
    return battery, available_ics, loads, constraints

def expand_ic_instances(available_ics: List[PowerIC], loads: List[Load], battery: Battery, constraints: Dict[str, Any]) -> Tuple[List[PowerIC], Dict[str, List[str]]]:
    print("\n⚙️  IC 인스턴스 확장 및 복제 시작...")
    potential_vout = sorted(list(set(load.voltage_typical for load in loads)))
    battery.vout = (battery.voltage_min + battery.voltage_max) / 2
    potential_vin = sorted(list(set([battery.vout] + potential_vout))); candidate_ics, ic_groups = [], {}
    for template_ic in available_ics:
        for vin in potential_vin:
            for vout in potential_vout:
                if not (template_ic.vin_min <= vin <= template_ic.vin_max): continue
                if not (template_ic.vout_min <= vout <= template_ic.vout_max): continue
                if isinstance(template_ic, LDO):
                    if vin < (vout + template_ic.v_dropout): continue
                elif isinstance(template_ic, BuckConverter):
                    if vin <= vout: continue
                num_potential_loads = sum(1 for load in loads if load.voltage_typical == vout)
                group_key = f"{template_ic.name}@{vin:.1f}Vin_{vout:.1f}Vout"; current_group = []
                for i in range(num_potential_loads):
                    concrete_ic = copy.deepcopy(template_ic); concrete_ic.vin, concrete_ic.vout = vin, vout
                    concrete_ic.name = f"{group_key}_copy{i+1}"
                    
                    # --- [핵심 수정] 열 마진 계산 전, 원래 스펙 저장 ---
                    concrete_ic.original_i_limit = template_ic.i_limit
                    # --- 수정 끝 ---

                    derated_limit = calculate_derated_current_limit(concrete_ic, constraints)
                    if derated_limit <= 0: continue
                    concrete_ic.i_limit = derated_limit # 열 마진이 적용된 값으로 덮어쓰기
                    candidate_ics.append(concrete_ic); current_group.append(concrete_ic.name)
                if current_group: ic_groups[group_key] = current_group
    print(f"   - (필터링 포함) 생성된 최종 후보 IC 인스턴스: {len(candidate_ics)}개")
    return candidate_ics, ic_groups

def _initialize_model_variables(model, candidate_ics, loads, battery):
    """모델의 기본 변수들(노드, 엣지, IC 사용 여부)을 생성하고 반환합니다."""
    all_ic_and_load_nodes = candidate_ics + loads
    parent_nodes = [battery] + candidate_ics
    all_nodes = parent_nodes + all_ic_and_load_nodes
    node_names = list(set(n.name for n in all_nodes))
    ic_names = [ic.name for ic in candidate_ics]
    
    edges = {}
    for p in parent_nodes:
        for c in all_ic_and_load_nodes:
            if p.name == c.name: continue
            is_compatible = False
            if p.name == battery.name:
                if isinstance(c, PowerIC) and (c.vin_min <= battery.voltage_min and battery.voltage_max <= c.vin_max):
                    is_compatible = True
            elif isinstance(p, PowerIC):
                child_vin_req = c.vin if hasattr(c, 'vin') else c.voltage_typical
                if p.vout == child_vin_req:
                    is_compatible = True
            if is_compatible:
                edges[(p.name, c.name)] = model.NewBoolVar(f'edge_{p.name}_to_{c.name}')
    
    ic_is_used = {ic.name: model.NewBoolVar(f'is_used_{ic.name}') for ic in candidate_ics}
    
    print(f"   - (필터링 후) 생성된 'edge' 변수: {len(edges)}개")
    # `parent_nodes`를 반환 값에 추가
    return all_nodes, parent_nodes, node_names, ic_names, edges, ic_is_used

# --- 💡 2. 각 제약 조건을 추가하는 함수들 ---
def add_base_topology_constraints(model, candidate_ics, loads, battery, edges, ic_is_used):
    """전력망의 가장 기본적인 연결 규칙을 정의합니다."""
    all_ic_and_load_nodes = candidate_ics + loads
    parent_nodes = [battery] + candidate_ics

    # 사용되는 IC는 반드시 출력이 있어야 함
    for ic in candidate_ics:
        outgoing = [edges[ic.name, c.name] for c in all_ic_and_load_nodes if (ic.name, c.name) in edges]
        if outgoing:
            model.Add(sum(outgoing) > 0).OnlyEnforceIf(ic_is_used[ic.name])
            model.Add(sum(outgoing) == 0).OnlyEnforceIf(ic_is_used[ic.name].Not())
        else:
            model.Add(ic_is_used[ic.name] == False)
    # 모든 부하는 반드시 하나의 부모를 가져야 함
    for load in loads:
        possible_parents = [edges[p.name, load.name] for p in parent_nodes if (p.name, load.name) in edges]
        if possible_parents: model.AddExactlyOne(possible_parents)
    # 사용되는 IC는 반드시 하나의 부모를 가져야 함
    for ic in candidate_ics:
        incoming = [edges[p.name, ic.name] for p in parent_nodes if (p.name, ic.name) in edges]
        if incoming:
            model.Add(sum(incoming) == 1).OnlyEnforceIf(ic_is_used[ic.name])
            model.Add(sum(incoming) == 0).OnlyEnforceIf(ic_is_used[ic.name].Not())

def add_ic_group_constraints(model, ic_groups, ic_is_used):
    """복제된 IC 그룹 내에서의 사용 순서를 강제합니다."""
    for copies in ic_groups.values():
        for i in range(len(copies) - 1):
            model.AddImplication(ic_is_used[copies[i+1]], ic_is_used[copies[i]])

def add_current_limit_constraints(model, candidate_ics, loads, constraints, edges):
    """IC의 전류 한계(열 마진, 전기 마진) 제약 조건을 추가합니다."""
    all_ic_and_load_nodes = candidate_ics + loads
    
    child_current_draw = {node.name: int(node.current_active * SCALE) for node in loads}
    potential_loads_for_ic = defaultdict(list)
    for ic in candidate_ics:
        for load in loads:
            if ic.vout == load.voltage_typical:
                potential_loads_for_ic[ic.name].append(load.current_active)
    for ic in candidate_ics:
        max_potential_i_out = sum(potential_loads_for_ic[ic.name])
        realistic_i_out = min(ic.i_limit, max_potential_i_out)
        child_current_draw[ic.name] = int(ic.calculate_input_current(vin=ic.vin, i_out=realistic_i_out) * SCALE)

    current_margin = constraints.get('current_margin', 0.1)
    for p in candidate_ics:
        terms = [child_current_draw[c.name] * edges[p.name, c.name] for c in all_ic_and_load_nodes if (p.name, c.name) in edges]
        if terms:
            model.Add(sum(terms) <= int(p.i_limit * SCALE))
            model.Add(sum(terms) <= int(p.original_i_limit * (1 - current_margin) * SCALE))

def add_power_sequence_constraints(model, candidate_ics, loads, battery, constraints, node_names, edges, ic_is_used):
    """
    (개선된 방식) 정수 '스테이지' 변수를 사용하여 전원 시퀀스 제약 조건을 효율적으로 추가합니다.
    - edge(p->c)가 활성화되면 stage[c] > stage[p]
    - 시퀀스 규칙(j가 k보다 먼저)이 있으면, k의 부모 IC 스테이지 > j의 부모 IC 스테이지
    """
    if 'power_sequences' not in constraints or not constraints['power_sequences']:
        return

    print("   - (개선) 스테이지 변수 기반 Power Sequence 제약 조건 추가...")

    num_nodes = len(node_names)
    # 1. 각 노드에 대한 스테이지 정수 변수 생성
    stage = {name: model.NewIntVar(0, num_nodes - 1, f"stage_{name}") for name in node_names}

    # 2. 배터리는 항상 스테이지 0으로 고정 (이제 'battery'가 정의되어 오류가 발생하지 않습니다)
    model.Add(stage[battery.name] == 0)

    # 3. 엣지가 활성화되면, 자식의 스테이지는 부모보다 커야 함
    for (p_name, c_name), edge_var in edges.items():
        # stage[c] >= stage[p] + 1
        model.Add(stage[c_name] >= stage[p_name] + 1).OnlyEnforceIf(edge_var)


    parent_candidates = [battery] + candidate_ics

    # 4. Power Sequence 규칙 적용
    for seq in constraints['power_sequences']:
        if seq.get('f') != 1:
            continue
        
        j_name, k_name = seq['j'], seq['k']

        # 각 부하(j, k)에 연결될 수 있는 모든 부모 IC 후보를 찾습니다.
        j_parents = [(p.name, edges[p.name, j_name]) for p in candidate_ics if (p.name, j_name) in edges]
        k_parents = [(p.name, edges[p.name, k_name]) for p in candidate_ics if (p.name, k_name) in edges]
        
        if not j_parents or not k_parents:
            continue

        # j와 k가 각각 어떤 부모에 연결되었을 때, 그 부모의 스테이지를 나타낼 변수
        j_parent_stage = model.NewIntVar(0, num_nodes - 1, f"stage_parent_of_{j_name}")
        k_parent_stage = model.NewIntVar(0, num_nodes - 1, f"stage_parent_of_{k_name}")
        
        # 부모-자식 관계가 활성화되면, 부모의 스테이지 값을 가져옴
        for p_name, edge_var in j_parents:
            model.Add(j_parent_stage == stage[p_name]).OnlyEnforceIf(edge_var)
        for p_name, edge_var in k_parents:
            model.Add(k_parent_stage == stage[p_name]).OnlyEnforceIf(edge_var)
        
        # 핵심 제약: k 부모의 스테이지가 j 부모의 스테이지보다 커야 한다 (시간적 선후 관계)
        model.Add(k_parent_stage > j_parent_stage)

        # 기존의 '동일 부모 금지' 규칙도 함께 적용
        j_ic_parents = [p for p in j_parents if p[0] != battery.name]
        k_ic_parents = [p for p in k_parents if p[0] != battery.name]

        for p_ic_name, j_edge_var in j_parents:
            for q_ic_name, k_edge_var in k_parents:
                if p_ic_name == q_ic_name:
                    model.AddBoolOr([j_edge_var.Not(), k_edge_var.Not()])   

# --- 💡 3. 재구성된 메인 모델 생성 함수 수정 ---
def create_solver_model(candidate_ics, loads, battery, constraints, ic_groups):
    """
    OR-Tools 모델을 생성하고 모든 제약 조건을 추가한 뒤 반환합니다.
    """
    print("\n🧠 OR-Tools 모델 생성 시작...")
    model = cp_model.CpModel()

    # 1. 변수 초기화
    # `parent_nodes`를 변수로 받음
    all_nodes, parent_nodes, node_names, ic_names, edges, ic_is_used = _initialize_model_variables(
        model, candidate_ics, loads, battery
    )
    
    # 2. 제약 조건 추가
    add_base_topology_constraints(model, candidate_ics, loads, battery, edges, ic_is_used)
    add_ic_group_constraints(model, ic_groups, ic_is_used)
    add_current_limit_constraints(model, candidate_ics, loads, constraints, edges)
    #add_power_sequence_constraints(model, candidate_ics, loads, constraints, node_names, ic_names, edges)
    add_power_sequence_constraints(model, candidate_ics, loads, battery, constraints, node_names, edges, ic_is_used)
    
    # `parent_nodes`를 올바르게 전달
    add_independent_rail_constraints(model, loads, candidate_ics, all_nodes, parent_nodes, edges)

    is_always_on_path = add_always_on_constraints(model, all_nodes, loads, candidate_ics, battery, edges)
    add_sleep_current_constraints(model, battery, candidate_ics, loads, constraints, edges, is_always_on_path)

    # N. 목표 함수 설정
    cost_objective = sum(int(ic.cost * 10000) * ic_is_used[ic.name] for ic in candidate_ics)
    model.Minimize(cost_objective)
    
    print("✅ 모델 생성 완료!")
    return model, edges, ic_is_used

# --- 💡 Independent Rail 제약조건 함수 ---
def add_independent_rail_constraints(model, loads, candidate_ics, all_nodes, parent_nodes, edges):
    """
    독립 레일(Independent Rail) 제약 조건을 모델에 추가합니다.
    - exclusive_path: 부하로 가는 경로 전체를 다른 부하와 공유하지 않습니다.
    - exclusive_supplier: 부하에 전원을 공급하는 IC는 다른 어떤 자식도 가질 수 없습니다.
    """
    all_ic_and_load_nodes = candidate_ics + loads
    
    # 모든 자식(부하 + IC)의 수를 세는 변수
    num_children_all = {p.name: model.NewIntVar(0, len(all_ic_and_load_nodes), f"num_children_all_{p.name}") for p in parent_nodes}
    for p in parent_nodes:
        outgoing_edges = [edges[p.name, c.name] for c in all_ic_and_load_nodes if (p.name, c.name) in edges]
        model.Add(num_children_all[p.name] == sum(outgoing_edges))

    for load in loads:
        rail_type = load.independent_rail_type

        # exclusive_supplier: 부하/IC 통틀어 자식 1개
        if rail_type == 'exclusive_supplier':
            for p_ic in candidate_ics:
                if (p_ic.name, load.name) in edges:
                    model.Add(num_children_all[p_ic.name] == 1).OnlyEnforceIf(edges[(p_ic.name, load.name)])
        
        # exclusive_path: 경로 전체 격리
        elif rail_type == 'exclusive_path':
            is_on_exclusive_path = {node.name: model.NewBoolVar(f"on_exc_path_{load.name}_{node.name}") for node in all_nodes}
            model.Add(is_on_exclusive_path[load.name] == 1)
            for other_load in loads:
                if other_load.name != load.name:
                    model.Add(is_on_exclusive_path[other_load.name] == 0)
            
            for c_node in all_ic_and_load_nodes:
                for p_node in parent_nodes:
                    if (p_node.name, c_node.name) in edges:
                        model.AddImplication(is_on_exclusive_path[c_node.name], is_on_exclusive_path[p_node.name]).OnlyEnforceIf(edges[(p_node.name, c_node.name)])
            
            for p_ic in candidate_ics:
                # 이 경로 위에 있는 IC는 다른 어떤 자식도 가질 수 없음
                model.Add(num_children_all[p_ic.name] <= 1).OnlyEnforceIf(is_on_exclusive_path[p_ic.name])


# --- 💡 Always-On 및 Sleep Current 제약조건 함수 ---
def add_always_on_constraints(model, all_nodes, loads, candidate_ics, battery, edges):
    all_ic_and_load_nodes = candidate_ics + loads
    is_always_on_path = {node.name: model.NewBoolVar(f"is_ao_{node.name}") for node in all_nodes}
    
    # --- 👇 [수정 1] 배터리는 항상 AO(Always-On) 상태로 고정 ---
    model.Add(is_always_on_path[battery.name] == 1)

    for ld in loads:
        model.Add(is_always_on_path[ld.name] == int(ld.always_on_in_sleep))
    for ic in candidate_ics:
        children = [c for c in all_ic_and_load_nodes if (ic.name, c.name) in edges]
        if not children:
            model.Add(is_always_on_path[ic.name] == 0)
            continue
        z_list = []
        for ch in children:
            e = edges[(ic.name, ch.name)]
            z = model.NewBoolVar(f"ao_and_{ic.name}__{ch.name}")
            model.Add(z <= e); model.Add(z <= is_always_on_path[ch.name]); model.Add(z >= e + is_always_on_path[ch.name] - 1)
            z_list.append(z)
        for z in z_list: model.Add(is_always_on_path[ic.name] >= z)
        model.Add(is_always_on_path[ic.name] <= sum(z_list))
    return is_always_on_path


def add_sleep_current_constraints(model, battery, candidate_ics, loads, constraints, edges, is_always_on_path):
    """
    (최종 개선 로직) Sleep 전류 제약 조건을 추가합니다.
    - IC 상태 정의:
        1. is_ao: AO 경로에 포함되어 Iop(동작 전류) 소모
        2. use_ishut: 비-AO 경로지만, 부모가 AO라서 차단 전류(I_shut 또는 Iq) 소모
        3. no_current: 비-AO 경로이고, 부모도 비-AO라서 전류 소모 없음
    - 위 세 상태는 상호 배타적이며, 반드시 하나는 참이 되도록 제약합니다.
    """
    max_sleep = constraints.get('max_sleep_current', 0.0)
    if max_sleep <= 0:
        return

    # ... (헬퍼 함수 및 기본 변수 초기화는 이전과 동일) ...
    def bool_and(a, b, name):
        w = model.NewBoolVar(name)
        model.Add(w <= a); model.Add(w <= b); model.Add(w >= a + b - 1)
        return w

    def gate_const_by_bool(const_int, b, name):
        y = model.NewIntVar(0, max(0, const_int), name)
        model.Add(y == const_int).OnlyEnforceIf(b); model.Add(y == 0).OnlyEnforceIf(b.Not())
        return y

    def gate_int_by_bool(x, ub, b, name):
        y = model.NewIntVar(0, max(0, ub), name)
        model.Add(y == x).OnlyEnforceIf(b); model.Add(y == 0).OnlyEnforceIf(b.Not())
        return y

    parent_nodes = [battery] + candidate_ics
    all_ic_and_load_nodes = candidate_ics + loads
    
    # NODE_UB 계산 시 shutdown_current가 None일 수 있으므로 처리
    max_ic_self_current = sum(
        int(max(ic.operating_current, ic.quiescent_current, ic.shutdown_current or 0) * SCALE)
        for ic in candidate_ics
    )
    NODE_UB = max_ic_self_current + sum(int(ld.current_sleep * SCALE) for ld in loads) + 1

    node_sleep_in = {}
    node_sleep_ub = {}

    for ld in loads:
        const_val = max(0, int(ld.current_sleep * SCALE))
        v = gate_const_by_bool(const_val, is_always_on_path[ld.name], f"sleep_in_{ld.name}")
        node_sleep_in[ld.name] = v
        node_sleep_ub[ld.name] = const_val

    for ic in candidate_ics:
        node_sleep_in[ic.name] = model.NewIntVar(0, NODE_UB, f"sleep_in_{ic.name}")
        node_sleep_ub[ic.name] = NODE_UB

    # --- IC별 제약 조건 구성 (개선된 로직) ---
    for ic in candidate_ics:
        iop = max(0, int(ic.operating_current * SCALE))
        
        # --- 👇 [핵심 수정] shutdown_current가 없으면 quiescent_current 사용 ---
        # shutdown_current 값이 유효한지(None이 아니고 0보다 큰지) 확인
        if ic.shutdown_current is not None and ic.shutdown_current > 0:
            i_shut = max(0, int(ic.shutdown_current * SCALE))
        else:
            # 유효하지 않으면 quiescent_current를 대신 사용
            i_shut = max(0, int(ic.quiescent_current * SCALE))
        # --- 수정 완료 ---

        ic_self = model.NewIntVar(0, max(iop, i_shut), f"sleep_self_{ic.name}")
        
        is_ao = is_always_on_path[ic.name]

        # (A) IC의 3가지 상태(is_ao, use_ishut, no_current) 정의
        parent_is_ao = model.NewBoolVar(f"parent_of_{ic.name}_is_ao")
        
        # 1. parent_is_ao ⇔ OR(edge(p->ic) ∧ is_p_ao) 등가 관계 설정
        possible_parents = [p for p in parent_nodes if (p.name, ic.name) in edges]
        z_list = []
        if possible_parents:
            for p in possible_parents:
                is_p_ao = is_always_on_path[p.name]
                z = bool_and(edges[(p.name, ic.name)], is_p_ao, f"z_{p.name}_{ic.name}")
                z_list.append(z)
            
            model.AddBoolOr([parent_is_ao.Not()] + z_list)
            for z in z_list:
                model.AddImplication(z, parent_is_ao)
        else:
            model.Add(parent_is_ao == 0)

        # 2. 3가지 상태 변수 정의 및 상호 배타적 관계 설정
        use_ishut = bool_and(is_ao.Not(), parent_is_ao, f"use_ishut_{ic.name}")
        no_current = bool_and(is_ao.Not(), parent_is_ao.Not(), f"no_current_{ic.name}")
        model.AddExactlyOne([is_ao, use_ishut, no_current])

        # (B) 상태에 따른 IC 자체 소모 전류(ic_self) 할당
        model.Add(ic_self == iop).OnlyEnforceIf(is_ao)
        model.Add(ic_self == i_shut).OnlyEnforceIf(use_ishut)
        model.Add(ic_self == 0).OnlyEnforceIf(no_current)

        # ... (이하 로직은 이전과 동일합니다) ...

        # (C) 자식 노드들이 요구하는 전류 합산 (AO 경로 자식만)
        children = [c for c in all_ic_and_load_nodes if (ic.name, c.name) in edges]
        child_terms = []
        ub_sum = 0
        for c in children:
            edge_ic_c = edges[(ic.name, c.name)]
            
            # --- 👇 [핵심 수정] 자식이 AO인지 여부를 여기서 판단하지 않습니다. ---
            #      연결만 되어 있다면 자식의 입력 전류(node_sleep_in)를 그대로 더합니다.
            #      (기존: use_c_sleep = bool_and(edge_ic_c, is_always_on_path[c.name], ...))
            use_c_sleep = edge_ic_c
            # --- 수정 완료 ---

            ub_c = node_sleep_ub[c.name]
            term = gate_int_by_bool(node_sleep_in[c.name], ub_c, use_c_sleep, f"sleep_term_{ic.name}__{c.name}")
            child_terms.append(term)
            ub_sum += ub_c


        children_out = model.NewIntVar(0, max(0, ub_sum), f"sleep_out_{ic.name}")
        model.Add(children_out == (sum(child_terms) if child_terms else 0))

        # (D) 출력 전류를 입력 전류로 변환 (LDO/Buck)
        in_for_children = model.NewIntVar(0, NODE_UB, f"sleep_children_in_{ic.name}")
        if isinstance(ic, LDO):
            model.Add(in_for_children == children_out)
        elif isinstance(ic, BuckConverter):
            eff_sleep = constraints.get('sleep_efficiency_guess', 0.35)
            eff_sleep = max(0.05, min(eff_sleep, 0.85))
            vin_ref = ic.vin if ic.vin > 0 else battery.voltage_min
            vin_eff = max(1e-6, vin_ref * eff_sleep)
            p = max(1, int(round(ic.vout * 1000)))
            q = max(1, int(round(vin_eff * 1000)))
            model.Add(in_for_children * q >= children_out * p)
        else:
            model.Add(in_for_children == 0)

        # (E) IC의 총 입력 전류 = 자체 소모 + 자식 공급용
        model.Add(node_sleep_in[ic.name] == ic_self + in_for_children)

    # --- 최종 제약 조건: 배터리 관점 (이전과 동일) ---
    top_children = [c for c in all_ic_and_load_nodes if (battery.name, c.name) in edges]
    final_terms = []
    for c in top_children:
        term = gate_int_by_bool(node_sleep_in[c.name], node_sleep_ub[c.name], edges[(battery.name, c.name)], f"top_term_{c.name}")
        final_terms.append(term)

    model.Add(sum(final_terms) <= int(max_sleep * SCALE))

# 💡 원본의 병렬해 탐색 함수 수정
def find_all_load_distributions(base_solution, candidate_ics, loads, battery, constraints, viz_func, check_func):
    """
    (최종 개선 로직) 대표 해를 기반으로, exclusive_path와 exclusive_supplier 제약조건을
    위반하지 않으면서 부하를 재분배하여 가능한 모든 유효한 병렬해를 탐색합니다.
    """
    search_settings = constraints.get('parallel_search_settings', {})
    if not search_settings.get('enabled', False):
        print("\n👑 --- 병렬 해 탐색 비활성화됨 --- 👑")
        if check_func(base_solution, candidate_ics, loads, battery, constraints):
            viz_func(base_solution, candidate_ics, loads, battery, constraints, solution_index=1)
        return

    print("\n\n👑 --- 최종 단계: 모든 부하 분배 조합 탐색 --- 👑")
    max_solutions = search_settings.get('max_solutions_to_generate', 500)

    candidate_ics_map = {ic.name: ic for ic in candidate_ics}
    child_to_parent = {c: p for p, c in base_solution['active_edges']}
    parent_to_children = defaultdict(list)
    for p, c in base_solution['active_edges']:
        parent_to_children[p].append(c)

    # --- 👇 [핵심 수정] Exclusive Path와 Supplier에 속한 IC와 부하 모두 식별 ---
    exclusive_ics = set()
    exclusive_loads = set()
    for load in loads:
        if load.independent_rail_type == 'exclusive_path':
            current_node_name = load.name
            exclusive_loads.add(current_node_name)
            # 전체 경로를 추적
            while current_node_name in child_to_parent:
                parent_name = child_to_parent[current_node_name]
                if parent_name == battery.name:
                    break
                exclusive_ics.add(parent_name)
                current_node_name = parent_name
        elif load.independent_rail_type == 'exclusive_supplier':
            # 부하와 직계 부모 IC만 식별
            parent_name = child_to_parent.get(load.name)
            if parent_name and parent_name in candidate_ics_map:
                exclusive_loads.add(load.name)
                exclusive_ics.add(parent_name)
    # --- 수정 완료 ---

    ic_type_to_instances = defaultdict(list)
    for ic_name in base_solution['used_ic_names']:
        ic = candidate_ics_map.get(ic_name)
        # Exclusive 제약에 걸린 IC는 재분배 그룹에서 제외
        if ic and ic.name not in exclusive_ics:
            ic_type = f"📦 {ic.name.split('@')[0]} ({ic.vout:.1f}Vout)"
            ic_type_to_instances[ic_type].append(ic)

    target_group = None
    for ic_type, instances in ic_type_to_instances.items():
        if len(instances) > 1:
            total_load_pool = set()
            for inst in instances:
                # Exclusive 부하가 아닌 일반 부하만 재분배 대상에 추가
                loads_for_inst = [c for c in parent_to_children.get(inst.name, []) if c not in exclusive_loads]
                total_load_pool.update(loads_for_inst)

            if total_load_pool:
                target_group = {
                    'instances': [inst.name for inst in instances],
                    'load_pool': list(total_load_pool)
                }
                break

    if not target_group:
        print("\n -> 이 해답에는 생성할 병렬해가 없습니다.")
        # 대표해는 여전히 유효하므로 검증하고 시각화
        if check_func(base_solution, candidate_ics, loads, battery, constraints):
            viz_func(base_solution, candidate_ics, loads, battery, constraints, solution_index=1)
        return

    def find_partitions(items, num_bins):
        if not items:
            yield [[] for _ in range(num_bins)]
            return
        first = items[0]
        rest = items[1:]
        for p in find_partitions(rest, num_bins):
            for i in range(num_bins):
                yield p[:i] + [[first] + p[i]] + p[i+1:]
    
    valid_solutions = []
    seen_partitions = set()
    num_instances = len(target_group['instances'])
    load_pool = target_group['load_pool']
    solution_count = 0

    # 고정된 엣지 (Exclusive 엣지 및 재분배와 관련 없는 모든 엣지)
    fixed_edges = [edge for edge in base_solution['active_edges'] if edge[0] not in target_group['instances']]

    for p in find_partitions(load_pool, num_instances):
        if solution_count >= max_solutions:
            print(f"\n⚠️ 경고: 병렬 해 조합이 너무 많아 {max_solutions}개에서 탐색을 중단합니다.")
            break
        
        if len(p) != num_instances:
            continue

        canonical_partition = tuple(sorted([tuple(sorted(sublist)) for sublist in p]))
        if canonical_partition in seen_partitions:
            continue
        seen_partitions.add(canonical_partition)
        
        new_edges = list(fixed_edges)
        for i, instance_name in enumerate(target_group['instances']):
            for load_name in p[i]:
                new_edges.append((instance_name, load_name))
        
        new_solution = {"used_ic_names": base_solution['used_ic_names'], "active_edges": new_edges, "cost": base_solution['cost']}
        
        if check_func(new_solution, candidate_ics, loads, battery, constraints):
            valid_solutions.append(new_solution)
        solution_count += 1
    
    if not valid_solutions and check_func(base_solution, candidate_ics, loads, battery, constraints):
        # 만약 재분배 후 유효한 해가 하나도 없으면, 원본 대표해라도 결과에 포함
        print("\n -> 생성된 병렬해가 모두 유효하지 않아, 원본 대표해를 사용합니다.")
        valid_solutions.append(base_solution)

    print(f"\n✅ 총 {len(valid_solutions)}개의 유효한 병렬해 구조를 찾았습니다.")
    for i, solution in enumerate(valid_solutions):
        print(f"\n--- [병렬해 #{i+1}] ---")
        viz_func(solution, candidate_ics, loads, battery, constraints, solution_index=i+1)
