# pocat_core.py
import json
import copy
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from ortools.sat.python import cp_model

from common.pocat_classes import Battery, Load, PowerIC, LDO, BuckConverter
# 순환 참조를 피하기 위해 함수를 직접 임포트하지 않고, main에서 넘겨받도록 구조 변경
# from pocat_visualizer import check_solution_validity, print_and_visualize_one_solution

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
    SCALE = 1_000_000
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
        for p_ic_name, j_edge_var in j_parents:
            for q_ic_name, k_edge_var in k_parents:
                if p_ic_name == q_ic_name:
                    model.AddBoolOr([j_edge_var.Not(), k_edge_var.Not()])
    """전원 시퀀스(동일 부모 금지, 시간적 선후 관계) 제약 조건을 추가합니다
    if 'power_sequences' not in constraints or not constraints['power_sequences']:
        return
        
    is_ancestor = {
        (p, c): model.NewBoolVar(f'anc_{p}_to_{c}')
        for p in node_names for c in node_names if p != c
    }
    for p, c in edges:
        model.AddImplication(edges[p, c], is_ancestor[p, c])
    for a in node_names:
        for b in ic_names:
            for c in node_names:
                if a == b or b == c or a == c: continue
                model.AddBoolOr([is_ancestor[a, b].Not(), is_ancestor[b, c].Not(), is_ancestor[a, c]])
    
    parent_ic_vars = defaultdict(list)
    for load in loads:
        for p_ic in candidate_ics:
            if (p_ic.name, load.name) in edges:
                parent_ic_vars[load.name].append((p_ic.name, edges[p_ic.name, load.name]))

    for seq in constraints['power_sequences']:
        if seq.get('f') != 1: continue
        j_name, k_name = seq['j'], seq['k']
        for p in candidate_ics:
            if (p.name, j_name) in edges and (p.name, k_name) in edges:
                model.Add(edges[p.name, j_name] + edges[p.name, k_name] <= 1)
        for p_j_name, j_edge in parent_ic_vars[j_name]:
            for p_k_name, k_edge in parent_ic_vars[k_name]:
                if p_j_name == p_k_name: continue
                model.Add(is_ancestor[p_k_name, p_j_name] == 0).OnlyEnforceIf([j_edge, k_edge])
    """            

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

    is_always_on_path = add_always_on_constraints(model, all_nodes, loads, candidate_ics, edges)
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
def add_always_on_constraints(model, all_nodes, loads, candidate_ics, edges):
    all_ic_and_load_nodes = candidate_ics + loads
    is_always_on_path = {node.name: model.NewBoolVar(f"is_ao_{node.name}") for node in all_nodes}
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
    for p in candidate_ics:
        chs = [c for c in all_ic_and_load_nodes if (p.name, c.name) in edges]
        for i in range(len(chs) - 1):
            for j in range(i + 1, len(chs)):
                c1, c2 = chs[i], chs[j]
                model.Add(is_always_on_path[c1.name] == is_always_on_path[c2.name]).OnlyEnforceIf([edges[(p.name, c1.name)], edges[(p.name, c2.name)]])
    return is_always_on_path


def add_sleep_current_constraints(model, battery, candidate_ics, loads, constraints, edges, is_always_on_path):
    """
    Sleep-current constraint (battery viewpoint):
    - AO 레일: Iop 반영
    - 비-AO '탑레벨'(배터리 직결) 레일: Iq 반영
    - AO 부하의 sleep 전류를 상위로 전파
    - LDO: I_in = I_out
    - Buck: q*I_in = p*I_out  (p/q ≈ Vout / (Vin * eff_guess))
    - 모든 곱은 Bool 게이팅/정수비로 선형화
    """

    SCALE = 1_000_000
    max_sleep = constraints.get('max_sleep_current', 0.0)
    if max_sleep <= 0:
        return



    # ---------------- helpers ----------------
    def bool_and(a, b, name):
        """w = a AND b (동치)"""
        w = model.NewBoolVar(name)
        model.Add(w <= a)
        model.Add(w <= b)
        model.Add(w >= a + b - 1)
        return w

    def gate_const_by_bool(const_int, b, name):
        """y = const if b else 0"""
        y = model.NewIntVar(0, max(0, const_int), name)
        model.Add(y == const_int).OnlyEnforceIf(b)
        model.Add(y == 0).OnlyEnforceIf(b.Not())
        return y

    def gate_int_by_bool(x, ub, b, name):
        """y = x if b else 0  (x: IntVar, ub: 상한)"""
        y = model.NewIntVar(0, max(0, ub), name)
        model.Add(y == x).OnlyEnforceIf(b)
        model.Add(y == 0).OnlyEnforceIf(b.Not())
        return y

    # 넉넉한 상한(UB) 계산
    total_load_sleep = sum(max(0, int(ld.current_sleep * SCALE)) for ld in loads)
    total_ic_self = sum(max(0, int(max(ic.operating_current, ic.quiescent_current) * SCALE)) for ic in candidate_ics)
    NODE_UB = total_load_sleep + total_ic_self + 1

    # 각 노드 "입력핀에서 요구하는 슬립전류" 변수 미리 생성
    node_sleep_in = {}      # name -> IntVar
    node_sleep_ub = {}      # name -> int(UB)

    # Loads: AO일 때 고정값, 아니면 0
    for ld in loads:
        const_val = max(0, int(ld.current_sleep * SCALE))
        v = model.NewIntVar(0, const_val, f"sleep_in_{ld.name}")
        model.Add(v == const_val).OnlyEnforceIf(is_always_on_path[ld.name])
        model.Add(v == 0).OnlyEnforceIf(is_always_on_path[ld.name].Not())
        node_sleep_in[ld.name] = v
        node_sleep_ub[ld.name] = const_val

    # IC들: 우선 빈 변수를 만들어 두고, 아래에서 등식으로 정의
    for ic in candidate_ics:
        node_sleep_in[ic.name] = model.NewIntVar(0, NODE_UB, f"sleep_in_{ic.name}")
        node_sleep_ub[ic.name] = NODE_UB

    # IC별 제약 구성
    for ic in candidate_ics:
        ao_ic = is_always_on_path[ic.name]
        top_edge = edges.get((battery.name, ic.name), None)

        # (A) 자기소모: AO면 Iop, 비-AO & top이면 Iq, 그 외 0  (세 경우가 딱 한 개만 참)
        iop = max(0, int(ic.operating_current * SCALE))
        iq  = max(0, int(ic.quiescent_current * SCALE))
        ic_self = model.NewIntVar(0, max(iop, iq), f"sleep_self_{ic.name}")

        non_ao = model.NewBoolVar(f"non_ao_{ic.name}")
        model.Add(non_ao + ao_ic == 1)

        if top_edge is not None:
            # b1 := ao_ic
            b1 = ao_ic
            # b2 := (non_ao AND top_edge)
            b2 = bool_and(non_ao, top_edge, f"non_ao_top_{ic.name}")
            # b3 := (non_ao AND NOT top_edge)
            not_top = model.NewBoolVar(f"not_top_{ic.name}")
            model.Add(not_top + top_edge == 1)
            b3 = bool_and(non_ao, not_top, f"non_ao_not_top_{ic.name}")

            # 세 경우가 정확히 하나만 성립
            model.Add(b1 + b2 + b3 == 1)

            model.Add(ic_self == iop).OnlyEnforceIf(b1)
            model.Add(ic_self == iq ).OnlyEnforceIf(b2)
            model.Add(ic_self == 0  ).OnlyEnforceIf(b3)
        else:
            # 배터리 직결이 아닌 경우: AO면 Iop, 아니면 0
            model.Add(ic_self == iop).OnlyEnforceIf(ao_ic)
            model.Add(ic_self == 0  ).OnlyEnforceIf(ao_ic.Not())

        # (B) 자식 요구 전류 합산 (AO 자식만, 엣지 선택 시만 반영)
        children = [c for c in (candidate_ics + loads) if (ic.name, c.name) in edges]
        child_terms = []
        ub_sum = 0
        for c in children:
            edge_ic_c = edges[(ic.name, c.name)]
            use_c = bool_and(edge_ic_c, is_always_on_path[c.name], f"use_sleep_{ic.name}__{c.name}")
            ub_c = node_sleep_ub[c.name]
            term = gate_int_by_bool(node_sleep_in[c.name], ub_c, use_c, f"sleep_term_{ic.name}__{c.name}")
            child_terms.append(term)
            ub_sum += ub_c

        children_out = model.NewIntVar(0, max(0, ub_sum), f"sleep_out_{ic.name}")
        model.Add(children_out == (sum(child_terms) if child_terms else 0))

        # (C) 입력측 변환: LDO=1배, Buck=p/q
        in_for_children = model.NewIntVar(0, NODE_UB, f"sleep_children_in_{ic.name}")
        if isinstance(ic, LDO):
            model.Add(in_for_children == children_out)
        elif isinstance(ic, BuckConverter):
            # I_in = I_out * Vout/(Vin*eff_guess)  → q*I_in = p*I_out
            # Vin 후보가 고정 인스턴스(12V or 하위 레벨)로 들어온다는 전제

            # 보수적 슬립 효율 추정
            eff_sleep = getattr(ic,'eff_sleep',None)
            if not eff_sleep or eff_sleep <=0:
                eff_sleep = constraints.get('sleep_efficiency_guess',0.35)
            # 너무 과격/후한 값을 방지하기 위한 안전 범위
            eff_sleep = max(0.05,min(eff_sleep,0.85))

            # 1) ic.vin 있으면 그걸 쓰고, 없으면 배터리의 최저전압을 씀
            vin_ref = getattr(ic, 'vin', 0.0) or battery.voltage_min
            # 2) 최종적으로 '가능한 가장 낮은' Vin을 선택 (보수적)
            vin_ref = min(vin_ref, battery.voltage_min)
            # 3) 분모에 들어갈 V_in * η (효율) 계산. 0으로 나눔 방지용 최소치 포함
            vin_eff = max(1e-6, vin_ref * eff_sleep)

            vout = max(0.0, ic.vout)
            p = max(1, int(round(vout    * 1000)))   # 정수화
            q = max(1, int(round(vin_eff * 1000)))
            model.Add(in_for_children * q == children_out * p)
        else:
            model.Add(in_for_children == children_out)  # 안전 기본값

        # (D) 총 입력 = 자기소모 + 자식 공급을 위한 입력
        total_in = model.NewIntVar(0, NODE_UB, f"sleep_total_in_{ic.name}")
        model.Add(total_in == ic_self + in_for_children)
        model.Add(node_sleep_in[ic.name] == total_in)

    # (E) 배터리 관점 슬립전류: 배터리 직결 노드만 합산
    top_children = [c for c in (candidate_ics + loads) if (battery.name, c.name) in edges]
    final_terms = []
    for c in top_children:
        edge_batt_c = edges[(battery.name, c.name)]
        if isinstance(c, Load):
            # 안전하게 AO도 함께 게이팅 (실제로는 load 변수 내부에서 0/const로 처리됨)
            use_top = bool_and(edge_batt_c, is_always_on_path[c.name], f"top_use_{c.name}")
            const_val = node_sleep_ub[c.name]
            final_terms.append(gate_const_by_bool(const_val, use_top, f"top_term_{c.name}"))
        else:
            final_terms.append(gate_int_by_bool(node_sleep_in[c.name], node_sleep_ub[c.name], edge_batt_c, f"top_term_{c.name}"))

    model.Add(sum(final_terms) <= int(max_sleep * SCALE))

# 💡 원본의 병렬해 탐색 함수 수정
def find_all_load_distributions(base_solution, candidate_ics, loads, battery, constraints, viz_func, check_func):
    """
    대표 해를 기반으로, 부하를 재분배하여 가능한 모든 유효한 병렬해를 탐색합니다.
    config.json의 설정에 따라 실행 여부와 최대 탐색 개수가 제어됩니다.
    """
    # 설정 가져오기 (없으면 기본값 사용)
    search_settings = constraints.get('parallel_search_settings', {})
    if not search_settings.get('enabled', False):
        print("\n👑 --- 병렬 해 탐색 비활성화됨 --- 👑")
        # 비활성화 시, 대표 해만 검증하고 시각화
        if check_func(base_solution, candidate_ics, loads, battery, constraints):
            viz_func(base_solution, candidate_ics, loads, battery, constraints, solution_index=1)
        return

    print("\n\n👑 --- 최종 단계: 모든 부하 분배 조합 탐색 --- 👑")
    max_solutions = search_settings.get('max_solutions_to_generate', 500) # 최대 탐색 개수 제한

    candidate_ics_map = {ic.name: ic for ic in candidate_ics}
    ic_type_to_instances = defaultdict(list)
    for ic_name in base_solution['used_ic_names']:
        ic = candidate_ics_map.get(ic_name)
        if ic:
            ic_type = f"📦 {ic.name.split('@')[0]} ({ic.vout:.1f}Vout)"
            ic_type_to_instances[ic_type].append(ic)

    instance_to_children = defaultdict(set)
    for p, c in base_solution['active_edges']:
        if p in candidate_ics_map:
            instance_to_children[p].add(c)
    
    target_group = None
    for ic_type, instances in ic_type_to_instances.items():
        if len(instances) > 1:
            total_load_pool = set()
            for inst in instances:
                total_load_pool.update(instance_to_children[inst.name])
            if total_load_pool:
                target_group = {
                    'instances': [inst.name for inst in instances],
                    'load_pool': list(total_load_pool)
                }
                break

    if not target_group:
        print("\n -> 이 해답에는 생성할 병렬해가 없습니다.")
        if check_func(base_solution, candidate_ics, loads, battery, constraints):
            viz_func(base_solution, candidate_ics, loads, battery, constraints, solution_index=1)
        return

    def find_partitions(items, num_bins):
        if not items:
            yield [[] for _ in range(num_bins)]
        else:
            for partition in find_partitions(items[1:], num_bins):
                for i in range(num_bins):
                    yield partition[:i] + [[items[0]] + partition[i]] + partition[i+1:]
                if num_bins > len(partition):
                    yield partition + [[items[0]]]

    valid_solutions = []
    seen_partitions = set()
    num_instances = len(target_group['instances'])
    load_pool = target_group['load_pool']
    solution_count = 0

    for p in find_partitions(load_pool, num_instances):
        if solution_count >= max_solutions:
            print(f"\n⚠️ 경고: 병렬 해 조합이 너무 많아 {max_solutions}개에서 탐색을 중단합니다.")
            break
            
        if len(p) == num_instances:
            canonical_partition = tuple(sorted([tuple(sorted(sublist)) for sublist in p]))
            if canonical_partition in seen_partitions:
                continue
            seen_partitions.add(canonical_partition)
            new_edges = [edge for edge in base_solution['active_edges'] if edge[0] not in target_group['instances']]
            for i, instance_name in enumerate(target_group['instances']):
                for load_name in p[i]:
                    new_edges.append((instance_name, load_name))
            new_solution = {"used_ic_names": base_solution['used_ic_names'], "active_edges": new_edges, "cost": base_solution['cost']}
            if check_func(new_solution, candidate_ics, loads, battery, constraints):
                valid_solutions.append(new_solution)
        solution_count += 1
    print(f"\n✅ 총 {len(valid_solutions)}개의 유효한 병렬해 구조를 찾았습니다.")
    for i, solution in enumerate(valid_solutions):
        print(f"\n--- [병렬해 #{i+1}] ---")
        viz_func(solution, candidate_ics, loads, battery, constraints, solution_index=i+1)