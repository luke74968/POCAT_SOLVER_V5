# common/pocat_visualizer.py
from collections import defaultdict
from graphviz import Digraph
from .pocat_classes import LDO, BuckConverter # 필요한 클래스 임포트

def check_solution_validity(solution, candidate_ics, loads, battery, constraints):
    """주어진 해답이 모든 제약조건을 만족하는지 수동으로 검증하는 함수"""
    print("  -> 검증 중...", end="")
    candidate_ics_map = {ic.name: ic for ic in candidate_ics}
    loads_map = {load.name: load for load in loads}
    parent_to_children = defaultdict(list)
    child_to_parent = {c: p for p, c in solution['active_edges']}
    for p, c in solution['active_edges']: parent_to_children[p].append(c)
    
    # 1. 전류 한계 검증
    for p_name, children_names in parent_to_children.items():
        if p_name not in candidate_ics_map: continue
        parent_ic = candidate_ics_map[p_name]
        actual_i_out = 0
        for c_name in children_names:
            if c_name in loads_map: actual_i_out += loads_map[c_name].current_active
            elif c_name in candidate_ics_map:
                child_ic = candidate_ics_map[c_name]
                child_children = parent_to_children.get(c_name, [])
                child_i_out = sum(loads_map[gc_name].current_active for gc_name in child_children if gc_name in loads_map)
                actual_i_out += child_ic.calculate_input_current(child_ic.vin, child_i_out)
        
        if actual_i_out > parent_ic.i_limit:
            print(f" -> ❌ 열-전류 한계 위반 ({p_name})")
            return False
        if actual_i_out > parent_ic.original_i_limit * (1 - constraints.get('current_margin', 0.1)):
            print(f" -> ❌ 전기적 전류 마진 위반 ({p_name})")
            return False

    # 2. Independent Rail 검증 (개선된 버전)
    for load in loads:
        rail_type = load.independent_rail_type
        if not rail_type: continue
        
        parent_name = child_to_parent.get(load.name)
        if not parent_name: continue

        if rail_type == 'exclusive_supplier':
            if parent_name in parent_to_children and len(parent_to_children[parent_name]) > 1:
                print(f" -> ❌ Independent Rail 위반 ({parent_name}이 exclusive_supplier 규칙 위반)")
                return False
        elif rail_type == 'exclusive_path':
            current_node_name = load.name
            while current_node_name in child_to_parent:
                parent_name = child_to_parent[current_node_name]
                if parent_name == battery.name:
                    break
                
                if parent_name in parent_to_children and len(parent_to_children[parent_name]) > 1:
                    print(f" -> ❌ Independent Rail 위반 ({parent_name}가 exclusive_path 규칙 위반)")
                    return False
                current_node_name = parent_name
            
    # 3. Power Sequence 검증
    def is_ancestor(ancestor_candidate, node, parent_map):
        current_node = node
        while current_node in parent_map:
            parent = parent_map[current_node]
            if parent == ancestor_candidate:
                return True
            current_node = parent
        return False
    
    for rule in constraints.get('power_sequences', []):
        if rule.get('f') != 1:
            continue
        
        j_name, k_name = rule['j'], rule['k']
        j_parent = child_to_parent.get(j_name)
        k_parent = child_to_parent.get(k_name)

        if not j_parent or not k_parent:
            continue
            
        if j_parent == k_parent:
            print(f" -> ❌ Power Sequence 위반 ({j_name}와 {k_name}가 동일 부모 {j_parent} 공유)")
            return False
        
        if is_ancestor(ancestor_candidate=k_parent, node=j_parent, parent_map=child_to_parent):
            print(f" -> ❌ Power Sequence 위반 ({k_parent}가 {j_parent}의 전원 경로 상위에 있음)")
            return False

    print(" -> ✅ 유효")
    return True

# --- 👇 [핵심 수정 1] 함수 시그니처에 새로운 인자들 추가 ---
def visualize_tree(solution, candidate_ics, loads, battery, constraints, junction_temps, 
                   actual_i_ins, actual_i_outs, actual_i_ins_sleep, actual_i_outs_sleep, ic_self_consumption_sleep,
                   total_active_power, total_active_current, total_sleep_current, always_on_nodes):
    """솔루션 시각화 함수 (개선된 라벨링)"""
    dot = Digraph(comment=f"Power Tree - Cost ${solution['cost']:.2f}", format='png')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')

    margin_info = f"Current Margin: {constraints.get('current_margin', 0)*100:.0f}%"
    temp_info = f"Ambient Temp: {constraints.get('ambient_temperature', 25)}°C"
    dot.attr(rankdir='LR', label=f"{margin_info}\n{temp_info}\n\nSolution Cost: ${solution['cost']:.2f}", labelloc='t', fontname='Arial')
    
    battery_label = (f"🔋 {battery.name}\n\n"
        f"Total Active Power: {total_active_power:.2f} W\n"
        f"Total Active Current: {total_active_current * 1000:.1f} mA\n"
        f"Total Sleep Current: {total_sleep_current * 1000000:,.1f} µA")
    dot.node(battery.name, battery_label, shape='Mdiamond', color='darkgreen', fillcolor='white')
    
    used_ics_map = {ic.name: ic for ic in candidate_ics if ic.name in solution['used_ic_names']}
    for ic_name, ic in used_ics_map.items():
        calculated_tj = junction_temps.get(ic_name, 0)
        i_in_active = actual_i_ins.get(ic_name, 0)
        i_out_active = actual_i_outs.get(ic_name, 0)
        i_in_sleep = actual_i_ins_sleep.get(ic_name, 0)
        i_out_sleep = actual_i_outs_sleep.get(ic_name, 0)
        i_self_sleep = ic_self_consumption_sleep.get(ic_name, 0)
        
        thermal_margin = ic.t_junction_max - calculated_tj
        
        node_color = 'blue'
        fill_color = 'white' if ic_name in always_on_nodes else 'lightgrey'
        if thermal_margin < 10: node_color = 'red'
        elif thermal_margin < 25: node_color = 'orange'
        
        # --- 👇 [핵심 수정 2] 라벨 표기 방식 업데이트 ---
        label = (f"📦 {ic.name.split('@')[0]}\n\n"
            f"Vin: {ic.vin:.2f}V, Vout: {ic.vout:.2f}V\n"
            f"Iin: {i_in_active*1000:.1f}mA (Active) | {i_in_sleep*1000000:,.1f}µA (Sleep)\n"
            f"Iout: {i_out_active*1000:.1f}mA (Active) | {i_out_sleep*1000000:,.1f}µA (Sleep)\n"
            f"I_self: {ic.operating_current*1000:.1f}mA (Active) | {i_self_sleep*1000000:,.1f}µA (Sleep)\n"
            f"Tj: {calculated_tj:.1f}°C (Max: {ic.t_junction_max}°C)\n"
            f"Cost: ${ic.cost:.2f}")
        # --- 수정 완료 ---
        dot.node(ic_name, label, color=node_color, fillcolor=fill_color, penwidth='3')

    sequenced_loads = set()
    if 'power_sequences' in constraints:
        for seq in constraints['power_sequences']:
            sequenced_loads.add(seq['j']); sequenced_loads.add(seq['k'])
            
    for load in loads:
        fill_color = 'white' if load.name in always_on_nodes else 'lightgrey'
        label = f"💡 {load.name}\nActive: {load.voltage_typical}V | {load.current_active*1000:.1f}mA\n"
        if load.current_sleep > 0: label += f"Sleep: {load.current_sleep * 1000000:,.1f}µA\n"
        
        conditions = []
        if load.independent_rail_type:
            conditions.append(f"🔒 {load.independent_rail_type}")
        if load.name in sequenced_loads:
            conditions.append("⛓️ Sequence")
        if conditions:
            label += " ".join(conditions)
            
        penwidth = '1'
        if load.always_on_in_sleep: penwidth = '3'
        dot.node(load.name, label, color='dimgray', fillcolor=fill_color, penwidth=penwidth)
        
    for p_name, c_name in solution['active_edges']:
        dot.edge(p_name, c_name)
    print(f"\n🖼️  Generating diagram for solution with cost ${solution['cost']:.2f}...")
    return dot


def print_and_visualize_one_solution(solution, candidate_ics, loads, battery, constraints, solution_index=0):
    """
    하나의 솔루션을 콘솔에 출력하고, 다이어그램으로 시각화하여 저장합니다.
    """
    candidate_ics_map = {ic.name: ic for ic in candidate_ics}
    loads_map = {load.name: load for load in loads}
    print(f"\n{'='*20} 솔루션 (비용: ${solution['cost']:.2f}) {'='*20}")
    
    used_ic_objects = [ic for ic in candidate_ics if ic.name in solution['used_ic_names']]
    actual_current_draw = {load.name: load.current_active for load in loads}
    sleep_current_draw = {load.name: load.current_sleep for load in loads}
    
    # --- 👇 [핵심 수정 3] 계산 결과를 저장할 딕셔너리 추가 ---
    junction_temps, actual_i_ins, actual_i_outs = {}, {}, {}
    actual_i_ins_sleep, actual_i_outs_sleep, ic_self_consumption_sleep = {}, {}, {}
    # --- 수정 완료 ---

    processed_ics = set()
    child_to_parent = {c: p for p, c in solution['active_edges']}

    always_on_nodes = {l.name for l in loads if l.always_on_in_sleep}
    nodes_to_process = list(always_on_nodes)
    while nodes_to_process:
        node = nodes_to_process.pop(0)
        if node in child_to_parent:
            parent = child_to_parent[node]
            if parent not in always_on_nodes:
                always_on_nodes.add(parent)
                nodes_to_process.append(parent)

    while len(processed_ics) < len(used_ic_objects):
        progress_made = False
        
        for ic in used_ic_objects:
            if ic.name in processed_ics: 
                continue
            
            children_names = [c for p, c in solution['active_edges'] if p == ic.name]
            
            if all(c in loads_map or c in processed_ics for c in children_names):
                # Active current calculation
                total_i_out_active = sum(actual_current_draw.get(c, 0) for c in children_names)
                actual_i_outs[ic.name] = total_i_out_active
                i_in_active = ic.calculate_input_current(vin=ic.vin, i_out=total_i_out_active)
                actual_current_draw[ic.name] = i_in_active
                actual_i_ins[ic.name] = i_in_active
                power_loss = ic.calculate_power_loss(vin=ic.vin, i_out=total_i_out_active)
                ambient_temp = constraints.get('ambient_temperature', 25)
                junction_temps[ic.name] = ambient_temp + (power_loss * ic.theta_ja)
                
                # Sleep current calculation
                i_in_sleep = 0
                ic_self_sleep = 0
                total_i_out_sleep = 0
                parent_name = child_to_parent.get(ic.name)
                
                if ic.name in always_on_nodes:
                    # Case 1: IC가 AO 경로에 포함된 경우 (Iop 소모)
                    total_i_out_sleep = sum(sleep_current_draw.get(c, 0) for c in children_names)
                    ic_self_sleep = ic.operating_current
                    
                    if isinstance(ic, LDO):
                        i_in_sleep = total_i_out_sleep + ic_self_sleep
                    elif isinstance(ic, BuckConverter):
                        if ic.vin > 0:
                            eff_sleep = constraints.get('sleep_efficiency_guess', 0.35)
                            p_out_sleep = ic.vout * total_i_out_sleep
                            p_in_sleep = p_out_sleep / eff_sleep if p_out_sleep > 0 else 0
                            i_in_sleep = (p_in_sleep / ic.vin) + ic_self_sleep
                
                elif parent_name in always_on_nodes:
                    # Case 2: IC는 비-AO지만, 부모가 AO인 경우 (I_shut 또는 Iq 소모)
                    if ic.shutdown_current is not None and ic.shutdown_current > 0:
                        ic_self_sleep = ic.shutdown_current
                    else:
                        ic_self_sleep = ic.quiescent_current
                    i_in_sleep = ic_self_sleep # 비-AO IC는 출력이 없으므로
                
                # --- 👇 [핵심 수정 4] 계산된 sleep 값들을 딕셔너리에 저장 ---
                actual_i_ins_sleep[ic.name] = i_in_sleep
                actual_i_outs_sleep[ic.name] = total_i_out_sleep
                ic_self_consumption_sleep[ic.name] = ic_self_sleep
                # --- 수정 완료 ---
                sleep_current_draw[ic.name] = i_in_sleep

                processed_ics.add(ic.name)
                progress_made = True

        if not progress_made and len(used_ic_objects) > 0 and len(processed_ics) < len(used_ic_objects):
            print("\n⚠️ 경고: Power Tree에서 순환 참조가 발견되어 계산을 중단합니다.")
            unprocessed_ics = [ic.name for ic in used_ic_objects if ic.name not in processed_ics]
            if unprocessed_ics:
                 print(f"         (미처리 IC: {unprocessed_ics})")
            break

    primary_ics = [c_name for p_name, c_name in solution['active_edges'] if p_name == battery.name]
    total_active_current = sum(actual_i_ins.get(ic_name, 0) for ic_name in primary_ics)
    total_sleep_current = sum(actual_i_ins_sleep.get(ic_name, 0) for ic_name in primary_ics)
    battery_avg_voltage = (battery.voltage_min + battery.voltage_max) / 2
    total_active_power = battery_avg_voltage * total_active_current
    
    print(f"   - 시스템 전체 슬립 전류: {total_sleep_current * 1000:.4f} mA")
    print("\n--- Power Tree 구조 ---")
    
    tree_topology = defaultdict(list)
    for p, c in solution['active_edges']: 
        tree_topology[p].append(c)
        
    def format_node_name(name, show_instance_num=False):
        # ... (이 함수는 수정 없음)
        if name in candidate_ics_map:
            ic = candidate_ics_map[name]
            base_name = f"📦 {ic.name.split('@')[0]} ({ic.vout:.1f}Vout)"
            if show_instance_num and '_copy' in ic.name: 
                return f"{base_name} [#{ic.name.split('_copy')[-1]}]"
            return base_name
        elif name in loads_map: 
            return f"💡 {name}"
        elif name == battery.name: 
            return f"🔋 {name}"
        return name
        
    def print_instance_tree(parent_name, prefix=""):
        # ... (이 함수는 수정 없음)
        children = sorted(tree_topology.get(parent_name, []))
        for i, child_name in enumerate(children):
            is_last = (i == len(children) - 1)
            connector = "└── " if is_last else "├── "
            print(prefix + connector + format_node_name(child_name, show_instance_num=True))
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_instance_tree(child_name, new_prefix)
            
    print(format_node_name(battery.name))
    root_children = sorted(tree_topology.get(battery.name, []))
    for i, child_instance_name in enumerate(root_children):
        is_last = (i == len(root_children) - 1)
        connector = "└── " if is_last else "├── "
        print(connector + format_node_name(child_instance_name, show_instance_num=True))
        new_prefix = "    " if is_last else "│   "
        print_instance_tree(child_instance_name, new_prefix)
    
    # --- 👇 [핵심 수정 5] visualize_tree 함수 호출 시 새로운 인자들 전달 ---
    dot_graph = visualize_tree(
        solution, candidate_ics, loads, battery, constraints,
        junction_temps, actual_i_ins, actual_i_outs, actual_i_ins_sleep,
        actual_i_outs_sleep, ic_self_consumption_sleep, total_active_power, 
        total_active_current, total_sleep_current, always_on_nodes
    )
    # --- 수정 완료 ---
    
    output_filename = f'solution_{solution_index}_cost_{solution["cost"]:.2f}'
    dot_graph.render(output_filename, view=False, cleanup=True, format='png')
    
    print(f"\n✅ 다이어그램을 '{output_filename}.png' 파일로 저장했습니다.")