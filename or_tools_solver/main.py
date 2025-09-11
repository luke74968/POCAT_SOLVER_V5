# or_tools_solver/main.py

import json
import sys
from dataclasses import asdict
from ortools.sat.python import cp_model

# 💡 1. 여기서 'load_configuration'을 지우고, 아래에 새로운 import를 추가합니다.
from .pocat_core import (
    expand_ic_instances, create_solver_model,
    find_all_load_distributions
)
from common.pocat_visualizer import (
    check_solution_validity, print_and_visualize_one_solution
)
from common.pocat_preprocess import prune_dominated_ic_instances
# 💡 2. 새로운 공용 로더에서 함수를 가져옵니다.
from common.config_loader import load_configuration_from_file

def main():
    """메인 실행 함수"""
    # 💡 2. 명령행에서 파일 경로를 확인하고 가져옵니다.
    if len(sys.argv) < 2:
        print("오류: 설정 파일(.json)을 명령행 인자로 전달해야 합니다.")
        print("사용법: python main.py <config_filename.json>")
        return
    
    config_filename = sys.argv[1]
    print(f"📖 설정 파일 '{config_filename}' 로딩...")

    # 1. 설정 로드
    try:
        # 💡 3. 파일 내용을 읽는 대신, 공용 함수를 직접 호출하여 한 번에 로드합니다.
        battery, available_ics, loads, constraints = load_configuration_from_file(config_filename)
    except FileNotFoundError:
        print(f"오류: 설정 파일 '{config_filename}'을(를) 찾을 수 없습니다.")
        return
        
   
    # 2. 후보 IC 생성
    candidate_ics, ic_groups = expand_ic_instances(available_ics, loads, battery, constraints)
    
    # --- Dominance Pruning 단계 추가 ---
    print("\n🔪 Dominance Pruning 전처리 시작...")
    
    candidate_ics_dicts = [asdict(ic) for ic in candidate_ics]
    
    # --- 💡 함수 호출 및 반환 값 수정 ---
    # 지배 관계 맵(dominance_map)도 함께 받습니다.
    pruned_ics_dicts, dominance_map = prune_dominated_ic_instances(candidate_ics_dicts)
    
    candidate_ics_map = {ic.name: ic for ic in candidate_ics}
    
    pruned_candidate_names = {ic_dict['name'] for ic_dict in pruned_ics_dicts}
    candidate_ics = [ic for name, ic in candidate_ics_map.items() if name in pruned_candidate_names]

    original_count = len(candidate_ics_dicts)
    pruned_count = len(candidate_ics)
    print(f"   - {original_count - pruned_count}개의 지배되는 IC 인스턴스 제거 완료!")
    print(f"   - 남은 후보 IC 인스턴스: {pruned_count}개")
    # --- 💡 [핵심 수정] 최종 인스턴스 목록 로그 출력 ---
    print("\n--- ✅ 최종 후보 IC 인스턴스 목록 ---")
    for ic_dict in sorted(pruned_ics_dicts, key=lambda x: x['name']):
        print(f"   - {ic_dict['name']}")
    print("------------------------------------")
    # --- 수정 완료 ---
    # --- 💡 결과 표시 방법 개선 ---
    """
    if dominance_map:
        print("\n--- 🗑️ 제거된 IC 목록 (원인: 더 우수한 IC) ---")
        # 제거된 IC 이름을 기준으로 정렬하여 출력합니다.
        for removed_name, dominant_name in sorted(dominance_map.items()):
            print(f"   - {removed_name:<50} (by ▶️  {dominant_name})")
    """
    sanitized_ic_groups = {}
    for group_key, group_list in ic_groups.items():
        # 각 그룹 리스트에서 살아남은 IC 이름만 필터링하여 새로운 리스트 생성
        sanitized_group_list = [name for name in group_list if name in pruned_candidate_names]
        
        # 필터링 후에도 그룹에 2개 이상의 IC가 남아있으면, 새로운 딕셔너리에 추가
        if len(sanitized_group_list) > 1:
            sanitized_ic_groups[group_key] = sanitized_group_list
    # --- 결과 표시 끝 ---

    # 3. CP-SAT 모델 생성
    model, edges, ic_is_used = create_solver_model(candidate_ics, loads, battery, constraints, sanitized_ic_groups)
    
    # 4. 솔버 생성 및 탐색 시간 설정
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 300.0 # 최대 30초간 최적해 탐색
    
    # 5. 솔버 실행 (SolutionLogger 없이)
    print("\n🔍 최적의 대표 솔루션 탐색 시작...")
    status = solver.Solve(model)
    
    # 6. 결과 처리
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"\n🎉 탐색 완료! (상태: {solver.StatusName(status)})")
        
        # 탐색이 끝난 solver에서 직접 결과값을 가져와 base_solution 구성
        base_solution = {
            "score": solver.ObjectiveValue(),
            "cost": solver.ObjectiveValue() / 10000,
            "used_ic_names": {name for name, var in ic_is_used.items() if solver.Value(var)},
            "active_edges": [(p, c) for (p, c), var in edges.items() if solver.Value(var)]
        }
        
        # 대표해를 기반으로 병렬해 탐색
        all_solutions = find_all_load_distributions(
            base_solution, candidate_ics, loads, battery, constraints,
            viz_func=print_and_visualize_one_solution,
            check_func=check_solution_validity
        )
        
    else:
        print("\n❌ 유효한 솔루션을 찾지 못했습니다.")

if __name__ == "__main__":
    main()