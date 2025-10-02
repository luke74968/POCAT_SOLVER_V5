# transformer_solver/trainer.py
import torch
from tqdm import tqdm
import os
import time # 💡 시간 측정을 위해 time 모듈 추가
from datetime import datetime
from collections import defaultdict # 👈 이 줄을 추가해주세요.

import json
import logging

from common.utils.common import TimeEstimator, clip_grad_norms, unbatchify, batchify
from .model import PocatModel
from .pocat_env import PocatEnv
from common.pocat_visualizer import print_and_visualize_one_solution

from common.pocat_classes import Battery, LDO, BuckConverter, Load
from common.pocat_defs import PocatConfig, NODE_TYPE_IC, FEATURE_INDEX
from common.config_loader import load_configuration_from_file
from .pocat_env import BATTERY_NODE_IDX
from graphviz import Digraph


def update_progress(pbar, metrics):
    if pbar is None:
        return
    pbar.set_postfix({
        "Loss": f"{metrics['Loss']:.4f}",
        "Avg Cost": f"${metrics['Avg Cost']:.2f}",
        "Min Cost": f"${metrics['Min Cost']:.2f}",
        "T_Reset": f"{metrics['T_Reset']:.0f}ms",
    }, refresh=False)
    pbar.update(1)


def cal_model_size(model, log_func):
    param_count = sum(param.nelement() for param in model.parameters())
    buffer_count = sum(buffer.nelement() for buffer in model.buffers())
    log_func(f'Total number of parameters: {param_count}')
    log_func(f'Total number of buffer elements: {buffer_count}')

class PocatTrainer:
    # 💡 1. 생성자에서 device 인자를 받도록 수정
    def __init__(self, args, env: PocatEnv, device: str):
        self.args = args
        self.env = env
        self.device = device # 전달받은 device 저장

        self.result_dir = args.result_dir

        
        # 💡 2. CUDA 강제 설정 라인 삭제
        # torch.set_default_tensor_type('torch.cuda.FloatTensor') 
        
        # 💡 3. 모델을 생성 후, 지정된 device로 이동
        self.model = PocatModel(**args.model_params).to(self.device)
        cal_model_size(self.model, args.log)
        
        # 💡 float()으로 감싸서 값을 숫자로 강제 변환합니다.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(args.optimizer_params['optimizer']['lr']),
            weight_decay=float(args.optimizer_params['optimizer'].get('weight_decay', 0)),
        )
        
        if args.optimizer_params['scheduler']['name'] == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=args.optimizer_params['scheduler']['milestones'],
                gamma=args.optimizer_params['scheduler']['gamma']
            )
        else:
            raise NotImplementedError
            
        self.start_epoch = 1

        # 💡 모델 로딩 로직 추가
        if args.load_path is not None:
            args.log(f"Loading model checkpoint from: {args.load_path}")
            checkpoint = torch.load(args.load_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # 훈련을 이어서 할 경우 optimizer 상태도 불러올 수 있음
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # self.start_epoch = checkpoint['epoch'] + 1        
        self.time_estimator = TimeEstimator(log_fn=args.log)

        self.eval_batch_size = getattr(args, "eval_batch_size", 128)
        with torch.no_grad():
            self._eval_td_fixed = self.env.reset(batch_size=self.eval_batch_size).clone()
        self.best_eval_bom = float("inf")

    def run(self):
        args = self.args
        self.time_estimator.reset(self.start_epoch)
        
        if args.test_only:
            self.test()
            return

        for epoch in range(self.start_epoch, args.trainer_params['epochs'] + 1):
            args.log('=================================================================')
            
            self.model.train()
            
            total_steps = args.trainer_params['train_step']
            train_pbar = tqdm(
                total=total_steps,
                desc=f"Epoch {epoch}",
                dynamic_ncols=True,
                leave=False,
                miniters=1,
                mininterval=0.1,
                smoothing=0.1,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            )
            
            total_loss = 0.0
            total_cost = 0.0
            min_epoch_cost = float('inf') # 💡 **[변경 1]** 에포크 내 최소 비용을 기록할 변수 추가

            for step in range(1, total_steps + 1):
                self.optimizer.zero_grad()
                
                reset_start_time = time.time()
                td = self.env.reset(batch_size=args.batch_size)
                reset_time = time.time() - reset_start_time
                
                # --- 👇 [핵심 수정 1] 학습 시 데이터 확장 ---
                if args.num_pomo_samples > 1:
                    td = batchify(td, args.num_pomo_samples)
                # --- 수정 완료 ---


                model_start_time = time.time()
                # --- 👇 [핵심] log 함수를 모델에 전달 ---
                out = self.model(td, self.env, decode_type='sampling', pbar=train_pbar,
                                     status_msg=None, log_fn=args.log,
                                     log_idx=args.log_idx, log_mode=args.log_mode)
                model_time = time.time() - model_start_time
                
                bwd_start_time = time.time()
                num_starts = self.env.generator.num_loads
                reward = out["reward"].view(-1, num_starts)
                log_likelihood = out["log_likelihood"].view(-1, num_starts)
                
                advantage = reward - reward.mean(dim=0, keepdims=True)
                loss = -(advantage * log_likelihood).mean()
                loss.backward()
                
                # 그래디언트 클리핑 (옵션)
                max_norm = float(self.args.optimizer_params.get('max_grad_norm', 0))
                if max_norm > 0:
                    clip_grad_norms(self.optimizer.param_groups, max_norm=max_norm)

                # 가중치 업데이트
                self.optimizer.step()

                bwd_time = time.time() - bwd_start_time

                logging.debug(
                    "Epoch %d step %d reset=%.3fms model=%.3fms backward=%.3fms",
                    epoch,
                    step,
                    reset_time * 1000,
                    model_time * 1000,
                    bwd_time * 1000,
                )

                # 각 샘플 실행에서 찾은 최상의 보상을 가져옴
                best_reward_per_sample_run = reward.max(dim=1)[0]
                # 원본 배치 인스턴스별로 결과를 재구성
                best_reward_per_sample_run = best_reward_per_sample_run.view(args.batch_size, args.num_pomo_samples)
                # 각 인스턴스에 대해 샘플들 간의 평균 최상위 보상을 계산
                avg_of_bests = best_reward_per_sample_run.mean(dim=1)

                
                # 💡 **[변경 2]** 현재 배치의 평균 비용과 최소 비용 계산
                avg_cost = -avg_of_bests.mean().item()
                min_batch_cost = -avg_of_bests.max().item()
                min_epoch_cost = min(min_epoch_cost, min_batch_cost)


                total_loss += loss.item()
                total_cost += avg_cost
                
                update_progress(
                    train_pbar,
                    {
                        "Loss": loss.item(),
                        "Avg Cost": total_cost / step,
                        "Min Cost": min_epoch_cost,
                        "T_Reset": reset_time * 1000,
                    },
                )

            train_pbar.close()

            epoch_summary = (
                f"Epoch {epoch}/{args.trainer_params['epochs']} | "
                f"Loss {total_loss / total_steps:.4f} | "
                f"Avg Cost ${total_cost / total_steps:.2f} | "
                f"Min Cost ${min_epoch_cost:.2f}"
            )
            tqdm.write(epoch_summary)
            args.log(epoch_summary) # 에폭 종료 메시지도 로그에 기록
            val = self.evaluate(epoch)
            self.args.log(f"[Eval] Epoch {epoch} | Avg BOM ${val['avg_bom']:.2f} | Min BOM ${val['min_bom']:.2f}")

            self.scheduler.step()
            self.time_estimator.print_est_time(epoch, args.trainer_params['epochs'])
            
            if (epoch % args.trainer_params['model_save_interval'] == 0) or (epoch == args.trainer_params['epochs']):
                save_path = os.path.join(args.result_dir, f'epoch-{epoch}.pth')
                args.log(f"Saving model at epoch {epoch} to {save_path}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, save_path)

        args.log(" *** Training Done *** ")


    # ... (test, visualize_result 메소드는 기존과 동일) ...
    @torch.no_grad()
    def evaluate(self, epoch: int):
        """Greedy decode on a fixed validation set, CSV log, and save best checkpoint by avg BOM."""
        self.model.eval()
        # --- 👇 [핵심 수정 4] 평가 시 데이터 확장 ---
        eval_samples = self.args.test_num_pomo_samples
        td_eval = self._eval_td_fixed.clone()
        if eval_samples > 1:
            td_eval = batchify(td_eval, eval_samples)

        # Rebuild eval TD from the fixed snapshot (same instances every epoch)
        td_eval = self.env._reset(self._eval_td_fixed.clone())

        # Greedy decoding; reuse your model call signature
        out = self.model(
            td_eval, self.env, decode_type='greedy',
            pbar=None, status_msg="Eval",
            log_fn=self.args.log, log_idx=self.args.log_idx, log_mode=self.args.log_mode
        )

        # POMO starts: choose best per instance
        num_starts = self.env.generator.num_loads
        reward = out["reward"].view(num_starts, -1)
        best_reward_per_instance = reward.max(dim=0)[0]

        avg_bom = -best_reward_per_instance.mean().item()
        min_bom = -best_reward_per_instance.max().item()

        # CSV log
        import os, torch
        csv_path = os.path.join(self.result_dir, "val_log.csv")
        header = not os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8") as f:
            if header:
                f.write("epoch,avg_bom,min_bom,decode_type\n")
            f.write(f"{epoch},{avg_bom:.4f},{min_bom:.4f},greedy\n")

        # Save best
        if avg_bom < self.best_eval_bom:
            self.best_eval_bom = avg_bom
            save_path = os.path.join(self.result_dir, "best_cost.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, save_path)
            self.args.log(f"[Eval] ✅ New best avg_bom=${avg_bom:.2f} (min=${min_bom:.2f}) at epoch {epoch} → saved {save_path}")

        return {"avg_bom": avg_bom, "min_bom": min_bom}

    def test(self):
        self.model.eval()
        logging.info("==================== INFERENCE START ====================")

        # --- 👇 [핵심 수정 5] 테스트 시 데이터 확장 및 결과 처리 ---
        test_samples = self.args.test_num_pomo_samples
        td = self.env.reset(batch_size=1)
        if test_samples > 1:
            td = batchify(td, test_samples)
        
        pbar = tqdm(total=1, desc=f"Solving Power Tree (Mode: {self.args.decode_type}, Samples: {test_samples})")
        out = self.model(td, self.env, decode_type=self.args.decode_type, pbar=pbar, 
                         log_fn=logging.info, log_idx=self.args.log_idx, 
                         log_mode=self.args.log_mode)
        pbar.close()

        reward = out['reward']
        actions = out['actions']
        
        # 모든 샘플과 시작 노드 중에서 단 하나의 최고 결과를 선택
        best_idx = reward.argmax()
        final_cost = -reward[best_idx].item()
        best_action_sequence = actions[best_idx]

        # 최적해의 시작 노드 정보를 정확히 찾기
        num_starts = self.env.generator.num_loads
        _, start_nodes_idx = self.env.select_start_nodes(self.env.reset(batch_size=1))
        
        best_start_node_local_idx = best_idx % num_starts
        best_start_node_idx = start_nodes_idx[best_start_node_local_idx].item()
        best_start_node_name = self.env.generator.config.node_names[best_start_node_idx]
        print(f"Generated Power Tree (Best of {test_samples} samples, start: '{best_start_node_name}'), Cost: ${final_cost:.4f}")

        action_history = []
        # 💡 **[BUG FIX]** 시뮬레이션은 POMO로 확장된 배치가 아닌, 단일 인스턴스(배치 크기 1)에서 실행되어야 합니다.
        td_sim = self.env.reset(batch_size=1)

        # 첫 번째 액션은 시작 노드를 설정하는 것이며, POMO 설정에서 처리됩니다.
        # 따라서 모델이 만든 첫 *결정*부터 시뮬레이션을 시작합니다.
        td_sim.set("action", best_action_sequence[0])
        output_td = self.env.step(td_sim)
        td_sim = output_td["next"]
        
        for action_tensor in best_action_sequence[1:]:
            if td_sim["done"].all(): break
            current_head = td_sim["trajectory_head"].item()
            action_item = action_tensor.item()
            
            # 부모가 자식에게 할당될 때만 action_history에 추가합니다.
            if current_head != BATTERY_NODE_IDX:
                action_history.append((action_item, current_head))

            td_sim.set("action", action_tensor.unsqueeze(0)) # 배치 차원 추가
            output_td = self.env.step(td_sim)
            td_sim = output_td["next"]

        # --- 👇 [핵심 수정 3] 시각화 함수에 시작 노드 이름 전달 ---
        self.visualize_result(action_history, final_cost, best_start_node_name, td_sim)


    # 💡 [핵심 수정] visualize_result 메서드를 OR-Tools 수준으로 대폭 업그레이드
    def visualize_result(self, action_history, final_cost, best_start_node_name, final_td):
        if self.result_dir is None: return
        os.makedirs(self.result_dir, exist_ok=True)

        # 1. 정보 추출 및 맵 생성
        node_names = self.env.generator.config.node_names
        loads_map = {load['name']: load for load in self.env.generator.config.loads}
        # ⚠️ 사용된 IC의 '특화된' 정보를 가져오기 위해 generator의 전체 목록을 사용
        candidate_ics_map = {ic['name']: ic for ic in self.env.generator.config.available_ics}
        battery_conf = self.env.generator.config.battery
        constraints = self.env.generator.config.constraints
        final_features = final_td["nodes"][0]

        
        # 2. 사용된 노드, IC, 엣지 정보 재구성
        used_ic_names = set()
        child_to_parent = {}
        parent_to_children = defaultdict(list)

        for parent_idx, child_idx in action_history:
            parent_name = node_names[parent_idx]
            child_name = node_names[child_idx]
            child_to_parent[child_name] = parent_name
            parent_to_children[parent_name].append(child_name)
            if parent_name in candidate_ics_map:
                used_ic_names.add(parent_name)

        # 3. Always-On 경로 추적
        always_on_nodes = {
            name for name, conf in loads_map.items() if conf.get("always_on_in_sleep", False)
        }
        nodes_to_process = list(always_on_nodes)
        while nodes_to_process:
            node = nodes_to_process.pop(0)
            if node in child_to_parent:
                parent = child_to_parent[node]
                if parent != battery_conf['name'] and parent not in always_on_nodes:
                    always_on_nodes.add(parent)
                    nodes_to_process.append(parent)

        supplier_nodes = set()
        path_nodes = set()
        for name, conf in loads_map.items():
            rail_type = conf.get("independent_rail_type")
            if rail_type == 'exclusive_supplier':
                supplier_nodes.add(name)
                if name in child_to_parent:
                    supplier_nodes.add(child_to_parent[name])
            elif rail_type == 'exclusive_path':
                current_node = name
                while current_node in child_to_parent:
                    path_nodes.add(current_node)
                    parent = child_to_parent[current_node]
                    path_nodes.add(parent)
                    if parent == battery_conf['name']: break
                    current_node = parent

        # 4. 액티브/슬립 전류 및 전력 계산 (Bottom-up 방식)
        # 💡 [수정] 중복 선언을 제거하고 OR-Tools와 동일한 변수명으로 통일합니다.
        junction_temps, actual_i_ins_active, actual_i_outs_active = {}, {}, {}
        actual_i_ins_sleep, actual_i_outs_sleep, ic_self_consumption_sleep = {}, {}, {}
        
        # 초기값: 부하들의 전류 소모량 설정
        active_current_draw = {name: conf["current_active"] for name, conf in loads_map.items()}
        sleep_current_draw = {name: conf["current_sleep"] for name, conf in loads_map.items()}


        processed_ics = set()
        used_ic_objects = [candidate_ics_map[name] for name in used_ic_names]

        while len(processed_ics) < len(used_ic_objects):
            progress_made = False
            for ic_conf in used_ic_objects:
                ic_name = ic_conf['name']
                if ic_name in processed_ics: continue

                children_names = parent_to_children.get(ic_name, [])
                if all(c in loads_map or c in processed_ics for c in children_names):
                    ic_obj = LDO(**ic_conf) if ic_conf['type'] == 'LDO' else BuckConverter(**ic_conf)
                    
                    # Active 전류 계산
                    # 💡 [수정] _active 접미사가 붙은 변수명을 사용하도록 통일합니다.
                    total_i_out_active = sum(active_current_draw.get(c, 0) for c in children_names)
                    actual_i_outs_active[ic_name] = total_i_out_active
                    i_in_active = ic_obj.calculate_input_current(vin=ic_obj.vin, i_out=total_i_out_active)
                    active_current_draw[ic_name] = i_in_active
                    actual_i_ins_active[ic_name] = i_in_active


                    # Sleep 전류 계산
                    i_in_sleep, ic_self_sleep, total_i_out_sleep = 0, 0, 0
                    parent_name = child_to_parent.get(ic_name)
                    
                    if ic_name in always_on_nodes:
                        total_i_out_sleep = sum(sleep_current_draw.get(c, 0) for c in children_names)
                        ic_self_sleep = ic_obj.operating_current
                        if isinstance(ic_obj, LDO):
                            i_in_sleep = total_i_out_sleep + ic_self_sleep
                        elif isinstance(ic_obj, BuckConverter) and ic_obj.vin > 0:
                            eff_sleep = constraints.get('sleep_efficiency_guess', 0.35)
                            p_out_sleep = ic_obj.vout * total_i_out_sleep
                            p_in_sleep = p_out_sleep / eff_sleep if p_out_sleep > 0 else 0
                            i_in_sleep = (p_in_sleep / ic_obj.vin) + ic_self_sleep
                    elif parent_name in always_on_nodes or parent_name == battery_conf['name']:
                        ic_self_sleep = ic_obj.shutdown_current if (ic_obj.shutdown_current is not None and ic_obj.shutdown_current > 0) else ic_obj.quiescent_current
                        i_in_sleep = ic_self_sleep
                    
                    actual_i_ins_sleep[ic_name] = i_in_sleep
                    actual_i_outs_sleep[ic_name] = total_i_out_sleep
                    ic_self_consumption_sleep[ic_name] = ic_self_sleep
                    sleep_current_draw[ic_name] = i_in_sleep

                    processed_ics.add(ic_name)
                    progress_made = True
            if not progress_made and len(used_ic_objects) > 0: break

        # 5. 최종 시스템 전체 값 계산
        primary_nodes = parent_to_children.get(battery_conf['name'], [])
        total_active_current = sum(active_current_draw.get(name, 0) for name in primary_nodes)
        total_sleep_current = sum(sleep_current_draw.get(name, 0) for name in primary_nodes)
        battery_avg_voltage = (battery_conf['voltage_min'] + battery_conf['voltage_max']) / 2
        total_active_power = battery_avg_voltage * total_active_current

        # 6. Graphviz 다이어그램 생성
        dot = Digraph(comment=f"Power Tree - Cost ${final_cost:.4f}")
        dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
        dot.attr(rankdir='LR', label=f"Transformer Solution (Start: {best_start_node_name})\\nCost: ${final_cost:.4f}", labelloc='t')

        max_sleep_current_target = constraints.get('max_sleep_current', 0.0)
        battery_label = (f"🔋 {battery_conf['name']}\\n\\n"
            f"Total Active Power: {total_active_power:.2f} W\\n"
            f"Total Active Current: {total_active_current * 1000:.1f} mA\\n"
            f"Target Sleep Current: <= {max_sleep_current_target * 1000000:,.1f} µA\n"
            f"Total Sleep Current: {total_sleep_current * 1000000:,.1f} µA")
        dot.node(battery_conf['name'], battery_label, shape='Mdiamond', color='darkgreen', fillcolor='white')

        for ic_name in used_ic_names:
            ic_conf = candidate_ics_map[ic_name]
            ic_idx = node_names.index(ic_name)
            
            i_in_active_val = actual_i_ins_active.get(ic_name, 0)
            i_out_active_val = actual_i_outs_active.get(ic_name, 0)
            i_in_sleep_val = actual_i_ins_sleep.get(ic_name, 0)
            i_out_sleep_val = actual_i_outs_sleep.get(ic_name, 0)
            i_self_sleep_val = ic_self_consumption_sleep.get(ic_name, 0)
            calculated_tj = final_features[ic_idx, FEATURE_INDEX["junction_temp"]].item()
            
            thermal_margin = ic_conf['t_junction_max'] - calculated_tj
            node_color = 'blue'
            if thermal_margin < 10: node_color = 'red'
            elif thermal_margin < 25: node_color = 'orange'
            # --- 💡 [수정] 노드 스타일링 로직 ---
            node_style = 'rounded,filled'
            if ic_name not in always_on_nodes:
                node_style += ',dashed'

            fill_color = 'white'
            if ic_name in path_nodes:
                fill_color = 'lightblue'
            elif ic_name in supplier_nodes:
                fill_color = 'lightyellow'
            # --- 수정 완료 ---
            
            label = (f"📦 {ic_conf['name'].split('@')[0]}\\n\\n"
                     f"Vin: {ic_conf['vin']:.2f}V, Vout: {ic_conf['vout']:.2f}V\\n"
                     f"Iin: {i_in_active_val*1000:.1f}mA (Act) | {i_in_sleep_val*1000000:,.1f}µA (Slp)\\n"
                     f"Iout: {i_out_active_val*1000:.1f}mA (Act) | {i_out_sleep_val*1000000:,.1f}µA (Slp)\\n"
                     f"I_self: {ic_conf['operating_current']*1000:.1f}mA (Act) | {i_self_sleep_val*1000000:,.1f}µA (Slp)\\n"
                     f"Tj: {calculated_tj:.1f}°C (Max: {ic_conf['t_junction_max']}°C)\\n"
                     f"Cost: ${ic_conf['cost']:.2f}")
            dot.node(ic_name, label, color=node_color, fillcolor=fill_color, style=node_style, penwidth='3')

        for name, conf in loads_map.items():
            # --- 💡 [수정] 부하 노드 스타일링 로직 ---
            node_style = 'rounded,filled'
            if name not in always_on_nodes:
                node_style += ',dashed'

            fill_color = 'white'
            if name in path_nodes:
                fill_color = 'lightblue'
            elif name in supplier_nodes:
                fill_color = 'lightyellow'

            label = f"💡 {name}\\nActive: {conf['voltage_typical']}V | {conf['current_active']*1000:.1f}mA\\n"
            if conf['current_sleep'] > 0:
                label += f"Sleep: {conf['current_sleep'] * 1000000:,.1f}µA"
            penwidth = '3' if conf.get("always_on_in_sleep", False) else '1'
            dot.node(name, label, color='dimgray', fillcolor=fill_color, style=node_style, penwidth=penwidth)

        for p_name, children in parent_to_children.items():
            for c_name in children:
                dot.edge(p_name, c_name)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transformer_solution_cost_{final_cost:.4f}_{timestamp}"
        output_path = os.path.join(self.result_dir, filename)
        
        try:
            dot.render(output_path, view=False, format='png', cleanup=True)
            logging.info(f"✅ 상세 시각화 다이어그램을 {output_path}.png 파일로 저장했습니다.")
        except Exception as e:
            logging.error(f"❌ 시각화 렌더링 실패. Graphviz가 설치되어 있고 PATH에 등록되었는지 확인하세요. 오류: {e}")






