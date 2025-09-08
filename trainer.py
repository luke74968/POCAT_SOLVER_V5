# trainer.py
import torch
from tqdm import tqdm

from utils.common import TimeEstimator, clip_grad_norms, unbatchify
from model import PocatModel
from pocat_env import PocatEnv
import os # os 모듈 추가


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

    def run(self):
        args = self.args
        self.time_estimator.reset(self.start_epoch)
        
        if args.test_only:
            self.test()
            return

        for epoch in range(self.start_epoch, args.trainer_params['epochs'] + 1):
            args.log('=================================================================')
            
            self.model.train()
            train_pbar = tqdm(range(args.trainer_params['train_step']), 
                              bar_format='{desc}|{elapsed}+{remaining}|{n_fmt}/{total_fmt}', 
                              leave=False, dynamic_ncols=True)
            train_label = f"Train|E{str(epoch).zfill(3)}/{args.trainer_params['epochs']}"
            
            for step in train_pbar:
                self.optimizer.zero_grad()
                td = self.env.reset(
                    batch_size=args.batch_size, instance_repeats=args.instance_repeats
                )
                out = self.model(td, self.env)
                
                num_starts = self.env.generator.num_loads
                reward = unbatchify(out["reward"], num_starts)
                log_likelihood = unbatchify(out["log_likelihood"], num_starts)
                
                best_reward, best_idx = reward.max(dim=1)
                advantage = best_reward - reward.mean(dim=1)
                best_log_likelihood = log_likelihood.gather(1, best_idx).squeeze(-1)
                
                loss = -(advantage * best_log_likelihood).mean()
                loss.backward()
                clip_grad_norms(self.optimizer.param_groups, 1.0)
                self.optimizer.step()
                
                avg_cost = -best_reward.mean().item()
                train_pbar.set_description(f"🙏> {train_label}| Loss:{loss.item():.4f} Cost:{avg_cost:.4f}")

            self.scheduler.step()
            self.time_estimator.print_est_time(epoch, args.trainer_params['epochs'])
            
            # 💡 모델 저장 로직 추가
            if (epoch % args.trainer_params['model_save_interval'] == 0) or (epoch == args.trainer_params['epochs']):
                args.log(f"Saving model at epoch {epoch}...")
                checkpoint_path = os.path.join(args.result_dir, f"checkpoint-epoch-{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                }, checkpoint_path)

        args.log(" *** Training Done *** ")

    @torch.no_grad()
    def test(self):
        """저장된 모델을 불러와 Power Tree를 생성하고 결과를 시각화합니다."""
        args = self.args
        args.log("==================== INFERENCE START ====================")
        self.model.eval() # 모델을 평가 모드로 전환

        # 💡 1. 평가할 문제 생성
        # instance_repeats를 통해 동일한 문제를 여러 번 평가할 수 있습니다.
        td = self.env.reset(batch_size=1, instance_repeats=args.instance_repeats)
        
        # 💡 2. 모델로 Power Tree 생성
        out = self.model(td, self.env)
        
        # POMO 결과 중 가장 좋은 것 하나만 선택
        num_starts = self.env.generator.num_loads
        reward = unbatchify(out["reward"], num_starts) # (1*L, 1) -> (1, L, 1)
        actions = unbatchify(out["actions"], num_starts) # (1*L, S, 2) -> (1, L, S, 2)
        
        best_reward, best_idx = reward.max(dim=1)
        best_action_sequence = actions[0, best_idx.item()]
        final_cost = -best_reward.item()
        
        args.log(f"Generated Power Tree Cost: ${final_cost:.4f}")
        
        # 💡 3. 결과 시각화
        self.visualize_result(best_action_sequence, final_cost)

    def visualize_result(self, actions, cost):
        """모델이 생성한 action_sequence를 기반으로 결과를 시각화합니다."""
        # OR-Tools 프로젝트의 시각화 코드를 재활용할 수 있습니다.
        # 여기서는 간단하게 연결 정보를 텍스트로 출력합니다.
        node_names = self.env.generator.config.node_names
        
        print("\n--- Generated Power Tree ---")
        active_edges = []
        for action in actions:
            child_idx, parent_idx = action[0].item(), action[1].item()
            child_name = node_names[child_idx]
            parent_name = node_names[parent_idx]
            active_edges.append((parent_name, child_name))
            print(f"  {parent_name} -> {child_name}")
        print("----------------------------")
        
        # TODO: OR-Tools 프로젝트의 pocat_visualizer.py와 연동하여
        #       이미지 다이어그램을 생성하는 코드를 추가할 수 있습니다.