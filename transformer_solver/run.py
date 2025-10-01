# transformer_solver/run.py
import os
import sys
import time
import yaml
import json
import random
import torch
import logging
import argparse

if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
    torch.set_float32_matmul_precision('high')


from .trainer import PocatTrainer
from .pocat_env import PocatEnv

def setup_logger(result_dir):
    log_file = os.path.join(result_dir, 'log.txt')
    logging.basicConfig(filename=log_file, format='%(asctime)-15s %(message)s', level=logging.INFO)
    logger = logging.getLogger()
    console = logging.StreamHandler(sys.stdout)
    logger.addHandler(console)
    return logger

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.log(f"Using device: {device}")
    
    # --- 👇 1. PocatEnv 생성 시 instance_repeats 인자 제거 ---
    env = PocatEnv(
        generator_params={"config_file_path": args.config_file},
        device=device,
    )
    # --- 수정 완료 ---

    # 💡 model_params에 num_nodes 추가
    args.model_params['num_nodes'] = env.generator.num_nodes    
    
    trainer = PocatTrainer(args, env, device)

    if args.test_only:
        trainer.test()
    else:
        trainer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 훈련 관련 인자
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch_size")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Evaluation batch size (fixed set)")

    parser.add_argument("--config_file", type=str, default="config.json", help="Path to POCAT config file")
    parser.add_argument("--config_yaml", type=str, default="config.yaml", help="Path to model/training config YAML")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    # --- 👇 [수정] POMO 샘플링 횟수 인자 추가 ---
    parser.add_argument("--num_pomo_samples", type=int, default=8, 
                        help="Number of POMO samples to run during training.")
    parser.add_argument("--test_num_pomo_samples", type=int, default=8, 
                        help="Number of POMO samples for testing/evaluation. Defaults to num_pomo_samples if not set.")

    # 💡 추론을 위한 인자 추가
    parser.add_argument('--test_only', action='store_true', help="Only run test/inference")
    parser.add_argument('--load_path', type=str, default=None, help="Path to a saved model checkpoint (.pth)")
    
    # 💡 로그 관련 인자 추가
    parser.add_argument('--log_idx', type=int, default=0, help='Instance index to log (for POMO)')
    parser.add_argument('--log_mode', type=str, default='progress', choices=['progress', 'detail'],
                        help="Logging mode: 'progress' for progress bar, 'detail' for step-by-step logs.")

    parser.add_argument('--decode_type', type=str, default='greedy', choices=['greedy', 'sampling'],
                        help="Decoding strategy for test mode: 'greedy' or 'sampling'.")

    args = parser.parse_args()

    if args.test_num_pomo_samples is None:
     args.test_num_pomo_samples = args.num_pomo_samples

    
    args.start_time = time.strftime("%Y-%m%d-%H%M%S", time.localtime())
    args.result_dir = os.path.join('result', args.start_time)
    os.makedirs(args.result_dir, exist_ok=True)
    
    logger = setup_logger(args.result_dir)
    args.log = logger.info
    
    with open(args.config_yaml, "r", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)
    for key, value in cfg_yaml.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    args.ddp = False
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 💡 수정된 부분: 로깅 전 non-JSON-serializable 객체 제거
    args_dict_for_log = vars(args).copy()
    del args_dict_for_log['log']
    args.log(json.dumps(args_dict_for_log, indent=4))
    
    main(args)