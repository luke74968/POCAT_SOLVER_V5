#config.json

# 필요 라이브러리 설치
python -m pip install -r .\requirements.lock.txt

# 파이토치 설치 
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# 1. 현재 터미널의 정책만 임시로 변경
Set-ExecutionPolicy Bypass -Scope Process

# 2. 가상 환경 활성화
# window용 
.\.venv\Scripts\activate.ps1
# Linux용 
source .venv/bin/activate


# 3. Code 실행
# OR-TOOLS Solver 실행
python -m or_tools_solver.main config.json --max_sleep_current 0.01
# Tronsformer based solver 학습 실행
python -m transformer_solver.run --config_file config_4.json --config_yaml config.yaml --batch_size 1 --log_idx 8 --log_mode detail --decode_type sampling
python -m transformer_solver.run --config_file config_4.json --config_yaml config.yaml --batch_size 256 --log_idx 8 --log_mode progress  --decode_type sampling     
# 학습된 결과로 Tronsformer based solver test 실행
python -m transformer_solver.run --config_file config_4.json --test_only --log_mode detail --load_path "result/..." --
python -m transformer_solver.run --test_only --config_file config_4.json --log_mode detail --log_idx 0 --load_path "result\2025-0923-174528\epoch-5.pth"     




# git hub 사용방법
# git add . or git add 파일명
# git commit -m "description"
# git push origin master