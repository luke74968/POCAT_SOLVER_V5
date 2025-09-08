#config.json

# 필요 라이브러리 설치
python -m pip install -r .\requirements.lock.txt

# 파이토치 설치 
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# 1. 현재 터미널의 정책만 임시로 변경
Set-ExecutionPolicy Bypass -Scope Process

# 2. 가상 환경 활성화
.\.venv\Scripts\activate.ps1

# 3. Code 실행
# OR-TOOLS Solver 실행
python main.py json 파일명
# Tronsformer based solver 학습 실행
python run.py --config_file config.json --config_yaml config.yaml --batch_size 64
# 학습된 결과로 Tronsformer based solver test 실행
python run.py --test_only --load_path "result/2025-0905-171711/checkpoint-epoch-100.pth"





# git hub 사용방법
# git add . or git add 파일명
# git commit -m "description"
# git push origin master