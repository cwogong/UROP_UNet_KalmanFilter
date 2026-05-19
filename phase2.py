# 필요한 라이브러리 임포트
import yaml
import os

# 현재 디렉토리 확인
current_dir = os.getcwd()
print(f"현재 디렉토리: {current_dir}")

# config.yaml 파일 경로 설정
config_path = os.path.join(current_dir, 'config.yaml')

# YAML 파일 읽기
try:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print("config.yaml 파일을 성공적으로 읽었습니다.")
except FileNotFoundError:
    print(f"config.yaml 파일을 찾을 수 없습니다. 경로를 확인하세요: {config_path}")
except yaml.YAMLError as e:
    print(f"YAML 파일을 읽는 중 오류가 발생했습니다: {e}")

# 읽은 설정 출력
print("\n[config.yaml 파일 내용]")
print(config)

# 주요 설정 접근
print("\n[설정 값 출력]")
try:
    # 데이터셋 설정
    dataset_root = config['dataset']['root']
    batch_size = config['dataset']['batch_size']

    # 학습 설정
    epochs = config['trainer']['epochs']
    learning_rate = config['trainer']['lr']

    # 모델 설정
    model_name = config['model']['name']
    in_channels = config['model']['in_channels']

    # 칼만 필터 설정
    process_noise = config['kalman']['process_noise']
    measurement_noise = config['kalman']['measurement_noise']

    # 설정 값 출력
    print(f"Dataset root: {dataset_root}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Model: {model_name}, In channels: {in_channels}")
    print(f"Process noise: {process_noise}, Measurement noise: {measurement_noise}")
except KeyError as e:
    print(f"설정 파일에서 누락된 키가 있습니다: {e}")