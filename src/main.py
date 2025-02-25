import time
from GymWrapper import *
from GymEnvironment import *
from config_SimPy import *
from config_DQN import *

# 실행 시간 측정 시작
start_time = time.time()

# 환경 생성 (Single-Agent)
env = InventoryManagementEnv()

# DQN 학습 모델 초기화
wrapper = GymWrapper(
    env=env,
    action_dim=ACTION_MAX-ACTION_MIN+1,
    state_dim=STATE_DIM,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    gamma=GAMMA
)

if LOAD_MODEL:
    # 저장된 모델 불러오기 및 평가 수행
    print(f"Loading model from {MODEL_PATH}")
    try:
        wrapper.load_model(MODEL_PATH)
        print("Model loaded successfully")
        
        # 모델 평가
        training_end_time = time.time()
        wrapper.evaluate(N_EVAL_EPISODES)
    except FileNotFoundError:
        print(f"No saved model found at {MODEL_PATH}")
        exit()
else:
    # 새로운 모델 학습
    print("Starting training of new model...")
    wrapper.train(N_TRAIN_EPISODES, EVAL_INTERVAL)
    training_end_time = time.time()

    # 학습된 모델 평가
    print("\nStarting evaluation...")
    wrapper.evaluate(N_EVAL_EPISODES)

# 실행 시간 계산 및 출력
end_time = time.time()
print("\nTime Analysis:")
print(f"Total computation time: {(end_time - start_time)/60:.2f} minutes")
if not LOAD_MODEL:
    print(f"Training time: {(training_end_time - start_time)/60:.2f} minutes")
print(f"Evaluation time: {(end_time - training_end_time)/60:.2f} minutes")

# TensorBoard 실행 방법 안내
print("\nTo visualize training progress, run:")
print("tensorboard --logdir=runs")
