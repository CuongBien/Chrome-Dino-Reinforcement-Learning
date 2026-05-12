import numpy as np
import random
from pathlib import Path

from dino_env import SimpleDinoEnv

# Đường dẫn lưu / tải bảng Q (model)
MODEL_PATH = Path(__file__).resolve().parent / "dino_q_table.npy"
# True: nếu đã có file model thì tải lên trước khi train (tiếp tục học từ checkpoint)
LOAD_IF_EXISTS = True

env = SimpleDinoEnv()

# Các siêu tham số (Hyperparameters)
alpha = 0.1          # Tốc độ học (Learning Rate)
gamma = 0.95         # Hệ số chiết khấu (Discount Factor - Tầm nhìn xa)
epsilon = 1.0        # Tỷ lệ khám phá ban đầu (1.0 = 100% ngẫu nhiên)
epsilon_decay = 0.999
min_epsilon = 0.01

# Khởi tạo bảng Q-Table toàn số 0
# Shape: (2, 13, 3, 2) tương ứng với (Y_bins, X_bins, Speed_bins, Actions)
q_table = np.zeros((2, 13, 3, 2))
if LOAD_IF_EXISTS and MODEL_PATH.is_file():
    loaded = np.load(MODEL_PATH)
    if loaded.shape == q_table.shape:
        q_table = loaded.astype(np.float64, copy=False)
        print(f"Đã tải model: {MODEL_PATH}")
    else:
        print(f"Cảnh báo: {MODEL_PATH} shape {loaded.shape} không khớp, train từ đầu.")

def get_discrete_state(obs):
    """Gộp tọa độ thực thành index số nguyên để tra bảng"""
    dino_y, obs_x, speed = obs
    y_bin = 1 if dino_y >= 180 else 0
    x_bin = int(max(0, min(600, obs_x)) // 50)
    
    if speed < 12: s_bin = 0
    elif speed < 15: s_bin = 1
    else: s_bin = 2
        
    return (y_bin, x_bin, s_bin)


def greedy_action(q_values):
    """Chọn action Q cao nhất; nếu hòa điểm thì chọn ngẫu nhiên (tránh luôn thiên về action 0)."""
    m = float(np.max(q_values))
    ties = np.flatnonzero(q_values == m)
    return int(np.random.choice(ties))


print("Bắt đầu huấn luyện AI (Không bật đồ họa)...")
train_episodes = 5000

for ep in range(train_episodes):
    obs, _ = env.reset()
    state = get_discrete_state(obs)
    terminated = False
    
    while not terminated:
        # ========================================================
        # TODO 1: CHÍNH SÁCH EPSILON-GREEDY
        # Sinh ngẫu nhiên 1 số từ 0 đến 1. Nếu số này < epsilon -> Khám phá.
        # Ngược lại -> Khai thác (chọn hành động có điểm Q cao nhất ở trạng thái hiện tại).
        # ========================================================
        if random.uniform(0, 1) < epsilon:
            # Chọn hành động ngẫu nhiên từ môi trường
            action = env.action_space.sample()
        else:
            action = greedy_action(q_table[state])
            
        # Đưa action vào môi trường để lấy kết quả
        next_obs, reward, terminated, _, _ = env.step(action)
        next_state = get_discrete_state(next_obs)
        
        # ========================================================
        # TODO 2: CẬP NHẬT THEO PHƯƠNG TRÌNH BELLMAN
        # Terminal: target = R. Không terminal: target = R + gamma * max_a' Q(s',a')
        # Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))
        # ========================================================
        
        # 1. Lấy giá trị Q hiện tại trong bảng: Q(s,a)
        old_value = q_table[state][action]
        
        # 2. Mục tiêu Bellman: nếu thua (terminal) thì không bootstrap từ s'
        if terminated:
            target = reward
        else:
            target = reward + gamma * float(np.max(q_table[next_state]))

        # 3. Cập nhật Q(s,a)
        new_value = old_value + alpha * (target - old_value)
        
        # Cập nhật lại bảng
        q_table[state][action] = new_value
        
        # Cập nhật trạng thái hiện tại bằng trạng thái mới để đi tiếp
        state = next_state
        
    # Giảm dần độ tò mò (epsilon)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    if (ep + 1) % 100 == 0:
        print(f"Đã train {ep + 1} mạng... Epsilon: {epsilon:.2f}")

print("Huấn luyện xong! Bảng Q-Table đã đầy kinh nghiệm.")
np.save(MODEL_PATH, q_table)
print(f"Đã lưu model: {MODEL_PATH}")

# --- ĐOẠN DƯỚI ĐÂY LÀ ĐỂ XEM AI CHƠI THỬ SAU KHI TRAIN XONG ---
print("\nBật đồ họa để xem AI chơi (đóng cửa sổ để thoát sớm)...")

for ep in range(3):
    if env.closed:
        break
    obs, info = env.reset()
    state = get_discrete_state(obs)
    terminated = False
    score = 0

    while not terminated and not env.closed:
        env.render()
        # Không dùng time.sleep thêm: clock.tick trong render đã giới hạn ~30 FPS

        # Khi test: greedy (hòa điểm → chọn ngẫu nhiên giữa các action tốt nhất)
        action = greedy_action(q_table[state])

        obs, reward, terminated, _, info = env.step(action)
        state = get_discrete_state(obs)
        score += reward

    if env.closed:
        print("Đã đóng cửa sổ — dừng demo.")
        break
    print(f"Kết quả mạng {ep + 1}: Điểm {score:.1f} | Vượt qua {info['score']} xương rồng")

env.close()