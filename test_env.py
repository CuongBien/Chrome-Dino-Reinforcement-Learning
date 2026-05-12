import time
from dino_env import SimpleDinoEnv

# 1. Khởi tạo môi trường
env = SimpleDinoEnv()
obs, info = env.reset()

print("Bắt đầu test môi trường. Khủng long sẽ nhảy ngẫu nhiên!")

# 2. Vòng lặp chơi game
episodes = 3 # Chơi thử 3 mạng
for ep in range(episodes):
    obs, info = env.reset()
    terminated = False
    score = 0
    
    while not terminated:
        env.render() # Hiển thị màn hình
        time.sleep(0.01) # Chậm lại một chút để mắt người xem kịp
        
        # CHỌN HÀNH ĐỘNG: Ở đây ta đang cho AI chọn ngẫu nhiên 0 hoặc 1
        # (Ở bước sau, Mạng Neural sẽ thay thế dòng này)
        action = env.action_space.sample() 
        
        # Đưa hành động vào môi trường
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        
    print(f"Mạng {ep + 1} kết thúc! Điểm sống sót: {score:.1f} | Xương rồng vượt qua: {info['score']}")

env.close()