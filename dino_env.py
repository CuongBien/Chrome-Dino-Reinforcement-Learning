import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class SimpleDinoEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # 1. ACTION SPACE (Không gian hành động)
        # Chỉ có 2 hành động: 0 = Chạy bình thường, 1 = Nhảy
        self.action_space = spaces.Discrete(2)
        
        # 2. OBSERVATION SPACE (Không gian trạng thái - "Mắt" của AI)
        # AI sẽ nhìn thấy 3 thông số: [Toạ độ Y của Khủng long, Toạ độ X của Xương rồng, Tốc độ game]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([1000, 1000, 100]), 
            dtype=np.float32
        )

        # Thông số đồ hoạ & Vật lý
        self.screen_width = 600
        self.screen_height = 250
        self.dino_x = 50
        self.dino_y_ground = 180
        self.dino_size = 30
        self.obs_size = 30
        
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        """Hàm này chạy mỗi khi game bắt đầu lại (khi thua)"""
        super().reset(seed=seed)
        self.dino_y = self.dino_y_ground
        self.dino_vy = 0 # Vận tốc rơi
        self.obs_x = self.screen_width # Đưa chướng ngại vật ra mép phải
        self.game_speed = 10 # Tốc độ ban đầu
        self.score = 0
        self.is_jumping = False

        obs = np.array([self.dino_y, self.obs_x, self.game_speed], dtype=np.float32)
        return obs, {}

    def step(self, action):
        """Hàm này thực thi hành động của AI ở mỗi khung hình"""
        # Xử lý Nhảy
        if action == 1 and not self.is_jumping:
            self.dino_vy = -16  # Lực nhảy lên (số âm vì trục Y của Pygame hướng xuống)
            self.is_jumping = True

        # Cập nhật vật lý Khủng long (Trọng lực)
        self.dino_vy += 1.2 # Trọng lực kéo xuống
        self.dino_y += self.dino_vy
        
        # Chạm đất
        if self.dino_y >= self.dino_y_ground:
            self.dino_y = self.dino_y_ground
            self.is_jumping = False

        # Cập nhật xương rồng trôi sang trái
        self.obs_x -= self.game_speed
        if self.obs_x < 0:
            self.obs_x = self.screen_width + np.random.randint(0, 300) # Xuất hiện lại ở bên phải ngẫu nhiên
            self.game_speed += 0.2 # Game ngày càng nhanh
            self.score += 1

        # 3. KIỂM TRA VA CHẠM (Thua game)
        dino_rect = pygame.Rect(self.dino_x, self.dino_y, self.dino_size, self.dino_size)
        obs_rect = pygame.Rect(self.obs_x, self.dino_y_ground, self.obs_size, self.obs_size)
        terminated = dino_rect.colliderect(obs_rect)

        # 4. THIẾT KẾ PHẦN THƯỞNG (REWARD SHAPING)
        if terminated:
            reward = -10.0 # Phạt nặng nếu đụng xương rồng
        else:
            reward = 0.1   # Thưởng nhỏ cho mỗi bước còn sống sót

        obs = np.array([self.dino_y, self.obs_x, self.game_speed], dtype=np.float32)
        info = {"score": self.score}

        return obs, reward, terminated, False, info

    def render(self):
        """Hàm này để vẽ game ra màn hình cho con người xem"""
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("AI Dino Game")
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255)) # Nền trắng

        # Vẽ mặt đất
        pygame.draw.line(self.window, (83, 83, 83), (0, self.dino_y_ground + self.dino_size), (self.screen_width, self.dino_y_ground + self.dino_size), 2)

        # Vẽ Khủng long (Khối xanh lá)
        pygame.draw.rect(self.window, (0, 200, 0), (self.dino_x, self.dino_y, self.dino_size, self.dino_size))

        # Vẽ Xương rồng (Khối đỏ)
        pygame.draw.rect(self.window, (200, 0, 0), (self.obs_x, self.dino_y_ground, self.obs_size, self.obs_size))

        pygame.display.flip()
        self.clock.tick(30) # Chạy ở 30 FPS