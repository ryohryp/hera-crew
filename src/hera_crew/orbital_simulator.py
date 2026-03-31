import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# 物理定数 (SI単位系)
G = 6.67430e-11  # 万有引力定数
M = 5.972e24     # 地球質量
C = 299792458    # 光速
GM = G * M

class OrbitalSimulator:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.GM = torch.tensor(GM, device=self.device, dtype=torch.float64)
        self.C2 = torch.tensor(C**2, device=self.device, dtype=torch.float64)

    def get_acceleration(self, pos, vel):
        """
        シュヴァルツシルト計量に基づく一般相対性理論補正を含んだ加速度計算
        a = -GM/r^3 * r * (1 + 3L^2 / (c^2 * r^2))
        """
        r_sum_sq = torch.sum(pos**2)
        r = torch.sqrt(r_sum_sq)
        
        # ニュートン加速度
        acc_newton = -self.GM * pos / (r**3)
        
        # 角運動量 L = |pos x vel| (2D平面での外積の大きさ)
        # torch.linalg.cross は3Dベクトルが必要
        p3 = torch.cat([pos, torch.zeros(1, device=self.device, dtype=torch.float64)])
        v3 = torch.cat([vel, torch.zeros(1, device=self.device, dtype=torch.float64)])
        L_vec = torch.linalg.cross(p3, v3)
        L2 = torch.sum(L_vec**2)
        
        # 一般相対性理論による補正項 (GR correction factor)
        # 3L^2 / (c^2 * r^2)
        gr_factor = 1.0 + (3.0 * L2) / (self.C2 * r_sum_sq)
        
        return acc_newton * gr_factor

    def step_rk4(self, pos, vel, dt):
        """4次ルンゲ＝クッタ法による1ステップの更新"""
        
        def derivatives(p, v):
            return v, self.get_acceleration(p, v)

        k1_v, k1_a = derivatives(pos, vel)
        
        k2_v, k2_a = derivatives(pos + k1_v * dt / 2, vel + k1_a * dt / 2)
        
        k3_v, k3_a = derivatives(pos + k2_v * dt / 2, vel + k2_a * dt / 2)
        
        k4_v, k4_a = derivatives(pos + k3_v * dt, vel + k3_a * dt)
        
        new_pos = pos + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        new_vel = vel + (dt / 6.0) * (k1_a + 2*k2_a + 2*k3_a + k4_a)
        
        return new_pos, new_vel

    def simulate(self, initial_pos, initial_vel, dt, steps):
        pos = torch.tensor(initial_pos, device=self.device, dtype=torch.float64)
        vel = torch.tensor(initial_vel, device=self.device, dtype=torch.float64)
        
        history_pos = []
        
        for i in range(steps):
            if i % 100 == 0:
                history_pos.append(pos.cpu().numpy().copy())
            pos, vel = self.step_rk4(pos, vel, dt)
            
        return np.array(history_pos)

def run_example():
    sim = OrbitalSimulator()
    
    # 地球半径 ~6371km + 高度 500km
    altitude = 500000 
    r_earth = 6371000
    r0 = r_earth + altitude
    
    # 第一宇宙速度に近い速度
    v0 = np.sqrt(GM / r0) * 1.05  # 少し楕円軌道にする
    
    initial_pos = [r0, 0.0]
    initial_vel = [0.0, v0]
    
    dt = 1.0  # 1秒刻み
    steps = 10000 # 10000秒分 (約2.7時間)
    
    print(f"Simulation started: r0={r0/1000:.2f}km, v0={v0:.2f}m/s")
    path = sim.simulate(initial_pos, initial_vel, dt, steps)
    
    # プロット
    plt.figure(figsize=(10, 10))
    
    # 地球を描画
    earth_circle = plt.Circle((0, 0), r_earth, color='blue', alpha=0.3, label='Earth')
    plt.gca().add_patch(earth_circle)
    
    plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=1, label='Satellite Trajectory (GR corrected)')
    plt.scatter(path[0, 0], path[0, 1], color='green', label='Start', zorder=5)
    
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title("Satellite Orbit: General Relativity Simulation (Schwarzschild Metric)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    
    # カレントディレクトリに保存
    output_file = "orbital_plot.png"
    plt.savefig(output_file, dpi=150)
    print(f"Simulation completed. Plot saved as {output_file}")
    
    # ファイルが存在するか確認
    if os.path.exists(output_file):
        print(f"Verified: {output_file} exists.")

if __name__ == "__main__":
    run_example()
