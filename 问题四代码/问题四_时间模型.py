"""
问题四 时间模型模块
- TimeModel 抽象基类
- LinearTimeModel: 恒定速度，无加速度，不区分空满载
- NonlinearTimeModel: 梯形/三角形速度曲线，区分空满载
"""
import math

# ============ 层高计算 (与问题三一致) ============
COL_WIDTH = 0.4  # m

LH = {}
cum = 0.0
for z in range(1, 51):
    if z <= 8: h = 0.2
    elif z <= 42: h = 0.4
    else: h = 0.5
    LH[z] = cum + h / 2
    cum += h


class TimeModel:
    """时间模型抽象基类"""
    COL_WIDTH = 0.4
    T0 = 6.0

    def calc_leg(self, dx_m, dz_m, state):
        """单段行程时间，子类实现"""
        raise NotImplementedError

    def T_out(self, x, z, state):
        """原点→(x,z)单程时间"""
        dx = x * self.COL_WIDTH
        dz = LH[z]
        return self.calc_leg(dx, dz, state)

    def T_between(self, x1, z1, x2, z2, state):
        """两位间移动时间（顺序运动）"""
        dx = abs(x2 - x1) * self.COL_WIDTH
        dz = abs(LH[z2] - LH[z1])
        return self.calc_leg(dx, dz, state)


class LinearTimeModel(TimeModel):
    """线性匀速模型：vx, vz恒定，不分空满载"""
    def __init__(self, vx=2.5, vz=0.65):
        self.vx = vx
        self.vz = vz
        self.name = f'Linear(vx={vx},vz={vz})'

    def calc_leg(self, dx_m, dz_m, state=None):
        return dx_m / self.vx + dz_m / self.vz


class NonlinearTimeModel(TimeModel):
    """非线性速度模型：梯形/三角形速度曲线，区分空满载"""
    # 空载参数
    VX_E, VZ_E = 3.0, 0.75
    AX_E, AZ_E = 0.5, 0.15
    # 满载参数
    VX_L, VZ_L = 2.3, 0.58
    AX_L, AZ_L = 0.4, 0.10

    def __init__(self):
        self.name = 'Nonlinear(empty:3.0/0.75, loaded:2.3/0.58)'

    def calc_axis_time(self, distance_m, v_max, a):
        """单轴梯形/三角形速度曲线时间"""
        if distance_m < 1e-12:
            return 0.0
        d_accel_total = v_max * v_max / a  # 加速+减速所需最小距离
        if distance_m >= d_accel_total:
            # 梯形: 加速 + 匀速 + 减速
            t_accel = v_max / a
            t_const = (distance_m - d_accel_total) / v_max
            return 2.0 * t_accel + t_const
        else:
            # 三角形: 加速到某速度后立即减速
            return 2.0 * math.sqrt(distance_m / a)

    def calc_leg(self, dx_m, dz_m, state):
        if state == 'empty':
            tx = self.calc_axis_time(dx_m, self.VX_E, self.AX_E)
            tz = self.calc_axis_time(dz_m, self.VZ_E, self.AZ_E)
        else:  # 'loaded'
            tx = self.calc_axis_time(dx_m, self.VX_L, self.AX_L)
            tz = self.calc_axis_time(dz_m, self.VZ_L, self.AZ_L)
        return tx + tz


# ============ 自检：时间公式验证 ============
def self_test():
    """验证时间模型在典型位置的计算结果"""
    import sys
    linear = LinearTimeModel(vx=2.5, vz=0.65)
    nonlinear = NonlinearTimeModel()

    test_cases = [
        (1, 1),   # 最近
        (34, 25), # 中间
        (68, 50), # 最远
        (1, 50),  # 近列高层
        (68, 1),  # 远列低层
    ]

    print("=" * 70)
    print("时间模型自检")
    print("=" * 70)

    for x, z in test_cases:
        dx = x * COL_WIDTH
        dz = LH[z]
        print(f"\n({x=}, {z=}): dx={dx:.2f}m, dz={dz:.2f}m")

        # Linear
        t_lin = linear.T_out(x, z, 'empty')
        print(f"  Linear:         {t_lin:8.3f}s")

        # Nonlinear empty
        t_nl_e = nonlinear.T_out(x, z, 'empty')
        tx_e = nonlinear.calc_axis_time(dx, nonlinear.VX_E, nonlinear.AX_E)
        tz_e = nonlinear.calc_axis_time(dz, nonlinear.VZ_E, nonlinear.AZ_E)
        # Check which profile was used
        d_x_accel = nonlinear.VX_E**2 / nonlinear.AX_E
        d_z_accel = nonlinear.VZ_E**2 / nonlinear.AZ_E
        profile_x = '梯形' if dx >= d_x_accel else '三角'
        profile_z = '梯形' if dz >= d_z_accel else '三角'
        print(f"  Nonlinear(empty):  {t_nl_e:8.3f}s (x:{tx_e:.3f}s {profile_x}, z:{tz_e:.3f}s {profile_z})")

        # Nonlinear loaded
        t_nl_l = nonlinear.T_out(x, z, 'loaded')
        tx_l = nonlinear.calc_axis_time(dx, nonlinear.VX_L, nonlinear.AX_L)
        tz_l = nonlinear.calc_axis_time(dz, nonlinear.VZ_L, nonlinear.AZ_L)
        d_x_accel_l = nonlinear.VX_L**2 / nonlinear.AX_L
        d_z_accel_l = nonlinear.VZ_L**2 / nonlinear.AZ_L
        profile_xl = '梯形' if dx >= d_x_accel_l else '三角'
        profile_zl = '梯形' if dz >= d_z_accel_l else '三角'
        print(f"  Nonlinear(loaded): {t_nl_l:8.3f}s (x:{tx_l:.3f}s {profile_xl}, z:{tz_l:.3f}s {profile_zl})")

    # 纯出库比较
    print("\n--- 纯出库操作时间对比 ---")
    for x, z in test_cases:
        t_lin_ob = linear.T_out(x,z,'empty') + linear.T0 + linear.T_out(x,z,'empty')
        t_nl_ob = nonlinear.T_out(x,z,'empty') + nonlinear.T0 + nonlinear.T_out(x,z,'loaded')
        diff_pct = (t_nl_ob - t_lin_ob) / t_lin_ob * 100
        print(f"  ({x},{z}): Linear={t_lin_ob:.1f}s, Nonlinear={t_nl_ob:.1f}s, diff={diff_pct:+.1f}%")

    print("\n✓ 时间模型自检完成")
    return True


if __name__ == '__main__':
    self_test()
