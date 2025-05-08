import time
 
# 导出 PWM 控制器
def pwm_export(pwmid):
    with open('/sys/class/pwm/pwmchip%d/export' % pwmid, 'wb') as f:
        f.write("0".encode())
 
# 取消导出 PWM 控制器
def pwm_unexport(pwmid):
    with open('/sys/class/pwm/pwmchip%d/unexport' % pwmid, 'wb') as f:
        f.write("0".encode())
 
# 配置 PWM 周期和占空比
def pwm_config(pwmid, pwmPeriod, pwmDutyCycle):
    with open('/sys/class/pwm/pwmchip%d/pwm0/period' % pwmid, 'wb') as f:
        f.write(str(pwmPeriod).encode())
    with open('/sys/class/pwm/pwmchip%d/pwm0/duty_cycle' % pwmid, 'wb') as f:
        f.write(str(pwmDutyCycle).encode())
 
# 使能 PWM 控制器
def pwm_enable(pwmid):
    with open('/sys/class/pwm/pwmchip%d/pwm0/enable' % pwmid, 'wb') as f:
        f.write("1".encode())
 
# 禁用 PWM 控制器
def pwm_disable(pwmid):
    with open('/sys/class/pwm/pwmchip%d/pwm0/enable' % pwmid, 'wb') as f:
        f.write("0".encode())
 
# 设置 PWM 极性
def pwm_polarity(pwmid, pwmPolarity):
    with open('/sys/class/pwm/pwmchip%d/pwm0/polarity' % pwmid, 'wb') as f:
        f.write(pwmPolarity.encode())
 
# 初始化 PWM 控制器
def pwm_structure(pwmid, pwmPeriod, pwmDutyCycle, pwmPolarity):
    pwm_export(pwmid)
    pwm_config(pwmid, pwmPeriod, pwmDutyCycle)
    pwm_polarity(pwmid, pwmPolarity)
    pwm_enable(pwmid)
 
# 销毁 PWM 控制器
def pwm_destruction(pwmid):
    pwm_disable(pwmid)
    pwm_unexport(pwmid)
 
# 角度转换为 PWM 占空比
def set_servo_angle(angle):
    min_pulse = 500  # 最小脉冲宽度
    max_pulse = 2500  # 最大脉冲宽度
    # 限制角度在0到180度之间
    angle = max(0, min(180, angle))
    # 计算占空比
    pwmDutyCycle = 1000 * int((angle / 180.0) * (max_pulse - min_pulse) + min_pulse)
    # 配置 PWM
    pwm_config(1, 20000000, pwmDutyCycle)
 
# 比例控制函数
def P_control(initial_value, target_value, change_rate):
    while abs(initial_value - target_value) > 0.5:  # 当最后一个返回的值接近设定值时退出循环
        error = target_value - initial_value
        change = error * change_rate
        initial_value += change
        set_servo_angle(initial_value)
 
# 主函数
if __name__ == '__main__':
    pwm_index = 1
    pwm_structure(pwm_index, 20000000, 0, "normal")  # 初始化 PWM 控制器，设置周期和占空比
    #try:
        #set_servo_angle(45) 
    while True:
            # 使用比例控制器进行控制
        set_servo_angle(180) 
    pwm_destruction(1)  # 销毁 PWM 控制器
    #set_servo_angle(180) 
    #pwm_destruction(1)  # 销毁 PWM 控制器
    #finally:
    #    