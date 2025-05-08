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
def set_servo_angle(angle,pwmid):
    min_pulse = 500  # 最小脉冲宽度
    max_pulse = 2500  # 最大脉冲宽度
    # 限制角度在0到180度之间
    angle = max(0, min(180, angle))
    # 计算占空比
    pwmDutyCycle = 1000 * int((angle / 180.0) * (max_pulse - min_pulse) + min_pulse)
    # 配置 PWM
    pwm_config(pwmid, 20000000, pwmDutyCycle)
 
# 比例控制函数
def P_control(initial_value, target_value, change_rate):
    while abs(initial_value - target_value) > 0.5:  # 当最后一个返回的值接近设定值时退出循环
        error = target_value - initial_value
        change = error * change_rate
        initial_value += change
        set_servo_angle(initial_value)
 
step_x = 90
# 主函数
if __name__ == '__main__':
    pwm_structure(1, 20000000, 0, "normal")  # 初始化 PWM 控制器，设置周期和占空比
    pwm_structure(2, 20000000, 0, "normal") 
    set_servo_angle(90,1)
    set_servo_angle(90,2)
    time.sleep(3)
    try:
        while True:
            with open('shared_x_file.txt', 'r') as file:
                data_x = file.read()
            if data_x:
                print(f"Read data_x from file: {data_x}")
                
                data_x_int = int(float(data_x))-205
                print(data_x_int)
            if (data_x_int)>50:

                if(step_x>170):
                    step_x=170

                step_x-=2
                set_servo_angle(step_x,1)
                time.sleep(0.15)
            if(data_x_int) <-50:
                if(step_x<15):
                    step_x=10

                step_x+=2
                set_servo_angle(step_x,1)
                time.sleep(0.15)

            with open('shared_y_file.txt', 'r') as file:
                data_y = file.read()
            if data_y:
                print(f"Read data_y from file: {data_y}")

            data_y_int = int(float(data_y))-120
            data_y_int = 50
            print(data_y_int)
            if (data_y_int)>50:

                if(step_y>135):
                    step_y=135

                step_y-=2
                set_servo_angle(step_y,1)
                time.sleep(0.15)
            if(data_y_int) <-50:
                if(step_y<45):
                    step_y=45
                    
                step_y+=2
                set_servo_angle(step_y,1)
                time.sleep(0.15)
                # 使用比例控制器进行控制 
    except:
        pwm_destruction(1)
        pwm_destruction(2)
