/**
 * 步进电机驱动模块 (TB6600 + NEMA17)
 * ------------------------------------------------------------------
 * 采用**定时器中断软件脉冲**方案：TIM 以固定周期中断，
 * 中断里翻转 STEP 引脚；达到目标步数后停止。
 *
 * 相比"PWM 输出 + 计数"，软件脉冲便于：
 *   - 动态改速度（改 ARR）
 *   - 精确控制脉冲个数
 *   - 后续加入加减速
 * ------------------------------------------------------------------
 */
#ifndef TRINISCAN_MOTOR_H
#define TRINISCAN_MOTOR_H

#include <stdbool.h>
#include <stdint.h>

/* 初始化：绑定时器句柄、复位状态。在 MX_TIMx_Init() 之后调用 */
void motor_init(void);

/* 使能/脱机电机（脱机后可手动转动转台） */
void motor_enable(bool on);
bool motor_is_enabled(void);

/* 运动（非阻塞）：设定目标步数后立即返回，电机在中断中走完 */
void motor_move_steps(int32_t steps);
void motor_move_degrees(float deg);

/* 回零（非阻塞）：朝限位方向慢速运动，触发限位后停并清零计数 */
void motor_home(void);

/* 急停：立即停止脉冲输出（保留当前位置计数） */
void motor_stop(void);

/* 定时器中断处理：由 HAL_TIM_PeriodElapsedCallback 调用 */
void motor_on_timer_tick(void);

/* 状态查询 */
bool    motor_is_busy(void);
int32_t motor_position_steps(void);     /* 累计步数（相对/绝对，视回零） */
float   motor_position_degrees(void);

/* 设置速度（微秒/步），被限制在 [MIN_STEP_US, MAX_STEP_US] */
void motor_set_speed_us(uint32_t us);
uint32_t motor_get_speed_us(void);

#endif /* TRINISCAN_MOTOR_H */
