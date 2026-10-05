/**
 * 步进电机驱动模块 (TIM1 PWM + 中断计数)
 * ------------------------------------------------------------------
 * 用 TIM1_CH1 (PA8) 的 PWM 硬件输出 STEP 脉冲：
 *   - PWM 频率 = 步进脉冲频率（决定转速）
 *   - TIM1 更新中断里精确计数，到数即停
 *   - 前置条件：CubeMX 里给 TIM1 开 update interrupt
 * ------------------------------------------------------------------
 */
#ifndef TRINISCAN_MOTOR_H
#define TRINISCAN_MOTOR_H

#include <stdbool.h>
#include <stdint.h>

/* 初始化：复位状态（TIM1 已由 MX_TIM1_Init 配好，此处只做初值） */
void motor_init(void);

/* 使能 / 脱机 */
void motor_enable(bool on);
bool motor_is_enabled(void);

/* 运动（非阻塞）：设定目标后立即返回，靠 TIM1 中断推进 */
void motor_move_steps(int32_t steps);
void motor_move_degrees(float deg);

/* 回零（非阻塞） */
void motor_home(void);

/* 急停 */
void motor_stop(void);

/* TIM1 更新中断回调：每个 STEP 脉冲调用一次
 * （在 main.c 的 HAL_TIM_PeriodElapsedCallback 里调用） */
void motor_on_tick(void);

/* 状态 */
bool    motor_is_busy(void);
int32_t motor_position_steps(void);
float   motor_position_degrees(void);

/* 速度 */
void     motor_set_speed_us(uint32_t us);
uint32_t motor_get_speed_us(void);

#endif /* TRINISCAN_MOTOR_H */
