/**
 * 步进电机驱动模块 — 实现
 * ------------------------------------------------------------------
 * 依赖 CubeMX 生成的：
 *   - 全局定时器句柄  htim_step   (在 main.c 中定义)
 *   - 引脚宏 STEP_xxx / DIR_xxx / EN_xxx / LIMIT_xxx (在 main.h 中由 Label 生成)
 *
 * CubeMX 需配置：
 *   1. 基础定时器 (如 TIM2)：Internal Clock，仅开更新中断，
 *      预分频使计数频率 = 1 MHz（72MHz / 72）。
 *   2. GPIO：STEP/DIR/EN 输出，LIMIT 输入上拉。
 *
 * 工作原理：定时器中断里按 "拉高→拉低" 两拍生成一个 STEP 脉冲，
 * 每两拍计一步。脉冲周期 = 2 × ARR(tick)，即速度由 ARR 决定。
 * ------------------------------------------------------------------
 */
#include "motor.h"
#include "config.h"

/* 由 main.c 提供的定时器句柄 */
extern TIM_HandleTypeDef htim_step;

/* ================= 内部状态 ================= */
static volatile int32_t  s_steps_left   = 0;   /* 剩余步数（含回零上限） */
static volatile int32_t  s_position     = 0;   /* 累计位置（步）         */
static volatile uint32_t s_speed_us     = DEFAULT_STEP_US;
static volatile bool     s_enabled      = false;
static volatile bool     s_homing       = false;
static volatile bool     s_dir_positive = true;
static volatile uint8_t  s_step_phase   = 0;   /* 0 空闲 1 已拉高        */

/* ================= 内部辅助 ================= */
static inline void dir_apply(bool positive)
{
    s_dir_positive = positive;
    HAL_GPIO_WritePin(DIR_GPIO_Port, DIR_Pin,
                      positive ? DIR_POSITIVE_LEVEL
                               : (DIR_POSITIVE_LEVEL == GPIO_PIN_SET
                                      ? GPIO_PIN_RESET : GPIO_PIN_SET));
}

/* tick 数换算为定时器 ARR（计数频率 1 MHz，1 tick = 1 us）。
 * 保证 ARR >= 1，避免 0 导致异常。 */
static inline uint32_t ticks_to_arr(uint32_t ticks)
{
    if (ticks < 2U) ticks = 2U;
    return ticks - 1U;
}

/* 每个 STEP 脉冲占两个中断周期（高 + 低），
 * 故一个完整步的周期 = 2 × ARR(tick)。
 * 为与 Arduino 版"每步 speed_us"语义一致，这里把 ARR 设为 speed_us/2。 */
static void timer_start(uint32_t speed_us)
{
    if (speed_us < MIN_STEP_US) speed_us = MIN_STEP_US;
    if (speed_us > MAX_STEP_US) speed_us = MAX_STEP_US;
    uint32_t half = speed_us / 2U;   /* 高/低各占半个周期 */
    __HAL_TIM_SET_AUTORELOAD(&htim_step, ticks_to_arr(half));
    __HAL_TIM_SET_COUNTER(&htim_step, 0);
    HAL_TIM_Base_Start_IT(&htim_step);
}

static void timer_stop(void)
{
    HAL_TIM_Base_Stop_IT(&htim_step);
    HAL_GPIO_WritePin(STEP_GPIO_Port, STEP_Pin, STEP_IDLE_LEVEL);
    s_step_phase = 0;
}

/* ================= 对外接口 ================= */
void motor_init(void)
{
    HAL_GPIO_WritePin(STEP_GPIO_Port, STEP_Pin, STEP_IDLE_LEVEL);
    dir_apply(true);
    motor_enable(false);

    s_steps_left = 0;
    s_position   = 0;
    s_speed_us   = DEFAULT_STEP_US;
    s_homing     = false;
    s_step_phase = 0;
}

void motor_enable(bool on)
{
    s_enabled = on;
    HAL_GPIO_WritePin(EN_GPIO_Port, EN_Pin,
                      on ? ENABLE_ACTIVE_LEVEL
                         : (ENABLE_ACTIVE_LEVEL == GPIO_PIN_SET
                                ? GPIO_PIN_RESET : GPIO_PIN_SET));
}

bool motor_is_enabled(void) { return s_enabled; }

void motor_move_steps(int32_t steps)
{
    if (steps == 0) return;
    if (motor_is_busy()) return;         /* 忙时不打断（可改为排队） */

    dir_apply(steps > 0);
    s_homing     = false;
    s_steps_left = (steps > 0) ? steps : -steps;
    timer_start(s_speed_us);
}

void motor_move_degrees(float deg)
{
    int32_t mag = (int32_t)(deg * 1000.0f);
    if (mag < 0) mag = -mag;
    int32_t steps = (int32_t)(((int64_t)mag * STEPS_PER_DEG_X1000) / 1000000LL);
    if (steps == 0) steps = 1;           /* 极小角度也走一步 */
    motor_move_steps((deg >= 0) ? steps : -steps);
}

void motor_home(void)
{
    if (motor_is_busy()) return;

    dir_apply(false);                    /* 朝负方向找限位 */
    s_homing     = true;
    s_steps_left = HOME_MAX_STEPS;       /* 上限，防跑飞 */
    timer_start(HOME_SPEED_US);
}

void motor_stop(void)
{
    timer_stop();
    s_steps_left = 0;
    s_homing     = false;
}

bool motor_is_busy(void) { return s_steps_left > 0; }

int32_t motor_position_steps(void) { return s_position; }

float motor_position_degrees(void)
{
    return (float)s_position * 360.0f / (float)STEPS_PER_REV;
}

void motor_set_speed_us(uint32_t us)
{
    if (us < MIN_STEP_US) us = MIN_STEP_US;
    if (us > MAX_STEP_US) us = MAX_STEP_US;
    s_speed_us = us;
    /* 运动中且非回零：立即生效（同样按半步计算 ARR） */
    if (motor_is_busy() && !s_homing) {
        __HAL_TIM_SET_AUTORELOAD(&htim_step,
                                 ticks_to_arr(s_speed_us / 2U));
    }
}

uint32_t motor_get_speed_us(void) { return s_speed_us; }

/* ------------------------------------------------------------------
 * 定时器中断回调：由 main.c 的 HAL_TIM_PeriodElapsedCallback 调用
 * ------------------------------------------------------------------ */
void motor_on_timer_tick(void)
{
    /* 回零：命中限位则停并清零 */
    if (s_homing &&
        HAL_GPIO_ReadPin(LIMIT_GPIO_Port, LIMIT_Pin)
            == LIMIT_TRIGGERED_LEVEL) {
        timer_stop();
        s_position = 0;
        s_homing   = false;
        s_steps_left = 0;
        return;
    }

    /* 走完 → 停 */
    if (s_steps_left <= 0) {
        timer_stop();
        return;
    }

    /* 两拍一个脉冲 */
    if (s_step_phase == 0) {
        HAL_GPIO_WritePin(STEP_GPIO_Port, STEP_Pin, STEP_ACTIVE_LEVEL);
        s_step_phase = 1;
    } else {
        HAL_GPIO_WritePin(STEP_GPIO_Port, STEP_Pin, STEP_IDLE_LEVEL);
        s_step_phase = 0;

        s_steps_left--;
        s_position += s_dir_positive ? 1 : -1;
    }
}
