/**
 * 步进电机驱动模块 — TIM1 PWM + 中断计数（精确版）
 * ------------------------------------------------------------------
 * 原理：
 *   TIM1 配为 1 MHz 计数（PSC=71），PWM1 模式，50% 占空比。
 *   PWM 周期 = Period+1 (us) = 每步脉冲间隔。
 *   开启 PWM 后 PA8 自动输出方波，每个周期 = 一个 STEP 脉冲。
 *
 *   每次计数溢出触发 TIM1 更新中断：
 *     中断里把剩余步数 -1；减到 0 立即停 PWM。
 *   → 步数由硬件精确保证，不受主循环影响。
 *
 * ⚠️ 前置条件（CubeMX 必须配置）：
 *   Timers → TIM1 → NVIC Settings → 勾选 "TIM1 update interrupt"
 *   否则中断计数不工作。
 *
 * ⚠️ TIM1 是高级定时器，更新中断入口是 TIM1_UP_IRQHandler，
 *    CubeMX 会在 stm32f1xx_it.c 里生成它，最终调用
 *    HAL_TIM_PeriodElapsedCallback()（我们在下面重写）。
 * ------------------------------------------------------------------
 */
#include "motor.h"
#include "config.h"

extern TIM_HandleTypeDef htim1;

/* STEP = PA8 = TIM1_CH1（复用功能，CubeMX 不生成 GPIO 宏，这里手动定义） */
#define STEP_GPIO_Port  GPIOA
#define STEP_Pin        GPIO_PIN_8

/* ================= 内部状态 ================= */
static volatile int32_t  s_steps_left   = 0;   /* 剩余步数 */
static volatile int32_t  s_position     = 0;   /* 累计步数 */
static volatile uint32_t s_speed_us     = DEFAULT_STEP_US;
static volatile bool     s_enabled      = false;
static volatile bool     s_homing       = false;
static volatile bool     s_dir_positive = true;

/* ================= 内部辅助 ================= */
static void dir_apply(bool positive)
{
    s_dir_positive = positive;
    HAL_GPIO_WritePin(DIR_GPIO_Port, DIR_Pin,
                      positive ? DIR_POSITIVE_LEVEL
                               : (DIR_POSITIVE_LEVEL == GPIO_PIN_SET
                                      ? GPIO_PIN_RESET : GPIO_PIN_SET));
}

static void en_apply(bool on)
{
    s_enabled = on;
    HAL_GPIO_WritePin(EN_GPIO_Port, EN_Pin,
                      on ? ENABLE_ACTIVE_LEVEL
                         : (ENABLE_ACTIVE_LEVEL == GPIO_PIN_SET
                                ? GPIO_PIN_RESET : GPIO_PIN_SET));
}

/* 步间隔(us) → TIM1 Period（1MHz 计数，Period = us-1） */
static uint32_t speed_to_period(uint32_t us)
{
    if (us < MIN_STEP_US) us = MIN_STEP_US;
    if (us > MAX_STEP_US) us = MAX_STEP_US;
    return us - 1U;
}

/* 把 PA8 配置为 TIM1_CH1 复用推挽输出（AF_PP），供 PWM 使用 */
static void step_pin_to_af(void)
{
    GPIO_InitTypeDef gi = {0};
    gi.Pin   = STEP_Pin;
    gi.Mode  = GPIO_MODE_AF_PP;
    gi.Pull  = GPIO_NOPULL;
    gi.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(STEP_GPIO_Port, &gi);
}

/* 启动 PWM + 更新中断 */
static void pwm_start(uint32_t speed_us)
{
    uint32_t period = speed_to_period(speed_us);

    step_pin_to_af();                      /* 确保 PA8 复用态 */
    __HAL_TIM_SET_AUTORELOAD(&htim1, period);
    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_1, period / 2U);  /* 50% */
    __HAL_TIM_SET_COUNTER(&htim1, 0);

    /* 使能更新中断并清标志 */
    __HAL_TIM_CLEAR_IT(&htim1, TIM_IT_UPDATE);
    __HAL_TIM_ENABLE_IT(&htim1, TIM_IT_UPDATE);

    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
}

/* 停止 PWM 与中断；PA8 保持复用态（TIM1 停止后输出为无效电平，
 * 对 TB6600 而言无脉冲即为停止，无需切成 GPIO） */
static void pwm_stop(void)
{
    HAL_TIM_PWM_Stop(&htim1, TIM_CHANNEL_1);
    __HAL_TIM_DISABLE_IT(&htim1, TIM_IT_UPDATE);
}

/* ================= 对外接口 ================= */
void motor_init(void)
{
    pwm_stop();
    dir_apply(true);
    en_apply(false);

    s_steps_left = 0;
    s_position   = 0;
    s_speed_us   = DEFAULT_STEP_US;
    s_homing     = false;
}

void motor_enable(bool on) { en_apply(on); }
bool motor_is_enabled(void) { return s_enabled; }

void motor_move_steps(int32_t steps)
{
    if (steps == 0) return;
    if (motor_is_busy()) return;

    dir_apply(steps > 0);
    s_homing     = false;
    s_steps_left = (steps > 0) ? steps : -steps;

    pwm_start(s_speed_us);
}

void motor_move_degrees(float deg)
{
    int32_t mag = (int32_t)(deg * 1000.0f);
    if (mag < 0) mag = -mag;
    int32_t steps = (int32_t)(((int64_t)mag * STEPS_PER_DEG_X1000) / 1000000LL);
    if (steps == 0) steps = 1;
    motor_move_steps((deg >= 0) ? steps : -steps);
}

void motor_home(void)
{
    if (motor_is_busy()) return;

    dir_apply(false);
    s_homing     = true;
    s_steps_left = HOME_MAX_STEPS;
    pwm_start(HOME_SPEED_US);
}

void motor_stop(void)
{
    pwm_stop();
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
    /* 运动中且非回零：立即改速度 */
    if (motor_is_busy() && !s_homing) {
        uint32_t p = speed_to_period(us);
        __HAL_TIM_SET_AUTORELOAD(&htim1, p);
        __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_1, p / 2U);
    }
}

uint32_t motor_get_speed_us(void) { return s_speed_us; }

/* ------------------------------------------------------------------
 * TIM1 更新中断回调（由 HAL_TIM_PeriodElapsedCallback 调用）
 * 每个更新事件 = 一个 STEP 脉冲
 * ------------------------------------------------------------------ */
void motor_on_tick(void)
{
    /* 回零：命中限位立即停 */
    if (s_homing &&
        HAL_GPIO_ReadPin(LIMIT_GPIO_Port, LIMIT_Pin) == LIMIT_TRIGGERED_LEVEL) {
        pwm_stop();
        s_position   = 0;
        s_homing     = false;
        s_steps_left = 0;
        return;
    }

    if (s_steps_left <= 0) {
        pwm_stop();
        return;
    }

    s_steps_left--;
    s_position += s_dir_positive ? 1 : -1;

    if (s_steps_left == 0) {
        pwm_stop();
    }
}
