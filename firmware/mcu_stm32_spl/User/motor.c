/**
 * 步进电机驱动模块 — 标准外设库实现
 * ------------------------------------------------------------------
 * 定时器 TIM2 配置为 1 MHz 计数（72MHz / 72）。
 * 更新中断里按 "拉高→拉低" 两拍生成一个 STEP 脉冲，每两拍计一步。
 *
 * 中断频率由 ARR 决定：半周期 = speed_us/2，故整步周期 = speed_us。
 * ⚠️ 若使用其他定时器，同步改 motor_timer_isr() 里清标志的寄存器 (TIMx->SR)。
 * ------------------------------------------------------------------
 */
#include "motor.h"

/* ================= 内部状态 ================= */
static volatile int32_t  s_steps_left   = 0;
static volatile int32_t  s_position     = 0;
static volatile uint32_t s_speed_us     = DEFAULT_STEP_US;
static volatile uint8_t  s_enabled      = 0;
static volatile uint8_t  s_homing       = 0;
static volatile uint8_t  s_dir_positive = 1;
static volatile uint8_t  s_step_phase   = 0;   /* 0 空闲 1 已拉高 */

/* ================= 内部辅助 ================= */
static inline void step_high(void)
{
    if (STEP_ACTIVE_LEVEL)
        GPIO_SetBits(STEP_GPIO_PORT, STEP_GPIO_PIN);
    else
        GPIO_ResetBits(STEP_GPIO_PORT, STEP_GPIO_PIN);
}

static inline void step_low(void)
{
    if (STEP_IDLE_LEVEL)
        GPIO_SetBits(STEP_GPIO_PORT, STEP_GPIO_PIN);
    else
        GPIO_ResetBits(STEP_GPIO_PORT, STEP_GPIO_PIN);
}

static void dir_apply(uint8_t positive)
{
    s_dir_positive = positive;
    if (positive == (DIR_POSITIVE_LEVEL ? 1 : 0))
        GPIO_SetBits(DIR_GPIO_PORT, DIR_GPIO_PIN);
    else
        GPIO_ResetBits(DIR_GPIO_PORT, DIR_GPIO_PIN);
}

/* tick 数换算为 ARR（1 tick = 1 us），保证 >= 1 */
static inline uint16_t ticks_to_arr(uint32_t ticks)
{
    if (ticks < 2U) ticks = 2U;
    if (ticks > 0xFFFFU) ticks = 0xFFFFU;
    return (uint16_t)(ticks - 1U);
}

static void timer_start(uint32_t speed_us)
{
    if (speed_us < MIN_STEP_US) speed_us = MIN_STEP_US;
    if (speed_us > MAX_STEP_US) speed_us = MAX_STEP_US;

    TIM2->CNT = 0;
    TIM2->ARR = ticks_to_arr(speed_us / 2U);   /* 半周期 */
    TIM2->CR1 |= TIM_CR1_CEN;                  /* 启动 */
}

static void timer_stop(void)
{
    TIM2->CR1 &= (uint16_t)~TIM_CR1_CEN;
    step_low();
    s_step_phase = 0;
}

/* ================= GPIO / 定时器初始化 ================= */
static void gpio_init(void)
{
    GPIO_InitTypeDef gi;

    RCC_APB2PeriphClockCmd(STEP_GPIO_CLK | DIR_GPIO_CLK | EN_GPIO_CLK |
                           LED_GPIO_CLK, ENABLE);
    RCC_APB2PeriphClockCmd(LIMIT_GPIO_CLK, ENABLE);

    /* STEP / DIR / EN / LED 推挽输出 */
    gi.GPIO_Mode  = GPIO_Mode_Out_PP;
    gi.GPIO_Speed = GPIO_Speed_50MHz;

    gi.GPIO_Pin = STEP_GPIO_PIN;  GPIO_Init(STEP_GPIO_PORT,  &gi);
    gi.GPIO_Pin = DIR_GPIO_PIN;   GPIO_Init(DIR_GPIO_PORT,   &gi);
    gi.GPIO_Pin = EN_GPIO_PIN;    GPIO_Init(EN_GPIO_PORT,    &gi);
    gi.GPIO_Pin = LED_GPIO_PIN;   GPIO_Init(LED_GPIO_PORT,   &gi);

    /* LIMIT 上拉输入 */
    gi.GPIO_Mode = GPIO_Mode_IPU;
    gi.GPIO_Pin  = LIMIT_GPIO_PIN;
    GPIO_Init(LIMIT_GPIO_PORT, &gi);
}

static void timer_init(void)
{
    TIM_TimeBaseInitTypeDef ti;
    NVIC_InitTypeDef        ni;

    RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM2, ENABLE);

    /* 72 MHz / 72 = 1 MHz → 1 tick = 1 us */
    ti.TIM_Prescaler     = 72 - 1;
    ti.TIM_CounterMode   = TIM_CounterMode_Up;
    ti.TIM_Period        = 999;         /* 初值，运行时会改 */
    ti.TIM_ClockDivision = TIM_CKD_DIV1;
    ti.TIM_RepetitionCounter = 0;
    TIM_TimeBaseInit(TIM2, &ti);

    TIM_ClearITPendingBit(TIM2, TIM_IT_Update);
    TIM_ITConfig(TIM2, TIM_IT_Update, ENABLE);

    ni.NVIC_IRQChannel                   = TIM2_IRQn;
    ni.NVIC_IRQChannelPreemptionPriority = 1;
    ni.NVIC_IRQChannelSubPriority        = 0;
    ni.NVIC_IRQChannelCmd                = ENABLE;
    NVIC_Init(&ni);

    /* 先不启动，等运动时再 CEN */
}

/* ================= 对外接口 ================= */
void motor_init(void)
{
    gpio_init();
    timer_init();

    step_low();
    dir_apply(1);
    motor_enable(0);

    s_steps_left = 0;
    s_position   = 0;
    s_speed_us   = DEFAULT_STEP_US;
    s_homing     = 0;
    s_step_phase = 0;
}

void motor_enable(uint8_t on)
{
    s_enabled = on;
    if (on == (ENABLE_ACTIVE_LEVEL ? 1 : 0))
        GPIO_SetBits(EN_GPIO_PORT, EN_GPIO_PIN);
    else
        GPIO_ResetBits(EN_GPIO_PORT, EN_GPIO_PIN);
}

uint8_t motor_is_enabled(void) { return s_enabled; }

void motor_move_steps(int32_t steps)
{
    if (steps == 0) return;
    if (motor_is_busy()) return;

    dir_apply(steps > 0 ? 1 : 0);
    s_homing     = 0;
    s_steps_left = (steps > 0) ? steps : -steps;
    timer_start(s_speed_us);
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

    dir_apply(0);                /* 朝负方向找限位 */
    s_homing     = 1;
    s_steps_left = HOME_MAX_STEPS;
    timer_start(HOME_SPEED_US);
}

void motor_stop(void)
{
    timer_stop();
    s_steps_left = 0;
    s_homing     = 0;
}

uint8_t motor_is_busy(void) { return s_steps_left > 0; }

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
    if (motor_is_busy() && !s_homing)
        TIM2->ARR = ticks_to_arr(s_speed_us / 2U);
}

uint32_t motor_get_speed_us(void) { return s_speed_us; }

/* ------------------------------------------------------------------
 * 定时器更新中断服务：由 TIM2_IRQHandler 调用
 * ------------------------------------------------------------------ */
void motor_timer_isr(void)
{
    if (TIM_GetITStatus(TIM2, TIM_IT_Update) == RESET) return;
    TIM_ClearITPendingBit(TIM2, TIM_IT_Update);

    /* 回零：命中限位则停并清零 */
    if (s_homing &&
        GPIO_ReadInputDataBit(LIMIT_GPIO_PORT, LIMIT_GPIO_PIN) ==
            (uint8_t)LIMIT_TRIGGERED_LEVEL) {
        timer_stop();
        s_position   = 0;
        s_homing     = 0;
        s_steps_left = 0;
        return;
    }

    if (s_steps_left <= 0) {
        timer_stop();
        return;
    }

    /* 两拍一个脉冲 */
    if (s_step_phase == 0) {
        step_high();
        s_step_phase = 1;
    } else {
        step_low();
        s_step_phase = 0;
        s_steps_left--;
        s_position += s_dir_positive ? 1 : -1;
    }
}
