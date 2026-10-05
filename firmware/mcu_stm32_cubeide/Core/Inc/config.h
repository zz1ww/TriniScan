/**
 * TriniScan 转台控制器 — 编译期配置 (TIM1 PWM 版)
 * ------------------------------------------------------------------
 * 适配本工程实际引脚：
 *   STEP  = PA8  (TIM1_CH1 PWM)
 *   DIR   = PB0
 *   EN    = PB1
 *   LIMIT = PB2  (上拉)
 *   LED   = PC13
 * 所有可调参数集中在此。
 * ------------------------------------------------------------------
 */
#ifndef TRINISCAN_CONFIG_H
#define TRINISCAN_CONFIG_H

#include "main.h"

/* ================= 电机 / 驱动参数 ================= */
#define MOTOR_STEPS_PER_REV   200     /* 1.8° 步距角 → 200 整步/圈        */
#define MICROSTEP             32      /* TB6600 细分 (1/32)               */
#define GEAR_RATIO            5       /* 机械减速比                       */

/* 每圈总步数 = 200 × 32 × 5 = 32000 */
#define STEPS_PER_REV  ((int32_t)(MOTOR_STEPS_PER_REV * MICROSTEP * GEAR_RATIO))
#define STEPS_PER_DEG_X1000  (STEPS_PER_REV * 1000L / 360L)  /* 千分之一步/度 */

/* ================= 运动参数 ================= */
/* 脉冲频率 = 72MHz / (Prescaler+1) / (Period+1)   [TIM1, PSC=71]
 * 即 1MHz / (Period+1)。
 * 每步脉冲间隔(us) = Period+1，见 motor.c 的 speed_to_period()。 */
#define DEFAULT_STEP_US       900     /* 每步间隔 (us)，越大越慢          */
#define MIN_STEP_US           150     /* 最快                             */
#define MAX_STEP_US           5000    /* 最慢                             */
#define SETTLE_MS             200     /* 到位稳定延时 (配合"转-停-拍")     */
#define HOME_SPEED_US         1500    /* 回零慢速                         */
#define HOME_MAX_STEPS        (STEPS_PER_REV * 3 / 2)   /* 回零上限 = 1.5 圈 */

/* ================= 通信 ================= */
#define CMD_BAUD              115200
#define CMD_BUF_SIZE          64
#define TX_BUF_SIZE           96

/* ================= 极性 (按实际硬件调整) ================= */
#define ENABLE_ACTIVE_LEVEL   GPIO_PIN_RESET  /* ENA 低有效使能          */
#define DIR_POSITIVE_LEVEL    GPIO_PIN_SET    /* 正转 → DIR 高           */
#define LIMIT_TRIGGERED_LEVEL GPIO_PIN_RESET  /* 限位触发 → 读到低       */

/* ================= 状态机 ================= */
typedef enum {
    ST_IDLE = 0,
    ST_MOVING,
    ST_HOMING,
    ST_ERROR
} mcu_state_t;

#endif /* TRINISCAN_CONFIG_H */
