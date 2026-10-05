/**
 * TriniScan 转台控制器 (STM32F103C8T6 / 标准外设库) — 编译期配置
 * ------------------------------------------------------------------
 * 所有可调参数集中在此，改这里即可，不必动主逻辑。
 * 与 HAL 版 firmware/mcu_stm32/Core/Inc/config.h 语义完全一致。
 * ------------------------------------------------------------------
 */
#ifndef TRINISCAN_CONFIG_H
#define TRINISCAN_CONFIG_H

#include "stm32f10x.h"

/* ================= 引脚定义 (改这里即可换引脚) ================= */
#define STEP_GPIO_PORT      GPIOA
#define STEP_GPIO_PIN       GPIO_Pin_0
#define STEP_GPIO_CLK       RCC_APB2Periph_GPIOA

#define DIR_GPIO_PORT       GPIOA
#define DIR_GPIO_PIN        GPIO_Pin_1
#define DIR_GPIO_CLK        RCC_APB2Periph_GPIOA

#define EN_GPIO_PORT        GPIOA
#define EN_GPIO_PIN         GPIO_Pin_2
#define EN_GPIO_CLK         RCC_APB2Periph_GPIOA

#define LIMIT_GPIO_PORT     GPIOB
#define LIMIT_GPIO_PIN      GPIO_Pin_0
#define LIMIT_GPIO_CLK      RCC_APB2Periph_GPIOB

#define LED_GPIO_PORT       GPIOC
#define LED_GPIO_PIN        GPIO_Pin_13
#define LED_GPIO_CLK        RCC_APB2Periph_GPIOC

/* ================= 电机 / 驱动参数 ================= */
#define MOTOR_STEPS_PER_REV   200     /* 整步/圈 (1.8° 步距角)          */
#define MICROSTEP             32      /* TB6600 细分 (1/32)             */
#define GEAR_RATIO            5       /* 机械减速比 (整数)              */

/* 每圈总步数 = 200 × 32 × 5 = 32000 */
#define STEPS_PER_REV  ((int32_t)(MOTOR_STEPS_PER_REV * MICROSTEP * GEAR_RATIO))
#define STEPS_PER_DEG_X1000  (STEPS_PER_REV * 1000L / 360L)  /* 千分之一步/度 */

/* ================= 运动参数 ================= */
#define DEFAULT_STEP_US       900     /* 每步脉冲间隔 (微秒)，越大越慢      */
#define MIN_STEP_US           150     /* 最快 (受限于电机/驱动)            */
#define MAX_STEP_US           5000    /* 最慢                              */
#define SETTLE_MS             200     /* 到位后稳定延时 (配合"转-停-拍")    */
#define HOME_SPEED_US         1500    /* 回零时的慢速                      */
#define HOME_MAX_STEPS        (STEPS_PER_REV * 3 / 2)  /* 回零最大行程=1.5圈 */

/* ================= 通信 ================= */
#define CMD_BAUD              115200
#define CMD_BUF_SIZE          64      /* 环形缓冲大小                     */
#define TX_BUF_SIZE           96      /* 发送缓冲                         */

/* ================= 极性 (按实际硬件调整) ================= */
/*
 * TB6600 的 PUL/DIR/ENA 均为共阴(光耦)接法时通常"高电平有效"，
 * 共阳接法时"低电平有效"。按你的接线改下面三个即可，无需改逻辑。
 */
#define STEP_ACTIVE_LEVEL     1       /* 脉冲有效电平: 1=高有效 */
#define STEP_IDLE_LEVEL       0
#define ENABLE_ACTIVE_LEVEL   0       /* ENA 有效电平: 0=低有效使能 */
#define DIR_POSITIVE_LEVEL    1       /* DIR 正转电平: 1=正转 DIR 高 */

/* 限位开关: 上拉输入, 触发时读到低电平                              */
#define LIMIT_TRIGGERED_LEVEL 0

/* ================= 状态机 ================= */
typedef enum {
    ST_IDLE = 0,
    ST_MOVING,
    ST_HOMING,
    ST_ERROR
} mcu_state_t;

#endif /* TRINISCAN_CONFIG_H */
