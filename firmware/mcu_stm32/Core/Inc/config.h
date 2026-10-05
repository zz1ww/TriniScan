/**
 * TriniScan 转台控制器 (STM32F103C8T6) — 编译期配置
 * ------------------------------------------------------------------
 * 所有可调参数集中在此，改这里即可，不必动主逻辑。
 * 与 Arduino 版 firmware/mcu/config.h 保持语义一致。
 * ------------------------------------------------------------------
 */
#ifndef TRINISCAN_CONFIG_H
#define TRINISCAN_CONFIG_H

#include "main.h"          /* 由 CubeMX 生成，提供引脚宏 */

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
#define CMD_BUF_SIZE          64      /* 命令行缓冲                       */
#define TX_BUF_SIZE           96      /* 发送缓冲                         */

/* ================= 极性 (按实际硬件调整) ================= */
/*
 * TB6600 的 PUL/DIR/ENA 均为共阳(光耦)接法时，通常为"低电平有效"。
 * 本固件用开漏/推挽输出到光耦负极，故：信号有效 = 引脚 LOW。
 * 如你的接线不同，改这两个宏即可，无需改逻辑。
 */
#define STEP_ACTIVE_LEVEL     GPIO_PIN_SET    /* 脉冲上升沿有效 */
#define STEP_IDLE_LEVEL       GPIO_PIN_RESET
#define ENABLE_ACTIVE_LEVEL   GPIO_PIN_RESET  /* ENA 低有效使能 */

/* DIR: 设为 SET 表示"正转对应 DIR 高"                                     */
#define DIR_POSITIVE_LEVEL    GPIO_PIN_SET

/* 限位开关: INPUT_PULLUP, 触发时读到低电平                              */
#define LIMIT_TRIGGERED_LEVEL GPIO_PIN_RESET

/* ================= 状态机 ================= */
typedef enum {
    ST_IDLE = 0,
    ST_MOVING,
    ST_HOMING,
    ST_ERROR
} mcu_state_t;

#endif /* TRINISCAN_CONFIG_H */
