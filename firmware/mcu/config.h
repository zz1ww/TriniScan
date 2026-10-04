/**
 * TriniScan 转台控制器 — 配置文件
 */
#ifndef CONFIG_H
#define CONFIG_H

/* ========== 引脚定义 ========== */
#define PIN_STEP        2       // TB6600 PUL-
#define PIN_DIR         3       // TB6600 DIR-
#define PIN_ENABLE      4       // TB6600 ENA-
#define PIN_LIMIT       5       // 归零限位开关 (INPUT_PULLUP)

/* ========== 电机/驱动参数 ========== */
#define MOTOR_STEPS_PER_REV   200    // 1.8° 步距角 → 200 整步/圈
#define MICROSTEP             32     // TB6600 细分 (1/32)
#define GEAR_RATIO            5.0f   // 机械减速比

/* 每圈总步数 = 200 × 32 × 5 = 32000 */
#define STEPS_PER_REV  ((long)(MOTOR_STEPS_PER_REV * MICROSTEP * GEAR_RATIO))
#define STEPS_PER_DEG  (STEPS_PER_REV / 360.0f)

/* ========== 运动参数 ========== */
#define DEFAULT_SPEED_US      800    // 每步脉冲间隔 (微秒)，越小越快
#define MIN_SPEED_US          200
#define MAX_SPEED_US          3000
#define SETTLE_MS             200    // 到位后稳定延时

/* ========== 通信 ========== */
#define SERIAL_BAUD           115200

/* 状态枚举 */
enum State {
    STATE_IDLE = 0,
    STATE_MOVING,
    STATE_HOMING,
    STATE_ERROR
};

#endif // CONFIG_H
