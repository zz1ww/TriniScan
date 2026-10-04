/**
 * TriniScan 转台控制器固件
 * ------------------------------------------------------------------
 * 通过串口接收上位机指令，控制 TB6600 驱动步进电机，
 * 实现精确转角与"转-停-拍"时序。
 *
 * 串口协议 (115200, ASCII 行, '\n' 结尾):
 *   PING            → PONG
 *   HOME            → 归零, OK
 *   ROT <deg>       → 相对转动角度(度), OK
 *   STEPS <n>       → 相对转动步数, OK
 *   ENABLE 1/0      → 使能/脱机, OK
 *   STATUS          → STATUS <state>
 *   STOP            → 停止, OK
 *
 * 注意: 上位机在收到 OK 后应等待 SETTLE_MS 再触发相机。
 * ------------------------------------------------------------------
 */

#include "config.h"

/* ========== 全局状态 ========== */
float  g_speed_us     = DEFAULT_SPEED_US;
long   g_total_steps  = 0;      // 累计相对步数（可正可负）
State  g_state        = STATE_IDLE;
String g_cmd_buf      = "";

/* ========== 基础 IO ========== */
static inline void step_high() { digitalWrite(PIN_STEP, HIGH); }
static inline void step_low()  { digitalWrite(PIN_STEP, LOW);  }

void enable_motor(bool on) {
    // TB6600 使能通常低有效
    digitalWrite(PIN_ENABLE, on ? LOW : HIGH);
}

/* ========== 单步 ========== */
void do_one_step() {
    step_high();
    delayMicroseconds(2);
    step_low();
    delayMicroseconds(g_speed_us);
}

/* ========== 按步数运动 ========== */
void move_steps(long steps) {
    g_state = STATE_MOVING;

    bool dir = (steps >= 0);
    digitalWrite(PIN_DIR, dir ? HIGH : LOW);
    long n = labs(steps);

    for (long i = 0; i < n; i++) {
        do_one_step();
    }

    g_total_steps += steps;
    delay(SETTLE_MS);
    g_state = STATE_IDLE;
}

/* ========== 按角度运动 ========== */
void move_degrees(float deg) {
    long steps = (long)lroundf(deg * STEPS_PER_DEG);
    move_steps(steps);
}

/* ========== 归零 ========== */
void do_home() {
    g_state = STATE_HOMING;

    // 以较慢速度向负方向寻找限位
    float old_speed = g_speed_us;
    g_speed_us = 1500;

    digitalWrite(PIN_DIR, LOW);   // 假设 LOW 朝限位方向
    // 最多退一圈半，防止无限转
    long max_steps = (long)(STEPS_PER_REV * 1.5);
    for (long i = 0; i < max_steps; i++) {
        if (digitalRead(PIN_LIMIT) == LOW) {   // 触发限位
            break;
        }
        do_one_step();
    }

    g_speed_us = old_speed;
    g_total_steps = 0;            // 归零后重置计数
    g_state = STATE_IDLE;
}

/* ========== 指令处理 ========== */
void handle_command(String cmd) {
    cmd.trim();
    if (cmd.length() == 0) return;

    if (cmd == "PING") {
        Serial.println("PONG");
    } else if (cmd == "HOME") {
        do_home();
        Serial.println("OK");
    } else if (cmd == "STATUS") {
        Serial.print("STATUS ");
        Serial.println((int)g_state);
    } else if (cmd == "STOP") {
        g_state = STATE_IDLE;
        Serial.println("OK");
    } else if (cmd.startsWith("ENABLE")) {
        int v = cmd.substring(6).toInt();
        enable_motor(v != 0);
        Serial.println("OK");
    } else if (cmd.startsWith("ROT ")) {
        float deg = cmd.substring(4).toFloat();
        move_degrees(deg);
        Serial.println("OK");
    } else if (cmd.startsWith("STEPS ")) {
        long n = cmd.substring(6).toInt();
        move_steps(n);
        Serial.println("OK");
    } else if (cmd.startsWith("SPEED ")) {
        long us = cmd.substring(6).toInt();
        if (us >= MIN_SPEED_US && us <= MAX_SPEED_US) {
            g_speed_us = us;
            Serial.println("OK");
        } else {
            Serial.println("ERR range");
        }
    } else {
        Serial.println("ERR unknown");
    }
}

/* ========== setup / loop ========== */
void setup() {
    Serial.begin(SERIAL_BAUD);

    pinMode(PIN_STEP,   OUTPUT);
    pinMode(PIN_DIR,    OUTPUT);
    pinMode(PIN_ENABLE, OUTPUT);
    pinMode(PIN_LIMIT,  INPUT_PULLUP);

    digitalWrite(PIN_STEP, LOW);
    digitalWrite(PIN_DIR,  LOW);
    enable_motor(true);

    g_state = STATE_IDLE;
    Serial.println("READY");
}

void loop() {
    while (Serial.available() > 0) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            if (g_cmd_buf.length() > 0) {
                handle_command(g_cmd_buf);
                g_cmd_buf = "";
            }
        } else {
            g_cmd_buf += c;
            if (g_cmd_buf.length() > 64) g_cmd_buf = "";  // 防溢出
        }
    }
}
