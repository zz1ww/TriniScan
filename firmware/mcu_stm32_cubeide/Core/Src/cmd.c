/**
 * 串口命令解析模块 — 实现
 * ------------------------------------------------------------------
 * 中断只存字节进环形缓冲；主循环 cmd_poll() 提取整行并执行。
 * ------------------------------------------------------------------
 */
#include "cmd.h"
#include "config.h"
#include "motor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern UART_HandleTypeDef huart1;

/* ================= 接收环形缓冲 ================= */
static volatile uint8_t  s_ring[CMD_BUF_SIZE];
static volatile uint16_t s_head = 0;
static volatile uint16_t s_tail = 0;

static char     s_line[CMD_BUF_SIZE];
static uint16_t s_line_len = 0;

/* ================= 发送 ================= */
void cmd_send_line(const char *s)
{
    size_t n = strlen(s);
    HAL_UART_Transmit(&huart1, (uint8_t *)s, (uint16_t)n, 100);
    const uint8_t crlf[2] = {'\r', '\n'};
    HAL_UART_Transmit(&huart1, (uint8_t *)crlf, 2, 100);
}

/* ================= 中断喂字节 ================= */
void cmd_feed_byte(uint8_t byte)
{
    uint16_t next = (uint16_t)((s_head + 1) % CMD_BUF_SIZE);
    if (next != s_tail) {
        s_ring[s_head] = byte;
        s_head = next;
    }
}

static int ring_pop(uint8_t *out)
{
    if (s_tail == s_head) return 0;
    *out = s_ring[s_tail];
    s_tail = (uint16_t)((s_tail + 1) % CMD_BUF_SIZE);
    return 1;
}

/* ================= 命令执行 ================= */
static const char *state_name(int st)
{
    switch (st) {
    case 0: return "IDLE";
    case 1: return "MOVING";
    case 2: return "HOMING";
    default: return "ERROR";
    }
}

static void exec_command(char *cmd)
{
    while (*cmd == ' ' || *cmd == '\t') cmd++;

    if (strcmp(cmd, "PING") == 0) {
        cmd_send_line("PONG");
    } else if (strcmp(cmd, "HOME") == 0) {
        motor_home();
        cmd_send_line("OK");
    } else if (strcmp(cmd, "STOP") == 0) {
        motor_stop();
        cmd_send_line("OK");
    } else if (strncmp(cmd, "ENABLE", 6) == 0) {
        motor_enable(atoi(cmd + 6) != 0);
        cmd_send_line("OK");
    } else if (strncmp(cmd, "SPEED", 5) == 0) {
        long us = atol(cmd + 5);
        if (us < MIN_STEP_US || us > MAX_STEP_US) {
            cmd_send_line("ERR range");
        } else {
            motor_set_speed_us((uint32_t)us);
            cmd_send_line("OK");
        }
    } else if (strncmp(cmd, "ROT", 3) == 0) {
        motor_move_degrees((float)atof(cmd + 3));
        cmd_send_line("OK");
    } else if (strncmp(cmd, "STEPS", 5) == 0) {
        motor_move_steps((int32_t)atol(cmd + 5));
        cmd_send_line("OK");
    } else if (strcmp(cmd, "STATUS") == 0) {
        char buf[TX_BUF_SIZE];
        int st = motor_is_busy() ? 1 : 0;
        /* 角度用整数×100 输出，避免依赖浮点 printf */
        int32_t deg_x100 = (int32_t)(motor_position_degrees() * 100.0f);
        snprintf(buf, sizeof(buf),
                 "STATUS %s %ld %ld.%02ld %d",
                 state_name(st),
                 (long)motor_position_steps(),
                 (long)(deg_x100 / 100),
                 (long)(deg_x100 % 100 < 0 ? -(deg_x100 % 100) : deg_x100 % 100),
                 motor_is_busy() ? 1 : 0);
        cmd_send_line(buf);
    } else if (cmd[0] == '\0') {
        /* 空行忽略 */
    } else {
        cmd_send_line("ERR unknown");
    }
}

/* ================= 主循环轮询 ================= */
void cmd_poll(void)
{
    uint8_t b;
    while (ring_pop(&b)) {
        if (b == '\n' || b == '\r') {
            if (s_line_len > 0) {
                s_line[s_line_len] = '\0';
                exec_command(s_line);
                s_line_len = 0;
            }
        } else {
            if (s_line_len < CMD_BUF_SIZE - 1)
                s_line[s_line_len++] = (char)b;
            else
                s_line_len = 0;
        }
    }
}

void cmd_init(void)
{
    s_head = s_tail = 0;
    s_line_len = 0;
}
