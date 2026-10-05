/**
 * 串口命令解析模块 — 实现
 * ------------------------------------------------------------------
 * 采用"中断收字节进环形缓冲 + 主循环解析整行"的结构：
 *   - 中断只负责存字节，绝不阻塞、不做解析
 *   - 主循环 cmd_poll() 提取完整行并执行
 *
 * 依赖 main.c 提供的 uart 句柄 hcmd_uart（可指向 huart1 或 USB CDC）。
 * ------------------------------------------------------------------
 */
#include "cmd.h"
#include "config.h"
#include "motor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 由 main.c 提供的命令用 UART 句柄 */
extern UART_HandleTypeDef hcmd_uart;

/* ================= 接收缓冲（中断→主循环） ================= */
static volatile uint8_t  s_rx_ring[CMD_BUF_SIZE];
static volatile uint16_t s_rx_head = 0;
static volatile uint16_t s_rx_tail = 0;

/* 主循环解析用的行缓冲 */
static char s_line[CMD_BUF_SIZE];
static uint16_t s_line_len = 0;

/* ================= 发送 ================= */
void cmd_send_line(const char *s)
{
    size_t n = strlen(s);
    HAL_UART_Transmit(&hcmd_uart, (uint8_t *)s, (uint16_t)n, 100);
    const uint8_t crlf[2] = {'\r', '\n'};
    HAL_UART_Transmit(&hcmd_uart, (uint8_t *)crlf, 2, 100);
}

/* ================= 字节级中断喂入 ================= */
void cmd_feed_byte(uint8_t byte)
{
    uint16_t next = (uint16_t)((s_rx_head + 1) % CMD_BUF_SIZE);
    if (next != s_rx_tail) {          /* 未满则存 */
        s_rx_ring[s_rx_head] = byte;
        s_rx_head = next;
    }
    /* 满则丢弃该字节（避免死锁） */
}

/* ================= 从环形缓冲取字节 ================= */
static int ring_pop(uint8_t *out)
{
    if (s_rx_tail == s_rx_head) return 0;
    *out = s_rx_ring[s_rx_tail];
    s_rx_tail = (uint16_t)((s_rx_tail + 1) % CMD_BUF_SIZE);
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
    /* 去掉首尾空白 */
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
        int v = atoi(cmd + 6);
        motor_enable(v != 0);
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
        float deg = (float)atof(cmd + 3);  /* atof 兼容性优于 strtof */
        motor_move_degrees(deg);
        cmd_send_line("OK");

    } else if (strncmp(cmd, "STEPS", 5) == 0) {
        long n = atol(cmd + 5);
        motor_move_steps((int32_t)n);
        cmd_send_line("OK");

    } else if (strcmp(cmd, "STATUS") == 0) {
        char buf[TX_BUF_SIZE];
        int st = motor_is_busy() ? (1) : 0;  /* 简化：忙=MOVING */
        snprintf(buf, sizeof(buf),
                 "STATUS %s %ld %.2f %d",
                 state_name(st),
                 (long)motor_position_steps(),
                 (double)motor_position_degrees(),
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
            if (s_line_len < CMD_BUF_SIZE - 1) {
                s_line[s_line_len++] = (char)b;
            } else {
                s_line_len = 0;   /* 溢出保护：丢弃超长行 */
            }
        }
    }
}

/* ================= 初始化 ================= */
void cmd_init(void)
{
    s_rx_head = s_rx_tail = 0;
    s_line_len = 0;
}
