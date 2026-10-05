/**
 * 串口命令解析模块 — 标准库版
 * ------------------------------------------------------------------
 * 协议 (115200, ASCII 行, '\n' 结尾)：
 *   PING            → PONG
 *   HOME            → OK
 *   ROT <deg>       → OK
 *   STEPS <n>       → OK
 *   ENABLE 0/1      → OK
 *   SPEED <us>      → OK
 *   STATUS          → STATUS <state> <steps> <deg> <busy>
 *   STOP            → OK
 * ------------------------------------------------------------------
 */
#ifndef TRINISCAN_CMD_H
#define TRINISCAN_CMD_H

#include "config.h"

/* 初始化命令模块（内部会初始化 USART1） */
void cmd_init(void);

/* 收到一个字节时调用（在 USART1_IRQHandler 里喂） */
void cmd_feed_byte(uint8_t byte);

/* 主循环里调用，处理已收齐的整行命令 */
void cmd_poll(void);

/* 主动发送一行（自动补 '\r\n'） */
void cmd_send_line(const char *s);

#endif /* TRINISCAN_CMD_H */
