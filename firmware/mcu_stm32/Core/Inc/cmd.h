/**
 * 串口命令解析模块
 * ------------------------------------------------------------------
 * 协议与 Arduino 版完全一致 (115200, ASCII 行, '\n' 结尾)：
 *
 *   PING            → PONG
 *   HOME            → OK           回零
 *   ROT <deg>       → OK           相对转动角度（度，可为负）
 *   STEPS <n>       → OK           相对转动步数（可为负）
 *   ENABLE 0/1      → OK           使能/脱机
 *   SPEED <us>      → OK           设速度（微秒/步）
 *   STATUS          → STATUS <state> <pos_steps> <pos_deg> <busy>
 *   STOP            → OK           急停
 *
 * 收到 OK 后，上位机应等待 SETTLE_MS 再触发相机。
 * ------------------------------------------------------------------
 */
#ifndef TRINISCAN_CMD_H
#define TRINISCAN_CMD_H

#include <stdint.h>

/* 初始化命令模块（绑定 UART 句柄由 cmd.c 内部 extern 引用） */
void cmd_init(void);

/* 收到一个字节时调用（建议在 UART 接收中断里喂） */
void cmd_feed_byte(uint8_t byte);

/* 在主循环里调用，处理已收齐的整行命令 */
void cmd_poll(void);

/* 主动发送一行（自动补 '\r\n'），用于 READY / 状态上报 */
void cmd_send_line(const char *s);

#endif /* TRINISCAN_CMD_H */
