/**
 * 串口命令解析模块
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

#include <stdint.h>

void cmd_init(void);
void cmd_feed_byte(uint8_t byte);   /* USART 中断里调用 */
void cmd_poll(void);                /* 主循环调用 */
void cmd_send_line(const char *s);  /* 发一行（自动补 \r\n） */

#endif /* TRINISCAN_CMD_H */
