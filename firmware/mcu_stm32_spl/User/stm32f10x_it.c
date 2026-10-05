/**
 * 中断服务函数 (STM32F103C8T6 / 标准外设库)
 * ------------------------------------------------------------------
 * 只实现本工程用到的两个中断：TIM2(步进)、USART1(串口接收)。
 * 其余中断在标准库的 stm32f10x_it.c 里若已存在，请合并，避免重复定义。
 * ------------------------------------------------------------------
 */
#include "stm32f10x.h"
#include "cmd.h"
#include "motor.h"

/* ---- TIM2：步进脉冲 ---- */
void TIM2_IRQHandler(void)
{
    motor_timer_isr();
}

/* ---- USART1：串口接收 ---- */
void USART1_IRQHandler(void)
{
    if (USART_GetITStatus(USART1, USART_IT_RXNE) != RESET) {
        uint8_t b = (uint8_t)USART_ReceiveData(USART1);
        cmd_feed_byte(b);
        /* 读 DR 已自动清 RXNE */
    }

    /* 溢出错误处理，避免接收卡死 */
    if (USART_GetFlagStatus(USART1, USART_FLAG_ORE) != RESET) {
        (void)USART_ReceiveData(USART1);
    }
}
