/**
 * TriniScan 转台控制器 — 主程序 (STM32F103C8T6 / 标准外设库)
 * ------------------------------------------------------------------
 * 串口协议与 HAL 版、Arduino 版完全一致，上位机无需改动。
 * 系统时钟 72 MHz (HSE 8MHz × 9)。
 * ------------------------------------------------------------------
 */
#include "config.h"
#include "motor.h"
#include "cmd.h"

/* ================= 系统时钟配置 ================= */
static void SystemClock_Config(void)
{
    ErrorStatus HSEStartUpStatus;
    RCC_DeInit();
    RCC_HSEConfig(RCC_HSE_ON);

    HSEStartUpStatus = RCC_WaitForHSEStartUp();
    if (HSEStartUpStatus == SUCCESS) {
        /* Flash: 2 等待周期 (72MHz) */
        FLASH_PrefetchBufferCmd(FLASH_PrefetchBuffer_Enable);
        FLASH_SetLatency(FLASH_Latency_2);

        RCC_HCLKConfig(RCC_SYSCLK_Div1);   /* AHB  = 72 MHz */
        RCC_PCLK2Config(RCC_HCLK_Div1);    /* APB2 = 72 MHz */
        RCC_PCLK1Config(RCC_HCLK_Div2);    /* APB1 = 36 MHz */

        /* PLL: HSE × 9 = 72 MHz */
        RCC_PLLConfig(RCC_PLLSource_HSE_Div1, RCC_PLLMul_9);
        RCC_PLLCmd(ENABLE);
        while (RCC_GetFlagStatus(RCC_FLAG_PLLRDY) == RESET) { }

        RCC_SYSCLKConfig(RCC_SYSCLKSource_PLLCLK);
        while (RCC_GetSYSCLKSource() != 0x08) { }
    } else {
        /* HSE 起振失败 → 死循环（检查晶振） */
        while (1) { }
    }
}

/* ================= 简单软延时 ================= */
static void delay_ms(uint32_t ms)
{
    uint32_t i;
    while (ms--) {
        for (i = 0; i < 7200; i++) { __NOP(); }  /* 约 1ms @72MHz */
    }
}

/* ================= 主函数 ================= */
int main(void)
{
    SystemClock_Config();

    motor_init();          /* GPIO + TIM2 */
    cmd_init();            /* USART1 + 中断 */
    motor_enable(1);       /* 使能电机 */

    /* 让电机驱动/上位机就绪 */
    delay_ms(1000);
    cmd_send_line("READY");

    /* 上电点亮 LED */
    GPIO_SetBits(LED_GPIO_PORT, LED_GPIO_PIN);

    while (1) {
        cmd_poll();
    }
}
