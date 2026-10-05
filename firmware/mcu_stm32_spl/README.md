# TriniScan 转台控制器固件（STM32F103C8T6 + 标准外设库 / Keil5）

基于 **STM32F10x 标准外设库（SPL）** 的转台控制固件，
**Keil MDK5 工程**，功能与 HAL 版、Arduino 版**完全等价**，
串口协议一致，上位机无需任何改动。

## ⚠️ 先读这个

本目录**只提供应用代码**，**不含 ST 标准外设库**。
标准库 ST 已停止维护，需你自行准备（获取方式见
[`Doc/Keil搭建说明.md`](Doc/Keil搭建说明.md) 第一节）。

**最快上手路径**：
1. 按 `Doc/Keil搭建说明.md` 准备好标准库文件
2. 双击 `Project/TriniScan.uvprojx`
3. 核对 Include Paths 和 Define
4. 编译 → 烧录 → 串口测 `PING`

## 目录结构

```
mcu_stm32_spl/
├── README.md                     本文件
├── Doc/
│   └── Keil搭建说明.md            ★ 库获取 + 建工程 + 排错（必读）
├── User/                         ★ 应用代码（全部由本目录提供）
│   ├── main.c                    主程序（时钟 + 主循环）
│   ├── motor.c / motor.h         步进电机驱动（TIM2 中断脉冲）
│   ├── cmd.c   / cmd.h           串口命令解析（环形缓冲）
│   ├── config.h                  编译期配置（引脚/速度/极性）
│   └── stm32f10x_it.c            中断服务（TIM2 + USART1）
├── Libraries/                    ← 需你放入标准库
│   ├── CMSIS/
│   └── STM32F10x_StdPeriph_Driver/
├── Start/                        ← 需你放入启动文件
│   └── startup_stm32f10x_md.s
└── Project/
    └── TriniScan.uvprojx         Keil 工程模板
```

## 引脚分配

| 功能 | 引脚 | 说明 |
|---|---|---|
| STEP | PA0 | → TB6600 PUL- |
| DIR | PA1 | → TB6600 DIR- |
| EN | PA2 | → TB6600 ENA- |
| LIMIT | PB0 | 回零限位，上拉，低有效 |
| LED | PC13 | 板载，状态指示 |
| USART1 TX | PA9 | → 上位机 RX |
| USART1 RX | PA10 | ← 上位机 TX |
| SWDIO | PA13 | ST-Link |
| SWCLK | PA14 | ST-Link |

> 换引脚：改 `User/config.h` 顶部的 `*_GPIO_PORT / *_GPIO_PIN / *_GPIO_CLK`。

## 串口协议（与 HAL / Arduino 版一致）

| 命令 | 响应 | 说明 |
|---|---|---|
| `PING` | `PONG` | 连通测试 |
| `HOME` | `OK` | 回零 |
| `ROT <deg>` | `OK` | 相对转动角度（度，可负） |
| `STEPS <n>` | `OK` | 相对转动步数（可负） |
| `ENABLE 0/1` | `OK` | 脱机 / 使能 |
| `SPEED <us>` | `OK` | 设速度（微秒/步，150~5000） |
| `STATUS` | `STATUS <state> <steps> <deg> <busy>` | 查询状态 |
| `STOP` | `OK` | 急停 |

- 波特率 **115200**，ASCII，`\n` 结尾
- 上电发 `READY`
- 上位机收到 `OK` 后应等待 `SETTLE_MS`（默认 200ms）再触发相机

## 关键设计

```
     ┌──────────────┐
     │ USART1 中断   │──┐  只存字节
     └──────────────┘  │
                       ▼
                 环形缓冲 (64B)
                       │
     ┌──────────────┐  │
     │ 主循环 cmd_poll│◄─┘  提取整行 → 解析 → 执行
     └──────┬───────┘
            ▼
     ┌──────────────┐
     │ motor 模块    │  设定剩余步数
     └──────┬───────┘
            ▼
     ┌──────────────┐
     │ TIM2 中断     │  每两拍发一个 STEP 脉冲
     └──────────────┘
```

- **中断只做最少的事**（存字节 / 翻引脚 / 清标志）
- **解析在主循环**，避免中断里跑 `strtof`/`snprintf`
- **运动由硬件定时器驱动**，精度不受主循环影响
- 运动**非阻塞**：转台转动时上位机仍可查询状态、急停

## 三版固件对照

| | Arduino 版 | HAL 版 | **标准库版（本目录）** |
|---|---|---|---|
| 位置 | `firmware/mcu/` | `firmware/mcu_stm32/` | `firmware/mcu_stm32_spl/` |
| 工具 | Arduino IDE | STM32CubeIDE | **Keil MDK5** |
| 库 | Arduino | STM32 HAL | **STM32 标准外设库** |
| 脉冲 | 阻塞延时 | 定时器中断 | 定时器中断 |
| 协议 | ✅ 一致 | ✅ 一致 | ✅ 一致 |

## 待办（二期）

- [ ] 加减速曲线（梯形/S 曲线）
- [ ] 多段运动队列
- [ ] 硬件看门狗
- [ ] 板载 USB CDC 替代 USB-TTL

## 排错

见 [`Doc/Keil搭建说明.md`](Doc/Keil搭建说明.md) 第七节。
最常犯的两个错：
1. Define 少写 `USE_STDPERIPH_DRIVER` 或 `STM32F10X_MD`
2. Include Paths 漏了 `Libraries\CMSIS`
