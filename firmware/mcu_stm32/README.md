# TriniScan 转台控制器固件（STM32F103C8T6 + HAL）— 参考版

> ⚠️ **本目录是"从零集成参考模板"（TIM2 基础定时器 + 两拍脉冲方案）。**
> **实际使用请看 [`../mcu_stm32_cubeide/`](../mcu_stm32_cubeide/)** ——
> 那是**已编译烧录验证**的完整工程（TIM1 PWM + 中断计数方案）。

基于 **STM32CubeIDE + HAL** 的转台控制固件，功能与 `firmware/mcu/`（Arduino 版）
**完全等价**，串口协议一致，可直接替换。

## 目录

```
mcu_stm32/
├── README.md              本文件
├── main_user_code.md      ★ main.c 各 USER CODE 区填写指南
└── Core/
    ├── Inc/
    │   ├── config.h       编译期配置（电机/速度/极性）
    │   ├── motor.h        步进电机模块接口
    │   └── cmd.h          串口命令模块接口
    └── Src/
        ├── motor.c        步进电机模块实现（定时器中断脉冲）
        └── cmd.c          命令解析实现（环形缓冲 + 行解析）
```

> **注意**：本目录只提供**你该写的模块代码**。
> CubeMX 自动生成的 `main.c`、`stm32f1xx_it.c`、HAL 库等
> **不在此提供**，由你在 CubeIDE 里生成后按
> [`main_user_code.md`](main_user_code.md) 填写 USER CODE 区。

## 快速开始

1. **建工程**：CubeIDE → New STM32 Project → STM32F103C8Tx
2. **配外设**：按 `main_user_code.md` 第一节配置
   RCC(72MHz) / SYS(SWD) / USART1(115200) / TIM2(1MHz) / GPIO
3. **加模块**：把 `Core/Inc/*.h`、`Core/Src/*.c` 复制进工程，Refresh
4. **填 USER CODE**：按 `main_user_code.md` 第三节逐段填写
5. **编译烧录**：ST-Link SWD
6. **验证**：`PING`→`PONG`，`STEPS 3200`→电机转 36°

## 串口协议（与 Arduino 版一致）

| 命令 | 响应 | 说明 |
|---|---|---|
| `PING` | `PONG` | 连通测试 |
| `HOME` | `OK` | 回零（触发限位后原点清零） |
| `ROT <deg>` | `OK` | 相对转动角度（度，可负） |
| `STEPS <n>` | `OK` | 相对转动步数（可负） |
| `ENABLE 0/1` | `OK` | 脱机 / 使能 |
| `SPEED <us>` | `OK` | 设速度（微秒/步，200~5000） |
| `STATUS` | `STATUS <state> <steps> <deg> <busy>` | 查询状态 |
| `STOP` | `OK` | 急停 |

- 波特率 **115200**，ASCII，`\n` 结尾
- 上电发 `READY`
- 上位机收到 `OK` 后应等待 `SETTLE_MS`（默认 200ms）再触发相机

## 与 Arduino 版的差异

| 项目 | Arduino | STM32 |
|---|---|---|
| 脉冲方式 | `delayMicroseconds` 阻塞 | **定时器中断，非阻塞** |
| 主循环 | 阻塞式运动 | 运动中仍响应命令 |
| 命令处理 | 字符串拼接 | **环形缓冲 + 行解析**（更稳） |
| 速度切换 | 改全局变量 | 动态改 ARR，运动中可改 |
| 状态上报 | 只有状态码 | 含位置步数/角度/忙标志 |

**升级点**：STM32 版运动**非阻塞**，转台转动时上位机仍可查询状态、
急停，更适合"转-停-拍"精确同步。

## 关键设计

```
     ┌──────────────┐
     │ USART1 中断   │──┐  只存字节
     └──────────────┘  │
                       ▼
                 环形缓冲 (256B)
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

- **中断只做最少的事**（存字节 / 翻引脚）
- **解析在主循环**，避免中断里跑 `strtof`
- **运动由硬件定时器驱动**，精度不受主循环影响

## 待办（二期）

- [ ] 加减速曲线（梯形/S 曲线），减少启停丢步
- [ ] 多段运动队列（一次下发多个视角）
- [ ] 硬件看门狗
- [ ] 板载 USB CDC 替代 USB-TTL（省一个模块）

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

## 接线（务必共地）

```
STM32           TB6600          NEMA17
─────           ──────          ──────
PA0  ────────  PUL-
PA1  ────────  DIR-
PA2  ────────  ENA-
GND  ────────  GND ─────────── 电源地
               A+ A- B+ B- ───  电机线圈
               VCC ────────── 12~24V 电源+
```
