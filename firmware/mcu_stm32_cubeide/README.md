# TriniScan 转台控制器 — STM32CubeIDE 完整工程

**已实测编译通过并烧录运行**的 STM32F103C8T6 工程（HAL 库）。
打开即可用，串口协议与标准库版（`firmware/mcu_stm32_spl/`）完全一致。

## 硬件配置（本工程实际使用）

| 功能 | 引脚 | 说明 |
|---|---|---|
| STEP | **PA8** | TIM1_CH1 PWM 输出 → TB6600 PUL- |
| DIR | **PB0** | GPIO 输出 → TB6600 DIR- |
| EN | **PB1** | GPIO 输出 → TB6600 ENA- |
| LIMIT | **PB2** | GPIO 上拉输入，回零限位 |
| LED | PC13 | 板载 LED |
| USART1 TX | PA9 | → 上位机 RX |
| USART1 RX | PA10 | ← 上位机 TX |
| SWDIO / SWCLK | PA13 / PA14 | ST-Link |

**系统时钟**：HSE 8MHz × PLL9 = **72 MHz**

## 关键设计：TIM1 PWM + 中断计数

```
PA8 ──> TIM1_CH1 PWM ──> TB6600 ──> NEMA17
              │
              └─ 每次更新中断 = 一个 STEP 脉冲
                 中断里剩余步数 -1，减到 0 立即停 PWM
```

- **脉冲由硬件产生**，频率 = `72MHz / (PSC+1) / (ARR+1)`
  - PSC = 71 → 1 MHz 计数
  - ARR = 每步间隔(µs) - 1，如 899 → 900µs/步
- **步数由中断精确计数**，不受主循环影响
- 运动**非阻塞**：转台转动时上位机仍可查询/急停

### ⚠️ 必备配置

CubeMX 里 **TIM1 必须开 update interrupt**：
`Timers → TIM1 → NVIC Settings → 勾选 "TIM1 update interrupt"`

否则：发 `STEPS n` 后**电机一直转不会停**（中断不触发，步数不减）。

## 目录说明

```
mcu_stm32_cubeide/
├── TriniScan.ioc            CubeMX 配置（可重新生成代码）
├── .cproject / .project     CubeIDE 工程文件
├── STM32F103C8TX_FLASH.ld   链接脚本
└── Core/
    ├── Inc/
    │   ├── config.h         ★ 引脚/电机/速度/极性配置
    │   ├── motor.h          ★ 步进电机模块
    │   ├── cmd.h            ★ 串口命令模块
    │   ├── main.h           (CubeMX 生成)
    │   ├── stm32f1xx_it.h   (CubeMX 生成)
    │   └── stm32f1xx_hal_conf.h (CubeMX 生成)
    └── Src/
        ├── motor.c          ★ TIM1 PWM + 中断计数
        ├── cmd.c            ★ 环形缓冲 + 行解析
        ├── main.c           ★ USER CODE 区已填好
        └── (其余为 CubeMX 生成)
```

> ★ = 本项目代码，其余为 CubeMX 自动生成。

## ⚠️ 缺少 Drivers 目录

本仓库**不含 `Drivers/`**（ST HAL 库，约 4 MB）。
导入 CubeIDE 后会缺失，二选一解决：

**方法 A（推荐）**：用 `.ioc` 重新生成
1. CubeIDE → `File → Open Projects from File System` 导入本工程
2. 双击 `TriniScan.ioc` 打开 CubeMX
3. `Project → Generate Code`（会自动补全 Drivers）

**方法 B**：从其他工程拷贝 `Drivers/` 目录过来。

## 导入步骤

1. CubeIDE → `File → Import → General → Existing Projects into Workspace`
2. 选择 `mcu_stm32_cubeide` 目录
3. 若缺 Drivers → 按上面方法 A 重新生成
4. `Ctrl+B` 编译 → ST-Link 烧录

## 串口协议

| 命令 | 响应 | 说明 |
|---|---|---|
| `PING` | `PONG` | 连通测试 |
| `HOME` | `OK` | 回零 |
| `ROT <deg>` | `OK` | 相对转动角度 |
| `STEPS <n>` | `OK` | 相对转动步数 |
| `ENABLE 0/1` | `OK` | 脱机 / 使能 |
| `SPEED <us>` | `OK` | 设速度（150~5000 µs/步） |
| `STATUS` | `STATUS <state> <steps> <deg> <busy>` | 查询状态 |
| `STOP` | `OK` | 急停 |

- **115200 8N1**，ASCII，`\n` 结尾
- 上电发 `READY`
- 上位机收到 `OK` 后等待 `SETTLE_MS`（默认 200ms）再触发相机

## 验证记录

- [x] 编译通过（arm-none-eabi-gcc，0 error）
- [x] 烧录成功（ST-Link SWD）
- [x] 串口输出 `READY` 正常
- [ ] `PING` → `PONG`（待测）
- [ ] `STEPS 3200` → 转 36°（待测，等供电）

## 参数速查（`Core/Inc/config.h`）

| 宏 | 默认 | 含义 |
|---|---|---|
| `MICROSTEP` | 32 | TB6600 细分，须与拨码一致 |
| `GEAR_RATIO` | 5 | 机械减速比 |
| `STEPS_PER_REV` | 32000 | 每圈总步数 |
| `DEFAULT_STEP_US` | 900 | 每步间隔（越大越慢） |
| `SETTLE_MS` | 200 | 到位稳定延时 |
| `ENABLE_ACTIVE_LEVEL` | RESET | 使能极性 |
| `DIR_POSITIVE_LEVEL` | SET | 方向极性 |
| `LIMIT_TRIGGERED_LEVEL` | RESET | 限位触发极性 |
