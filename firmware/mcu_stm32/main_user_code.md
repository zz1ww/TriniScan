# STM32 main.c 集成指南（照抄即可）

> 本文件给出 **CubeMX 生成的 `Core/Src/main.c`** 中，
> 每个 `USER CODE BEGIN/END` 区域应填的内容。
> **只改这些区域，其余代码一律不要动。**

---

## 一、CubeMX 需要配置的外设

| 外设 | 配置要点 |
|---|---|
| **RCC** | HSE = Crystal/Ceramic；时钟树 SYSCLK = **72 MHz** |
| **SYS** | Debug = **Serial Wire**（否则下次烧不进） |
| **USART1** | 115200 / 8N1 / 无校验；**开全局中断**；PA9=TX, PA10=RX |
| **TIM2** | Internal Clock；**Prescaler = 71**（1 MHz 计数）；ARR 初值随便；**开更新中断** |
| **GPIO** | 见下表 |

### GPIO 引脚分配（在 CubeMX 里设 Label）

| 引脚 | 模式 | Label |
|---|---|---|
| PA0 | GPIO_Output | `STEP` |
| PA1 | GPIO_Output | `DIR` |
| PA2 | GPIO_Output | `EN` |
| PB0 | GPIO_Input + Pull-up | `LIMIT` |
| PC13 | GPIO_Output | `LED` |

> **Label 很重要**：CubeMX 会据此在 `main.h` 生成
> `STEP_Pin`、`STEP_GPIO_Port` 等宏，模块代码依赖它们。

### NVIC
- **USART1 global interrupt**：Enable
- **TIM2 global interrupt**：Enable（优先级默认即可）

---

## 二、把两个模块加进工程

1. 把 `Core/Inc/config.h`、`motor.h`、`cmd.h` 复制到工程的 `Core/Inc/`
2. 把 `Core/Src/motor.c`、`cmd.c` 复制到工程的 `Core/Src/`
3. CubeIDE 里右键工程 → **Refresh**（F5），让新文件出现在 `Core/Src` 下
4. 编译一次确认无错

---

## 三、逐段填写 USER CODE

### ① 顶部包含（`USER CODE BEGIN Includes`）

```c
/* USER CODE BEGIN Includes */
#include <string.h>
#include "motor.h"
#include "cmd.h"
#include "config.h"
/* USER CODE END Includes */
```

### ② 私有变量（`USER CODE BEGIN PV`）

```c
/* USER CODE BEGIN PV */
/* 命令用的 UART 句柄别名（模块通过 extern hcmd_uart 访问） */
UART_HandleTypeDef hcmd_uart;

/* 步进定时器句柄别名 */
TIM_HandleTypeDef htim_step;

/* UART 单字节接收缓冲 */
static uint8_t g_rx_byte;
/* USER CODE END PV */
```

### ③ 私有函数声明（`USER CODE BEGIN PFP`）

```c
/* USER CODE BEGIN PFP */
static void cmd_uart_start_rx(void);
/* USER CODE END PFP */
```

### ④ main() 初始化后（`USER CODE BEGIN 2`）

```c
/* USER CODE BEGIN 2 */
/* 把 CubeMX 生成的句柄别名给模块用 */
hcmd_uart = huart1;
htim_step = htim2;

/* 初始化模块 */
motor_init();
motor_enable(true);
cmd_init();

/* 开启 UART 中断接收 */
cmd_uart_start_rx();

/* 上电点亮 LED 提示就绪 */
HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_SET);

/* 等待 1 秒让电机驱动/上位机就绪，然后发 READY */
HAL_Delay(1000);
cmd_send_line("READY");
/* USER CODE END 2 */
```

### ⑤ 主循环（`USER CODE BEGIN 3`）

```c
/* USER CODE BEGIN 3 */
/* 处理已收到的命令（非阻塞） */
cmd_poll();
/* USER CODE END 3 */
```

### ⑥ main() 之后（`USER CODE BEGIN 4`）

```c
/* USER CODE BEGIN 4 */

/* 启动一次 UART 中断接收（每次收完要重新开） */
static void cmd_uart_start_rx(void)
{
    HAL_UART_Receive_IT(&hcmd_uart, &g_rx_byte, 1);
}

/* ---- UART 接收完成回调：把字节喂给命令模块，再开接收 ---- */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1) {
        cmd_feed_byte(g_rx_byte);
        cmd_uart_start_rx();
    }
}

/* ---- 定时器更新中断：驱动步进脉冲 ---- */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM2) {
        motor_on_timer_tick();
    }
}

/* ---- 串口错误回调：重启接收，避免卡死 ---- */
void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1) {
        __HAL_UART_CLEAR_OREFLAG(huart);
        cmd_uart_start_rx();
    }
}
/* USER CODE END 4 */
```

---

## 三·补、必须开启浮点 printf ⚠️

`cmd.c` 的 `STATUS` 用 `snprintf("%.2f")` 输出浮点角度。
**CubeIDE 默认关闭浮点 printf**，不改会导致角度输出为空或编译告警。

**设置方法**：
`Project → Properties → C/C++ Build → Settings →`
`MCU GCC Linker → Libraries` →
勾选 **Use float with printf from newlib-nano (`-u _printf_float`)**

> 也可在 Linker flags 手动加 `-u _printf_float`。
> 若不改，`STATUS` 仍可用（步数正确），只是浮点角度显示不出来。

**另外**：若用 `newlib-nano` 且代码里有 `strtof`，`-u _scanf_float`
可不加（`strtof` 属于标准库，通常能正常链接）。

---

## 四、验证步骤

### 1. 先测串口
- 用 USB-TTL 或板载 USB CDC 连电脑
- 串口助手 115200 8N1
- 发 `PING` → 应回 `PONG`

### 2. 再测电机（**先脱开负载**！）
```
ENABLE 0        → OK      （脱机，可手动转）
ENABLE 1        → OK      （使能，电机锁死）
STEPS 3200      → OK      （走 3200 步 = 36°）
```
- 若电机**不转**：查 STEP/DIR 接线、ENABLE 极性、TB6600 细分设置
- 若**转向反**：交换 DIR 两根线，或改 `config.h` 的 `DIR_POSITIVE_LEVEL`

### 3. 测回零
```
HOME            → OK
```

### 4. 测角度
```
ROT 2           → OK      （转 2°，正是每次扫描的步进）
```

---

## 五、常见问题

| 现象 | 排查 |
|---|---|
| 编译报 `STEP_GPIO_Port` 未定义 | CubeMX 里引脚 Label 没设，重新设成 `STEP` 等并生成 |
| `htim_step` 未定义 | 确认 CubeMX 里定时器命名为 **TIM2** |
| 电机嗡嗡不转 | ARR 太小（速度太快）→ 加大 `DEFAULT_STEP_US` |
| 电机丢步 | 速度太快/电流不足 → 调 TB6600 电流、加大 STEP_US |
| 串口乱码 | 波特率不对，或时钟没配对（确认 72MHz） |
| 收到 OK 但电机不动 | ENABLE 极性反了 → 改 `ENABLE_ACTIVE_LEVEL` |
| 回零不停 | 限位开关接线/极性 → 改 `LIMIT_TRIGGERED_LEVEL` |

---

## 六、参数速查（`config.h`）

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
