# Keil MDK5 标准库工程搭建说明

> ⚠️ **重要**：本目录提供**全部应用代码**（`User/`），但**不包含 ST 标准外设库**。
> 标准库（StdPeriph Lib）ST 已停止维护，需你自行准备（见下）。
> 我也提供了一个 `.uvprojx` 模板，但**库文件路径要按你的实际情况改**。

---

## 一、先准备好标准外设库

### 获取方式（任选其一）

1. **Keil 官方 Pack**（最方便）
   - Keil MDK5 → `Pack Installer` → 搜索 `STM32F1`
   - 安装 **Keil::STM32F1xx_DFP**，它会带 SPL 示例
   - 库通常在：`C:\Keil_v5\ARM\Packs\Keil\STM32F1xx_DFP\<版本>\Drivers\...`
     （新版 Pack 主要是 HAL，SPL 需另找）

2. **ST 官方旧版标准外设库**（推荐，资料最全）
   - 搜索 `STM32F10x_StdPeriph_Lib_V3.5.0`（2011 年最后版）
   - 解压后目录：
     ```
     STM32F10x_StdPeriph_Lib_V3.5.0/
     ├── Libraries/
     │   ├── CMSIS/CM3/CoreSupport/     ← core_cm3.c/.h
     │   ├── CMSIS/CM3/DeviceSupport/ST/STM32F10x/
     │   │                              ← stm32f10x.h, system_stm32f10x.c/.h,
     │   │                                 startup_stm32f10x_md.s
     │   └── STM32F10x_StdPeriph_Driver/
     │       ├── inc/   ← 所有 .h
     │       └── src/   ← 所有 .c
     └── Project/STM32F10x_StdPeriph_Template/
         └── stm32f10x_conf.h, stm32f10x_it.c/.h（模板）
     ```

3. 买个"STM32 标准库例程包"（淘宝/公众号常见，几块钱），里面通常直接有配好的 Keil 工程。

### 建议的目录组织

把库放到本工程旁的 `Libraries/`：

```
mcu_stm32_spl/
├── User/                    ← 本工程代码（已提供）
│   ├── main.c
│   ├── motor.c / motor.h
│   ├── cmd.c   / cmd.h
│   ├── config.h
│   ├── stm32f10x_it.c
│   └── stm32f10x_conf.h     ← 见下方"需你补充"
├── Libraries/
│   ├── CMSIS/
│   │   ├── core_cm3.c
│   │   ├── core_cm3.h
│   │   ├── stm32f10x.h
│   │   ├── system_stm32f10x.c
│   │   └── system_stm32f10x.h
│   └── STM32F10x_StdPeriph_Driver/
│       ├── inc/   （全部 .h）
│       └── src/   （全部 .c）
├── Start/
│   └── startup_stm32f10x_md.s   ← 中容量 (C8T6 = 64KB Flash)
└── Project/
    ├── TriniScan.uvprojx
    └── TriniScan.uvoptx
```

---

## 二、需你补充的两个文件

### 1. `stm32f10x_conf.h`
从库的模板 `Project/STM32F10x_StdPeriph_Template/stm32f10x_conf.h` 直接拷贝。
它负责**启用/禁用各外设驱动**（GPIO/RCC/USART/TIM/FLASH 等）。
**保持默认全开**即可（默认 `#include` 所有外设头）。

### 2. `system_stm32f10x.c`
从库的 CMSIS 目录拷贝。提供 `SystemInit()`，默认已配 72MHz 时钟。

> 若使用 `system_stm32f10x.c` 的默认 `SystemInit()`，
> 你的 `main.c` 里的 `SystemClock_Config()` 就是**二次确认**，无害。
> 也可删掉 `SystemClock_Config()`，直接用库的。

---

## 三、新建 Keil 工程（图形化，推荐）

### 1. 建工程
- Keil → `Project → New µVision Project`
- 存到 `Project/TriniScan.uvprojx`
- 选器件：**STM32F103C8**（若没有，装 `Keil::STM32F1xx_DFP`）

### 2. 弹出的 "Manage Run-Time Environment"
- **不勾任何东西**（我们用标准库，不用 RTE 的 HAL/CMSIS）
- 直接 OK

### 3. 建三个 Group 并加文件

右键 `Target 1` → `Manage Components` → `Groups` 建：

| Group | 加入的文件 |
|---|---|
| **User** | `main.c`、`motor.c`、`cmd.c`、`stm32f10x_it.c` |
| **Startup** | `startup_stm32f10x_md.s` |
| **CMSIS** | `core_cm3.c`、`system_stm32f10x.c` |
| **StdPeriph** | `Libraries/STM32F10x_StdPeriph_Driver/src/` 下**全部 .c**（约 30 个） |

> 想省事：StdPeriph 只加**用到的**：`stm32f10x_rcc.c`、`stm32f10x_gpio.c`、
> `stm32f10x_usart.c`、`stm32f10x_tim.c`、`stm32f10x_flash.c`、`misc.c`。
> 但全加最稳妥（Flash 64KB 够用）。

### 4. 头文件路径（`Options for Target → C/C++ → Include Paths`）

加这几条（按你的实际路径）：
```
..\User
..\Libraries\CMSIS
..\Libraries\STM32F10x_StdPeriph_Driver\inc
```

### 5. 预定义宏（`Options for Target → C/C++ → Define`）

```
USE_STDPERIPH_DRIVER, STM32F10X_MD
```

> ⚠️ 两个宏**必须**：
> - `USE_STDPERIPH_DRIVER`：让 `stm32f10x_conf.h` 生效
> - `STM32F10X_MD`：中容量（C8T6 = 64KB Flash / 20KB RAM）
>   - 若报 `stm32f10x.h` 里 `assert_param` 相关错，多半是这个宏错
>   - 大容量用 `STM32F10X_HD`，小容量 `STM32F10X_LD`

### 6. 输出 HEX（`Options for Target → Output`）
- 勾选 **Create HEX File**（方便用 FlyMcu 串口下载）

### 7. 调试器（`Options for Target → Debug`）
- 选 **ST-Link Debugger** → `Settings`
- **Port = SW**（不是 JTAG）
- Flash Download → 勾 `Reset and Run`（烧完自动跑）

### 8. 库选项（`Options for Target → Target`）
- 建议勾 **Use MicroLIB**（代码更小，且 `snprintf`/`atof` 够用）
- 本工程**不依赖浮点 printf**：`STATUS` 的角度已改成
  用整数×100 输出（`snprintf("%ld.%02ld")`），
  所以**不需要** `-u _printf_float`，MicroLIB 直接可用 ✅
- `ROT` 命令用 `atof` 解析（非 `strtof`），MicroLIB 兼容

### 9. 编译烧录
- `F7` 编译 → `F8` 下载
- 或 `Load` 按钮

---

## 四、用我提供的 `.uvprojx` 模板

本目录 `Project/TriniScan.uvprojx` 是**模板**，含：
- 正确的器件型号、宏定义、包含路径结构
- 三个 Group（User / Startup / CMSIS / StdPeriph）

**使用步骤**：
1. 把库文件按上面第一节的目录结构放好
2. 双击打开 `.uvprojx`
3. 若报"文件找不到" → `Manage Components` 里把**库里那几个文件重新添加一次**
4. `Options for Target` 里核对 Include Paths 和 Define
5. 编译

> 由于库文件在你的机器上位置未知，`.uvprojx` 里的路径**必须你核对一遍**。

---

## 五、验证

```
PING        → PONG
ENABLE 0    → OK      脱机（可手转电机）
ENABLE 1    → OK      使能（电机锁死）
STEPS 3200  → OK      转 36°（32000 步/圈）
HOME        → OK      回零
ROT 2       → OK      转 2°（扫描步进）
STATUS      → STATUS IDLE <步数> <角度> 0
```

**⚠️ 第一次测电机务必脱开负载**，确认转向/步数对了再装转盘。

---

## 六、与 HAL 版的差异

| 项目 | HAL 版 | 标准库版 |
|---|---|---|
| 依赖 | STM32Cube HAL | STM32F10x StdPeriph Lib |
| 生成方式 | CubeMX 生成 | 手写（本目录已提供） |
| 工程文件 | CubeIDE `.ioc`+自动 | Keil `.uvprojx` |
| 代码量 | 多（HAL 抽象层厚） | **少（直接寄存器封装，更直观）** |
| 可读性 | 一般 | **寄存器风格，初学者更易理解** |
| 维护性 | ST 主推 | ST 已停更（但足够用） |

**功能、协议、引脚完全一致**，两者可互换。

---

## 七、常见问题

| 现象 | 排查 |
|---|---|
| `stm32f10x.h` 找不到 | Include Paths 漏了 `Libraries\CMSIS` |
| `assert_param` 未定义 | Define 缺 `USE_STDPERIPH_DRIVER` |
| `STM32F10X_MD` 报错 | 检查是否写成 `STM32F10X_MD`（不是 `STM32F103_MD`） |
| `TIM2_IRQHandler` 重复定义 | 库模板的 `stm32f10x_it.c` 里也有 → 删掉一个 |
| 编译过但 LED 不亮 | 时钟没起振 / BOOT0=1 / 没接 LED 引脚 |
| `strtof` 未定义 | 标准库下可用；若报错改成 `atof` |
| 电机不转 | 查 STEP/DIR/EN 接线、TB6600 细分、极性宏 |
| 电机方向反 | 改 `DIR_POSITIVE_LEVEL` 或交换 DIR 线 |
| 电机嗡嗡不转 | 速度太快 → 加大 `DEFAULT_STEP_US` |
| 串口乱码 | 波特率错 / 时钟非 72MHz |
