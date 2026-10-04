# 单片机固件（转台控制器）

上位机通过串口发送指令，MCU 控制 TB6600 驱动步进电机，实现精确转角与
"转-停-拍"时序。

## 硬件

- MCU：Arduino / STM32 / ESP32（示例基于 Arduino 风格）
- 驱动器：TB6600
- 电机：NEMA17 17HS4401
- 限位：归零限位开关（可选）

## 接线

| MCU 引脚 | 连接到 | 说明 |
|---|---|---|
| D2 (STEP) | TB6600 PUL- | 脉冲（共阳时负端接 MCU）|
| D3 (DIR)  | TB6600 DIR- | 方向 |
| D4 (EN)   | TB6600 ENA- | 使能 |
| D5        | 限位开关 | 归零（INPUT_PULLUP）|
| 5V        | TB6600 PUL+/DIR+/ENA+ | 共阳 |
| GND       | TB6600 信号 GND | **必须共地** |

> 电机电源（12–24V）由独立电源/电池供给 TB6600 的 VCC/GND。

## 串口协议

波特率 115200，ASCII 行协议，`\n` 结尾。

| 指令 | 作用 | 返回 |
|---|---|---|
| `PING` | 握手 | `PONG` |
| `HOME` | 归零 | `OK` |
| `ROT <deg>` | 相对转动角度（度，浮点）| `OK` |
| `STEPS <n>` | 相对转动步数（整数）| `OK` |
| `ENABLE 1/0` | 使能/脱机 | `OK` |
| `STATUS` | 查询状态 | `STATUS <state>` |
| `STOP` | 停止 | `OK` |

## 运动学

```
每圈整步数     = 200            (1.8° 步距角)
微步           = 32             (TB6600 细分)
减速比         = 5              (机械减速)
每圈步数       = 200 × 32 × 5 = 32000
每度步数       = 32000 / 360 ≈ 88.89
```

## 编译烧录

### Arduino
```bash
# 使用 arduino-cli
arduino-cli compile --fqbn arduino:avr:uno .
arduino-cli upload  --fqbn arduino:avr:uno -p COM3 .
```

或将 `turntable_controller.ino` 用 Arduino IDE 打开烧录。

### STM32 / ESP32
参考 `config.h` 修改引脚与定时器配置，用对应平台工具链编译。

## 参数配置

见 `config.h`：引脚、细分、减速比、速度等。

## 待办

- [ ] 串口指令解析与错误处理完善
- [ ] 加减速曲线（防止丢步/冲击）
- [ ] 限位归零流程
- [ ] 编码器闭环（可选）
- [ ] 硬件触发相机（后期优化）
