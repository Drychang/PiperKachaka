# 全身 IK 驗證 (Kachaka + Piper)

在 MuJoCo 模擬器中驗證 9-DOF 行動操作機器人（Kachaka 底盤 + Piper 手臂）的全身 IK 正確性。

**設計思路**：使用 [mink](https://github.com/kevinzakka/mink)（QP-based 微分 IK）+ base damping 100x，讓 IK **優先動手臂，底盤只在手臂不可達時才動**。參考 [priyasundaresan/homer](https://github.com/priyasundaresan/homer) 的 mink mobile manipulator 設計。

## 檔案結構

```
mm_ik_validation/
├── mobile_manipulator.xml   # 重寫的 MJCF：planar 3-DOF base + Piper 接成單一 kinematic chain
├── whole_body_ik.py         # mink-based IK + CLI/interactive 兩種使用模式
├── self_test.py             # Headless 自動測試
└── README.md
```

原始檔（`combined_scene.xml`、`kachaka_description/`、`piper_description/`）**未動**。

## 與原模型相比改了什麼？

| 項目 | 原模型 (`combined_scene.xml`) | 新模型 (`mobile_manipulator.xml`) |
|---|---|---|
| Base 自由度 | `<freejoint/>`（6 DOF）+ 兩個輪子速度控制 | `base_x`(slide) → `base_y`(slide) → `base_yaw`(hinge) |
| Piper 掛載 | 自己的 `<freejoint/>` + `<weld>` constraint | 直接以子 body 形式接在 `kachaka_base_link` 下 |
| End-effector | 無 | 在 `link6` 新增 `<site name="ee">` |
| 互動 target | 半透明小白框（看不太到） | 粉紅球 + 粗 RGB 軸 |

## 環境

Server 上已建好 venv，已安裝 `mujoco 3.8.1` + `mink 1.1.1` + `scipy` + `numpy`：

```bash
PY=$(which python)
```

## IK 設計（重點）

3 個 mink task：

```python
ee_task        = FrameTask("ee", position_cost=1.0, orientation_cost=1.0)
posture_task   = PostureTask(cost_arm=1e-3)              # 對手臂做輕度 regularization
damping_task   = DampingTask(cost_base=100)              # 對 base x/y/yaw 加 100 倍 cost
```

`damping_task` 是關鍵：base 移動的 cost 比手臂高 **100×**，QP solver 自然會優先轉手臂。

### Damping anneal（自動退讓策略）

`solve_with_restart()` 按以下順序嘗試：

| Stage | base damping | 行為 |
|---|---|---|
| 1 | 100x（完整） | 純手臂解 — 近距優選 |
| 2 | 10x | 允許小幅 base 移動 |
| 3 | 1x | base 自由 — 處理較遠目標 |
| 4 | 0x + 手臂擾動 | 最後手段（避開 local minimum） |

**只回傳第一個成功的 stage** → 自動取「最少 base 移動」的解。

實測對 3 個距離：

| 目標 | 收斂 stage | base 移動 | 手臂變化 |
|---|---|---|---|
| (0.3, 0.1, 0.8) 近 | stage 1 | 0 cm | 6 個關節大幅旋轉 |
| (0.7, 0.3, 0.7) 中 | stage 3 | 35 cm | 手臂展開 + base 微調 |
| (2.0, 0, 0.6) 遠 | stage 3 | 129 cm | base 走過去 + 手臂展開 |

## 使用方式

```bash
cd mm_ik_validation
PY=$(which python)
```

### 模式 1：給定 6D pose 一次解出來

```bash
$PY whole_body_ik.py solve --pose "0.5 0.3 0.8 0 1.5708 0"
```

格式：`"x y z roll pitch yaw"`（m / rad）。輸出包含：
- 走到第幾個 stage 才收斂
- 末端位置 / 姿態誤差
- 9 個 IK 關節值
- Base 實際移動量（直接告訴你 base 動了多少）

**測試 arm-prefer 行為**：
```bash
# 近距：base 應該幾乎不動
$PY whole_body_ik.py solve --pose "0.3 0.1 0.8 0 1.5708 0"

# 同 pose 但關掉 damping 對比
$PY whole_body_ik.py solve --pose "0.3 0.1 0.8 0 1.5708 0" --no-base-damping
```

### 模式 2：互動式拖曳目標（最直覺）

```bash
DISPLAY=:0 $PY whole_body_ik.py interactive
```

操作（MuJoCo viewer）：
- **雙擊粉紅球**選中它
- **Ctrl + 右鍵拖曳** → 平移
- **Ctrl + 中鍵拖曳** → 旋轉
- **ENTER** → 切換 fix_base（base 凍結 vs 自由）
- **SPACE** → 暫停
- **R** → reset 回 home

Terminal 每 0.5 秒印追蹤狀態：
```
[FIX_BASE] pos_err=  0.42mm rot_err=  0.18deg  base=(+0.00,+0.00,+0.0deg)
```

### 模式 3：自動測試

```bash
$PY self_test.py
```

實測結果（mink + damping anneal）：

| 測試 | 成功率 | 位置誤差中位數 |
|---|---|---|
| FK round-trip（保證可達） | **100%** (50/50) | 0.59 mm |
| Random workspace pose | 72% (36/50) | 0.89 mm |
| **Arm-only sufficient** | 3% (1/30) | — |
| **Needed base motion** | 43% (13/30) | — |
| Unreachable | 53% (16/30) | — |

Test 3 直接量化 base 是「補強」角色：只有 3% 的 workspace pose 純手臂能搞定，43% 額外需要 base。

## 接下來銜接實機

當你要把這套東西搬到實體 Kachaka + Piper：

1. **手臂端**：把 IK 解出來的 `joint1`~`joint6` 用 **joint-level command** 送到 Piper，**不要走 `/pos_cmd`**（firmware 會重算 IK 蓋掉你的協調）。
2. **底盤端**：把 `[base_x, base_y, base_yaw]` 轉成 Kachaka 的 `/navigate_to_pose` 或自己寫 differential-drive controller。
3. **時序**：實機建議「底盤先到位、停下、手臂再動」，避免 base 抖動傳到 EE。我們的 damping anneal 已經讓 base 移動「成本敏感」——大部分 case 它只動手臂，這對實機很友善。

## 調整參數

如果你覺得 base 還是動太多/動太少，調整 `--base-damping` flag：

```bash
$PY whole_body_ik.py --base-damping 500 solve --pose "..."   # 更傾向手臂
$PY whole_body_ik.py --base-damping 10  solve --pose "..."   # 更傾向 base
```

預設 100，可根據實際使用調整。

## 技術細節

- **IK library**: [mink 1.1.1](https://github.com/kevinzakka/mink) — Kevin Zakka 的 QP-based differential IK for MuJoCo
- **QP solver**: daqp
- **Tasks**: `FrameTask`（末端追蹤）+ `PostureTask`（手臂 regularization）+ `DampingTask`（base damping）
- **Constraints**: `ConfigurationLimit`（關節極限）
- **EE site 位置**: `link6` 座標系 `(0, 0, 0.13503)`，對應 link7/link8 掛載點。要追蹤夾爪尖端的話把 z 改大一點（例如 0.18）。

## 參考

- Mink mobile_tidybot example: https://github.com/kevinzakka/mink/blob/main/examples/mobile_tidybot.py
- HoMeR（用 mink 的 mobile manipulator imitation learning）: https://github.com/priyasundaresan/homer
