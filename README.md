# PiperKachaka — Mobile Manipulator 全身 IK 驗證

在 MuJoCo 中驗證 9-DOF 行動操作機器人（Kachaka 底盤 + Piper 6-DOF 手臂）的全身 IK，含模擬端的 teleop 工具與 ROS2 bridge。

> IK 設計、實驗數據、操作方式請看 [`mm_ik_validation/README.md`](mm_ik_validation/README.md)。本檔案只負責整體導覽。

## 為什麼做這個

商用 Kachaka 移動底盤上方掛 Agilex Piper 6-DOF 手臂後，工作空間從 1.5 m 半徑的手臂 reach 擴展成「整個房間」。問題是 **9 個 DOF 怎麼協調**：手臂能搞定的就不該動底盤（底盤抖動會傳到末端），手臂不夠才讓底盤上前。

這個 repo 用 [mink](https://github.com/kevinzakka/mink)（QP-based differential IK）配合 base damping anneal 策略，讓 IK 自動取「最少底盤移動」的解，並提供互動式 teleop 與 self-test 量化行為。

## 目錄結構

```
.
├── combined_scene.xml              原始合成場景（freejoint + weld constraint）
├── kachaka_description/            Kachaka 底盤 MJCF + STL
├── piper_description/              Piper 手臂 MJCF + OBJ
└── mm_ik_validation/               主要工作目錄
    ├── mobile_manipulator.xml      重寫的 MJCF（planar 3-DOF base + Piper 單一 kinematic chain）
    ├── whole_body_ik.py            mink-based IK（CLI / interactive 兩種模式）
    ├── whole_body_executor.py      Two-stage executor: IK → drive base → move arm
    ├── ros2_bridge.py              ROS2 bridge（接到實機 Kachaka + Piper）
    ├── mujoco_teleop_sim.py        Sim 端 teleop sender
    ├── teleop_ghost_viewer.py      Digital-twin viewer（讀 shared memory，不算 IK）
    ├── teleop_diagnostic.py        Teleop 診斷工具
    ├── pose_publisher_gui.py       Pose target GUI
    ├── self_test.py                Headless 自動測試
    └── README.md                   IK 詳細設計 + 實驗結果
```

## 環境設定

需求：Python 3.10+ on Linux。建議使用 venv：

```bash
python3 -m venv .venv_ik
source .venv_ik/bin/activate
pip install mujoco==3.8.1 mink==1.1.1 scipy numpy
```

ROS2 bridge 額外需要 `rclpy`（依你 ROS2 distro 安裝；本專案在 Humble 上測試）。

## Quick start

啟用 venv 後：

### 互動式 IK demo（推薦先跑這個）

```bash
cd mm_ik_validation
DISPLAY=:0 python whole_body_ik.py interactive
```

雙擊粉紅球 → Ctrl+右鍵拖曳 → IK 即時跟上。詳細操作見 [mm_ik_validation/README.md](mm_ik_validation/README.md)。

### Headless self-test（量化驗證）

```bash
python mm_ik_validation/self_test.py
```

跑 FK round-trip + random workspace pose + arm-only sufficiency 三項測試。

## 設計重點（一句話版本）

3 個 mink task：`FrameTask`（末端追蹤）+ `PostureTask`（手臂 regularization）+ `DampingTask`（base damping 100×）。QP solver 看到「動底盤的 cost 是動手臂的 100 倍」自然優先轉手臂；近距 0 cm base 移動、遠距才讓 base 上前。完整數據見 [`mm_ik_validation/README.md`](mm_ik_validation/README.md)。

## 參考

- [mink](https://github.com/kevinzakka/mink) — Kevin Zakka 的 QP-based differential IK for MuJoCo
- [priyasundaresan/homer](https://github.com/priyasundaresan/homer) — 用 mink 的 mobile manipulator imitation learning（task 設計參考）
- [Preferred Robotics Kachaka](https://kachaka.life/) — 移動底盤
- [Agilex Piper](https://www.agilex.ai/products/piper) — 6-DOF 手臂
