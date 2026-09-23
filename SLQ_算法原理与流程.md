# SLQ (Statistically-Lossless Quantization) 算法原理与复现流程

> 自动归档版 — 基于 SLQ 论文 (arXiv 2605.02404) 与本仓库 `SliderQuant` 的 GPTQ + Shapley 复现实现。
> 适用模型：Qwen3-8B；环境：NPU 910B3（容器 `slq-sgct-8b`）。

---

## 1. 总览

SLQ 是对大语言模型做**离线后训练量化（PTQ）**的方法，目标是在**不重新训练**的前提下，把每层权重分配到**不同位宽**，使总平均位宽最小，同时保证量化后模型"统计无损"。

与原始 GPTQ（全局统一位宽）的关键区别：

| 方面 | 原始 GPTQ | SLQ |
|---|---|---|
| 位宽 | 全局统一（如全 4-bit） | **逐层/逐组混合位宽**（3..8 bit） |
| 决策目标 | 最小化单层重建 MSE | **最小化分布/任务级失真**（EAR 或恢复率/KL） |
| 是否保证无损 | 只保证重建误差小 | 显式保证统计/任务无损（≥阈值） |
| 计算管线 | 逐层量化 | 敏感度归因 → 背包分配 → 量化重放 → 校验 |

> **重要澄清**：SLQ **没有修改 GPTQ 的损失函数**。对同一个指定位宽，SLQ 与 GPTQ 走完全相同的量化算法与重建损失；SLQ 新增的是"**每个层用几位**"这一外层选择问题的目标函数（EAR / 恢复率）。

---

## 2. 两个"无损"判据（SLQ 的核心创新）

SLQ 提出两套可选的"统计无损"判据，对应两条复现路线：

### 2.1 DL（Distribution-lossless）

用 **EAR（Expected Accuracy Recovery）** 度量 top-10 概率质量的重叠：

\[
\text{EAR}=\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{10}\min\big(p_i(k),\ q_i(k)\big)
\]

- \(p_i\)：BF16 参考模型在样本 \(i\) 输出的 top-10 分布（温度 \(T=0.6\)）
- \(q_i\)：量化模型同位置的分布
- 要求 \(\text{EAR}\ge 0.99\)（绝对）或相对 baseline 的 EAR 损失在阈值内

**局限（本复现实证）**：Qwen3-8B 即使全 8bit，EAR 天然上限约 0.974（RTN-8=0.97384、GPTQ-8=0.97381 几乎一致），0.99 在数学上不可达。故 8B 采用 TL 路线。

### 2.2 TL（Task-lossless）

用**下游任务恢复率**度量：

\[
\text{recovery}=\frac{\text{量化模型任务得分}}{\text{BF16 模型任务得分}}\ge 0.99
\]

观察：在 KL 较大区间外，**恢复率损失 ≈ 常熟 × KL**（Taylor 线性近似），因此任务级无损可退化为"预测 KL ≤ 阈值"这一可逐层累加的目标（详见第 5 节）。

---

## 3. 底层量化器：GPTQ（误差补偿）

实现：[`slq/gptq.py`](slq/gptq.py) `gptq_quantize_weight`

对单个权重 \(W\in\mathbb{R}^{out\times in}\) 与校准激活 \(X\in\mathbb{R}^{n\times in}\)（\(n=256\) 行）：

**Step 1：构造输出 Hessian 并求逆**
\[
H=2X^\top X+\lambda\cdot\overline{\text{diag}(H)}\cdot I,\qquad H_{inv}=H^{-1}
\]
damp 项（默认 0.01）保证正定。

**Step 2：逐 group（in 维每 128 列一组）固定 scale / zero_point**
非对称模式用每 out 行的 min/max：
\[
\text{scale}_r=\frac{\max-\min}{2^b-1},\qquad \text{zp}_r=\text{round}\Big(q_{\min}-\frac{\min}{\text{scale}_r}\Big)
\]

**Step 3：组内逐列贪心误差补偿（核心）**
对组内每列 \(j\)：
1. 保存原列 \(w_j\)
2. 量化-反量化 \(dq_j=\text{clamp}(\text{round}(w_j/\text{scale})+\text{zp},\ q_{\min},q_{\max})\cdot\text{scale}\)
3. 写回 \(W[:,j]=dq_j\)
4. 误差 \(err=(w_j-dq_j)/H_{inv}[j,j]\)
5. 补偿到组内后续未量化列：\(W[:,k]\gets W[:,k]-err\cdot H_{inv}[j,k],\ k>j\)

**NPU 适配**：逐列 scalar 索引（`Hinv[j,j]`）与小算子会触发海量 D2H 同步挂起，故 Hessian/inverse/逐列循环全部在 CPU 执行（12288 维 inverse ~10s），结果回传设备。

---

## 4. 敏感度归因：Multi-Bitwidth Shapley

实现：[`slq/gptq_shapley.py`](slq/gptq_shapley.py) `shapley_sensitivity`

**目的**：量化前知道"把组 \(m\) 从 8bit 降到 \(b\) bit 的质量损失边际"。Shapley 值是所有排列中该组边际贡献的平均，满足公平性与可加性：

\[
\phi_m(b)=\frac{1}{P}\sum_{p=1}^{P}\Big[\mathcal{D}(S^{<m}\cup\{m\})-\mathcal{D}(S^{<m})\Big]
\]

**计算过程**：
1. **预计算权重副本**：对每个目标位宽 \(b\in\{3..7\}\)，对全 72 组跑一次 GPTQ 得到 `Wb`。关键优化：**排列循环内不重跑 GPTQ，只做 `copy_`**（写回），把巨量重复量化压缩到 5×72 次。
2. **P 个随机排列**（P=2）：先生成组序 `perm`，写回全 Wmax(8bit) 测基线 \((EAR_b,KL_b)\)。
3. **逐组切换并记录边际**：按排列顺序把第 \(m\) 组换成 `Wb(b)`，前向测 \((EAR,KL)\)，记录：
   \[
   \Delta EAR_m=(EAR_{prev}-EAR),\qquad \Delta KL_m=(KL-KL_{prev})
   \]
   累加进 `sens[m][b]`。
4. **跨排列平均**：得到敏感度表 `c_ear[m][b]`、`c_kl[m][b]`（M×|B|）。

---

## 5. TL 单点标定 + 预测二分 + 背包分配

### 5.1 单点标定（把 KL 映射到恢复率）

实现：[`main_gptq_shapley.py`](main_gptq_shapley.py)

在 **uniform-6bit 锚点**处实测 \((\text{recovery}_{anchor}, KL_{anchor})\)：
\[
\alpha=\frac{1-\text{recovery}_{anchor}}{KL_{anchor}},\qquad D_{thresh}=\frac{1-0.99}{\alpha}
\]
同时标定预测/实测残差比：
\[
\rho=\frac{KL_{anchor}^{actual}}{\widehat{KL}(\text{uniform-}6)},\qquad
\widehat{KL}=\rho\Big(\text{base\_kl}+\sum_m c_{kl}[m][6]\Big)
\]

### 5.2 预测 KL（可加性）

位宽配置 \(\mathbf b=(b_1,..,b_M)\) 的预测 KL：
\[
\widehat{KL}(\mathbf b)=\rho\Big(\text{base\_kl}+\sum_{m=1}^{M} c_{kl}[m][b_m]\Big)
\]

### 5.3 位宽分配 = 多选背包（等价 ILP）

在总位宽预算 \(C\) 下最小化预测 KL：
\[
\min_{\mathbf b}\ \text{base\_kl}+\sum_m c_{kl}[m][b_m]\quad\text{s.t.}\ \sum_m \text{params}_m\, b_m\le C
\]
用 DP 求解（[`slq/allocation.py`](slq/allocation.py) `dp_allocate`，对 M=72、|B|=6 精确）。

### 5.4 外层二分（找最小平均位宽）

[`search_tl_predicted`](slq/gptq_shapley.py)：
- `lo=3, hi=8`；每次 `mid=(lo+hi)/2`，以 `budget=mid·Σparams` 跑背包
- 若 \(\rho\widehat{KL}\le D_{thresh}\) 可行 → `hi=mid`，否则 `lo=mid`
- 收敛（`hi-lo<0.02`）得到最小平均位宽配置

---

## 6. Guardrail（实测校验）与最终评测

实现：[`main_gptq_shapley.py`](main_gptq_shapley.py)

1. 恢复原权重 → 用 `alloc` 逐组 GPTQ 重放 → 实测 \(KL_{actual}\)
2. 计算 \(rho_{actual}=KL_{actual}/\widehat{KL}(alloc)\)，与锚点处 \(\rho\) 比值
3. 若 `ratio>2.0 或 <0.5` → 判定 guardrail 违反，需重新锚定
4. 通过后做 **lm_eval avg-6** 最终评测，验证恢复率 ≥0.99

---

## 7. 全流程步骤（对应代码调用链）

```
main_gptq_shapley.py
├─ load_model + build_groups          (pipeline.py)        加载 Qwen3-8B, 72 组
├─ load_calibration                   (main_slq.py)        wikitext2, 512 样本
├─ collect_reference                  (metrics.py)         BF16 参考 top-10 logits (T=0.6)
├─ snapshot_weights                   (main_slq.py)        存原始权重快照
├─ collect_activations                (gptq_shapley.py)    hook 采集每 Linear 激活 (256 行)
├─ BF16 基线 (eval_lm_eval, 缓存)     (main_slq.py)        avg-6=0.74073
├─ anchor: apply_config_gptq[6b]      (gptq_shapley.py)    uniform-6 锚点
├─   └─ model_metrics                 (metrics.py)         KL/recovery
├─ restore_weights                    (main_slq.py)        恢复原权重
├─ shapley_sensitivity                (gptq_shapley.py)    5 位宽×72组×P 排列
├─ 单点标定 α, ρ, D_thresh            (main_gptq_shapley.py)
├─ search_tl_predicted                (gptq_shapley.py)    二分 + 背包
├─ apply_config_gptq[alloc]           (gptq_shapley.py)    重放量化
├─ guardrail 校验                     (metrics.py)         KL 实测 vs 预测
└─ eval_lm_eval (最终 avg-6)          (main_slq.py)        recovery ≥ 0.99
```

---

## 8. 计算成本画像

| 环节 | 计算量 | 瓶颈 |
|---|---|---|
| BF16 参考 / 激活采集 | 多次前向 + hook | NPU |
| GPTQ 锚点(6b) + 5 位宽预计算(×72 组) | 6×72 层 CPU 逆矩阵+列循环 | **CPU**（约 50-100s/组）|
| Shapley 排列 | P×\|B\|×72 次前向 | NPU |
| 背包二分 | 30 次 DP | 内存 |
| 最终评测 | 1 次 lm_eval avg-6 | lm_eval |

**核心洞见**：SLQ 的自由度在"组×位宽"的选择问题（Shapley 归因 → 背包），而不在 GPTQ 本身。GPTQ 只保证单层量化精度，Shapley 通过预计算权重副本把重复量化降到最少，是能跑 8B 的关键。

---

## 9. 关键结论（Qwen3-8B 实证）

- **DL 不可达**：全 8bit EAR 上限 ~0.974（RTN/GPTQ 一致），top-10 质量 0.988<0.99，超出 DL 适用域。
- **TL 达标**：初步锚点 recovery≈0.998（GPTQ-6bit），高于 RTN 版锚点（0.9973）。
- **规模效应**：8B 位宽低于 0.6B，验证"大模型冗余更多"。