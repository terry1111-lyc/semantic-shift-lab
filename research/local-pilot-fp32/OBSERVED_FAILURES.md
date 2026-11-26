# 真实失败案例 / Observed failures

以下输出来自实际模型。案例是结果产生后的诊断选择，不是预先注册的独立检验；选择规则和完整分母同时公开。模板仅经助手检查，未做独立人工审核。

## 改写失败与编辑后沿用答案

“原题对→改写错”以原题正确的题组为分母；“编辑沿用”指原题正确、事实编辑后输出仍等于原题答案。后者只是行为描述，不能直接解释模型内部机制。

| 运行 | 原题正确 / 60 | 其中改写答错 | 其中编辑沿用旧答案 |
|---|---:|---:|---:|
| A | 0 / 60 | 0 / 0 | 0 / 0 |
| B-seed17 | 16 / 60 | 13 / 16 | 0 / 16 |
| C-seed17 | 12 / 60 | 8 / 12 | 0 / 12 |
| E-seed17 | 17 / 60 | 11 / 17 | 3 / 17 |
| B-seed29 | 16 / 60 | 13 / 16 | 0 / 16 |
| C-seed29 | 10 / 60 | 6 / 10 | 0 / 10 |
| E-seed29 | 19 / 60 | 14 / 19 | 1 / 19 |

零分母表示该条件无法估计比例，不是 0% 的条件错误率。

## 案例：B-seed17 / `s20261125-transfer_2-0000`

选择规则：`first original-correct / paraphrase-wrong`，按冻结数据顺序。

**original**（正确答案：57）

Lin has 43 cards, receives 2 cards, then receives 12 cards. How many cards does Lin have?

实际原始输出：

```text
57
```

**paraphrase**（正确答案：57）

Lin adds two gifts of 2 and 12 cards to an initial collection of 43. How many cards are there now?

实际原始输出：

```text
49
```

**reorder**（正确答案：57）

A second gift contains 12 cards and the first contains 2. Lin had 43 cards before receiving both. How many now?

实际原始输出：

```text
57
```

**edit**（正确答案：33）

Lin has 43 cards, receives 2 cards, then gives away 12 cards. How many cards does Lin have?

实际原始输出：

```text
33
```

证据：[完整预测](B-seed17/predictions.jsonl)、[冻结数据](data.jsonl)。

## 案例：E-seed17 / `s20261125-transfer_2-0008`

选择规则：`first original-correct / unchanged-edit`，按冻结数据顺序。

**original**（正确答案：99）

Lin has 79 cards, receives 12 cards, then receives 8 cards. How many cards does Lin have?

实际原始输出：

```text
99
```

**paraphrase**（正确答案：99）

Lin adds two gifts of 12 and 8 cards to an initial collection of 79. How many cards are there now?

实际原始输出：

```text
107
```

**reorder**（正确答案：99）

A second gift contains 8 cards and the first contains 12. Lin had 79 cards before receiving both. How many now?

实际原始输出：

```text
97
```

**edit**（正确答案：83）

Lin has 79 cards, receives 12 cards, then gives away 8 cards. How many cards does Lin have?

实际原始输出：

```text
99
```

证据：[完整预测](E-seed17/predictions.jsonl)、[冻结数据](data.jsonl)。

## 按题型拆解

每种题型只有一个测试模板、20 个题组。表内为“原题正确 / 四题全对”的百分比；不能把单个模板上的失败推广到整个题型。

| 运行 | 卡片转移 transfer | 比较 comparison | 分组 groups |
|---|---:|---:|---:|
| A | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| B-seed17 | 80.0% / 10.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| C-seed17 | 60.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| E-seed17 | 85.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| B-seed29 | 80.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| C-seed29 | 50.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| E-seed29 | 95.0% / 5.0% | 0.0% / 0.0% | 0.0% / 0.0% |

## 测量与运行失败

- [严格格式导致数学正确回答判错](../local-pilot/development/MEASUREMENT_FAILURE.md)：保留最初 12 条验证输出，未将事后宽松解析替代主指标。
- [bfloat16 训练中断](../local-pilot/TRAINING_INTERRUPTION.md)：保留失败记录；随后完成的是显式 float32 新轮次。
- A 在当前短答案协议下有大量格式失败和截断；A 到 SFT 的分数变化同时包含格式遵循和任务学习，不能全部解释为推理提升。
- E 的事实编辑配对正确率在两个种子间从 1.7% 变为 20.0%，提示小预算结果不稳定。主指标 E−C 两种子分别为 0 与 +1.7 个百分点，尚无可靠优势。

以上现象支持下一轮做更好的测量和控制实验；它们本身不证明某种内部机制、普遍权衡或新算法有效。
