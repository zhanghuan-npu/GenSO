# GenSO

[English](README.md)

## 超越线性决策规则：面向数据驱动优化的 LLM 引导表示发现

**Huan Zhang**、**Yang Wang**、**Hanzhang Qin**、**Yue Zhao**

- Huan Zhang、Yang Wang：西北工业大学管理学院
- Hanzhang Qin：新加坡国立大学工业系统工程与管理系
- Yue Zhao：北京大学汇丰商学院

[[SSRN 论文](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7127438)]

## 项目简介

为保证计算可处理性，随机优化通常将补救决策限制为仿射参数化形式，但这可能造成显著且不可消除的近似误差。大语言模型（LLM）为发现更丰富、面向具体问题的结构提供了新的可能，从而有助于缩小这一差距。我们提出**生成式随机优化**（Generative Stochastic Optimization，GenSO）：一个利用 LLM 为自适应补救决策发现表示，同时保持优化模型严谨性与可处理性的框架。

GenSO 的核心组件是**生成式线性决策规则**（Generative Linear Decision Rule，GenLDR）。在 GenLDR 中，补救决策被表示为若干闭式非线性基函数的线性组合；这些基函数由 LLM 根据问题结构和领域知识生成。因此，GenSO 在不牺牲计算可处理性与可解释性的前提下，提高了补救决策的表达能力。

我们针对固定的生成表示建立了有限样本保证，并进一步在一个由 LLM 经验缩放定律启发的假设下推导了性能界。我们在多周期报童问题，以及基于真实数据校准、包含多周期作业调度的数据中心选址问题上评估 GenSO。在两个实验场景中，GenSO 均在样本外评估中稳定优于现有基准，并发现了传统人工设计方法难以识别、但具有可解释性的非线性决策结构。

## 仓库中的项目

本仓库包含三个相互独立的项目，以及一个单独的实验结果归档：

- [`GenSO/`](GenSO/)：GenSO 实现及 LLM 引导的表示发现实验。
- [`DataCenter-demo/`](DataCenter-demo/)：独立的数据中心选址与多周期作业调度实验。
- [`MNV-demo/`](MNV-demo/)：独立的多周期报童问题实验。
- [`GenSO实验结果整理/`](GenSO实验结果整理/)：整理后的 GenSO 实验输出与分析文件。

## 主要特点

- **LLM 引导的表示发现：** 利用问题结构和领域知识，为自适应补救决策生成非线性基函数。
- **可处理且可解释：** GenLDR 在丰富决策表示的同时，保留线性优化层。
- **理论保证：** 针对固定生成表示给出有限样本保证，并给出由缩放定律启发的性能界。
- **真实场景验证：** 实验涵盖多周期报童问题，以及使用真实数据校准的数据中心选址与作业调度问题。

## 仓库结构

```text
.
├── GenSO/
│   └── eoh/
│       ├── setup.py
│       └── src/eoh/
│           ├── methods/       # LLM 引导的演化搜索
│           ├── problems/
│           │   ├── mnv/      # 多周期报童问题
│           │   └── dc/       # 数据中心选址与作业调度
│           ├── test/run.py   # GenSO 主实验脚本
│           └── utils/
├── DataCenter-demo/           # 独立的数据中心实验
├── MNV-demo/                  # 独立的报童问题实验
└── GenSO实验结果整理/          # 整理后的实验结果（Git LFS）
```

## 实验结果

`GenSO实验结果整理/` 目录包含约 3.47 GiB 的生成式实验文件，因此使用 [Git Large File Storage](https://git-lfs.com/) 存储。克隆仓库或拉取完整实验结果前，请先安装 Git LFS：

```bash
git lfs install
git lfs pull
```

## 快速开始

在仓库根目录执行：

```bash
cd GenSO
python -m pip install -e ./eoh
```

通过环境变量 `DEEPSEEK_API_KEY` 设置 DeepSeek API Key。例如，在 PowerShell 中：

```powershell
$env:DEEPSEEK_API_KEY = "your-api-key"
python eoh/src/eoh/test/run.py
```

优化实验还需要安装所选问题使用的求解器及 Python 依赖；仓库中提供的数学规划模型需要 Gurobi。

## 引用

如果 GenSO 对你的研究有所帮助，请引用：

> Huan Zhang, Yang Wang, Hanzhang Qin, and Yue Zhao. “Beyond Linear Decision Rules: LLM-Guided Representation Discovery for Data-Driven Optimization.” 2026. Available at SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7127438.

```bibtex
@misc{zhang2026beyond,
  title        = {Beyond Linear Decision Rules: LLM-Guided Representation Discovery for Data-Driven Optimization},
  author       = {Zhang, Huan and Wang, Yang and Qin, Hanzhang and Zhao, Yue},
  year         = {2026},
  howpublished = {SSRN},
  url          = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7127438}
}
```

## 联系方式

- Huan Zhang：zhhuan@mail.nwpu.edu.cn
- Yang Wang：yangw@nwpu.edu.cn
- Hanzhang Qin：hzqin@nus.edu.sg
- Yue Zhao：yzhao@phbs.pku.edu.cn
