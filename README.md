<!--
---
title: Review Screening Analyzer
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "3.25.0"
app_file: app.py
pinned: true
---
-->

<div align="center">
    <hr>
    <h1>Review Screening Analyzer</h1>
    <b>筛选文献小工具</b>
</div>

---

> [!important]
> This project is currently under development and marked as research in progress status, don't use it withour authors' permission.

## 目录

- [简介](#简介)
- [功能](#功能)
- [使用方法](#使用方法)
- [许可证](#许可证)
- [联系信息](#联系信息)

---

## 简介

Review Screening Analyzer 是一个强大的工具，旨在帮助研究人员和学者高效地筛选和分析文献。通过自动化的流程和智能的算法，您可以节省大量的时间和精力。

## 功能

- **文献筛选**：快速筛选出符合条件的文献。
- **数据分析**：提供详细的数据分析报告。
- **可视化**：生成直观的图表和图形。
- **多语言支持**：支持多种语言的文献。

## 文件结构
```
review-screening-analyzer/
│
├── test/
│   └── picos_analyzer.py
├── LICENSE
├── README.md
├── requirements.txt
└── main.py
```

## 使用方法

> [!warning]
> 下面的内容为临时方案。

请确保您的系统上已安装 [Python](https://www.python.org/) 和 [pip](https://pip.pypa.io/en/stable/)。

在项目目录`/`下创建环境变量`.env`：
```
# API Keys
DEEPSEEK_API_KEY=
QWEN_API_KEY=
GPTGE_API_KEY=
```

然后在运行以下命令：
```bash
bash
git clone https://github.com/chitsanfei/review-screening-analyzer.git
cd review-screening-analyzer
pip install -r requirements.txt
cd test
python picos_analyzer.py --help
```

## 许可证

本项目采用 [MIT 许可证](LICENSE)。
```
This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
You are free to use, modify and distribute this software, provided that you keep it open source and license it under the same terms.
For more details, see the full [GNU GPL v3.0 license text](https://www.gnu.org/licenses/gpl-3.0.html).
```


## 联系信息

如有任何问题或建议，请通过以下方式联系我们：

- 邮箱: chitsanfei@emu.ac.cn
- GitHub: [chitsanfei](https://github.com/chitsanfei)

---

感谢您的使用与支持！🌟


