---
title: Review Screening Analyzer
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "5.39.0"
app_file: app.py
pinned: true
---

<div align="center">
    <hr>
    <h1>Review Screening Analyzer</h1>
    <b>A Simple Literature Filtering Tool</b>
</div>

---

> [!important]
> This project is currently under development and marked as research in progress status, don't use it withour authors' permission.

> [!important]
> This is a demo code for the paper "Automated Literature Screening for Hepatocellular Carcinoma Treatment: Integrating Three Large Language Models" published in the Journal of Medical Internet Research Medical Informatics.

## 目录

- [Introduction](#Introduction)
- [Usage](#usage)
- [License](#license)
- [Contact Information](#contact-information)

---

## Introduction

Review Screening Analyzer is a literature screening tool that combines three large language models for analysis to determine the inclusion and exclusion of studies in systematic reviews based on PICOS criteria. 

This is a demo project for demonstration purposes, not a production application. If you find any bugs, please report them in the Issues.


## File Structure
```
review-screening-analyzer/
│
├── analyzer.py
├── deduplicator.py
├── file_processor.py
├── LICENSE
├── README.md
├── requirements.txt
└── app.py # Gradio Entry Point
```

## Usage

> [!warning]
> The following content is a temporary solution for local deployment.

Please ensure that [Python](https://www.python.org/) and [pip](https://pip.pypa.io/en/stable/) are installed on your system.

Create the environment variable file `.env` in the project directory [/](file:///Users/chitsanfei/Downloads/review-screening-analyzer/README.md):
```
# API Keys
DEEPSEEK_API_KEY=
QWEN_API_KEY=
GPTGE_API_KEY=
```

Then run the following commands:
```bash
bash
git clone https://github.com/chitsanfei/review-screening-analyzer.git
cd review-screening-analyzer
pip install -r requirements.txt
python3 app.py
```

## License

This project is licensed under the [MIT License](LICENSE).
```
This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
You are free to use, modify and distribute this software, provided that you keep it open source and license it under the same terms.
For more details, see the full [GNU GPL v3.0 license text](https://www.gnu.org/licenses/gpl-3.0.html).
```

## Contact Information

If you have any questions or suggestions, please contact us through the following methods:

- Email: chitsanfei@emu.ac.cn
- GitHub: [chitsanfei](https://github.com/chitsanfei)

---

Thank you for your usage and support! 🌟