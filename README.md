# PrompTrend: Continuous Community-Driven Vulnerability Discovery and Assessment for Large Language Models

[![License: Academic](https://img.shields.io/badge/License-Academic-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![Framework](https://img.shields.io/badge/Framework-PrompTrend-green.svg)](https://github.com/yourusername/promptrend)

## 📰 Paper

**PrompTrend: Continuous Community-Driven Vulnerability Discovery and Assessment for Large Language Models**

*Tarek Gasmi, Ramzi Guesmi, Mootez Aloui, Jihene Bennaceur*

[Read Paper](https://arxiv.org/abs/xxxx.xxxxx) | [Cite](#citation)

## 🎯 Overview

PrompTrend is a comprehensive system for continuous monitoring and evaluation of LLM vulnerabilities as they emerge in online communities. Unlike traditional static benchmarks, PrompTrend bridges the gap between formal security research and grassroots vulnerability discovery through real-time threat intelligence and multi-dimensional risk assessment.

### Key Findings

- 🔍 **198 unique vulnerabilities** analyzed across 9 commercial LLMs
- 📊 **199,368 test executions** using 71 transformation strategies
- ⚡ Psychological manipulation achieves **4.9% success rate** vs **2.7% for technical obfuscations**
- 🚨 Claude 4 models show **4.1% vulnerability rate** - a regression from earlier versions
- 🌐 Discord accounts for **42.9%** of first-seen vulnerabilities

## 🏗️ System Architecture

![PrompTrend Architecture](Pvaf_scoring%20logic.png)
*Figure: Three-stage PrompTrend processing pipeline transforming social media content into actionable vulnerability intelligence*

Our framework implements:

| Component | Description | Key Features |
|-----------|-------------|--------------|
| **Multi-Agent Collection** | Platform-specific agents for vulnerability discovery | Reddit, Discord, GitHub, Twitter/X monitoring |
| **PVAF Scoring Framework** | 6-dimensional vulnerability assessment | Technical + social dynamics evaluation |
| **Transformation Engine** | 71 adversarial strategies | Psychological, linguistic, technical attacks |
| **Continuous Monitoring** | Real-time threat tracking | Longitudinal vulnerability evolution |

## 📊 Vulnerability Assessment Card

![PrompTrend Vulnerability Card](PrompTrend_Vulnerability_Card.png)
*Figure: PrompTrend vulnerability assessment card showing multi-dimensional risk metrics and cross-platform propagation tracking*

### Card Features

- **🎯 PVAF Risk Scoring** - 6-dimensional assessment with visual breakdown
- **🔍 Attack Classification** - Vector analysis and technique tagging  
- **📈 Social Signals** - Community adoption and engagement metrics
- **🧪 Benchmark Results** - Real-world testing against 9 LLM models
- **📍 Cross-Platform Journey** - Vulnerability propagation tracking

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (tested with 3.9-3.11)
- API keys for target LLM services (OpenAI, Anthropic, Azure, AWS Bedrock)
- Platform access for data collection (Reddit API, Discord Bot Token, Twitter API)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/promptrend.git
cd promptrend
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Running PrompTrend

**Full data collection pipeline:**
```bash
python main.py
```

**Run specific agents:**
```bash
python -m agents.reddit_agent.cli start
python -m agents.discord_agent.cli start
```

**Execute benchmarks:**
```bash
python scripts/run_execution_benchmark.py \
  --input-dir Data/vulnerabilities_collected \
  --enable-transformations
```

## 📊 PVAF Scoring Framework

### 6-Dimensional Risk Assessment

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Harm Potential** | 20% | Severity of potential misuse |
| **Exploit Sophistication** | 20% | Technical complexity |
| **Community Adoption** | 15% | Spread across platforms |
| **Cross-Platform Efficacy** | 15% | Success across model families |
| **Temporal Resilience** | 15% | Persistence despite patches |
| **Propagation Velocity** | 15% | Speed of community spread |

### Risk Classifications

| Score Range | Classification | Action |
|-------------|----------------|--------|
| 67-100 | High Risk | Immediate escalation |
| 34-66 | Moderate Risk | Priority investigation |
| 0-33 | Low Risk | Routine monitoring |

## 🧪 Transformation Strategies

### Attack Categories & Success Rates

| Category | Top Strategy | Success Rate | Most Vulnerable |
|----------|--------------|--------------|-----------------|
| **Psychological** | Emotional Manipulation | 4.9% | Claude 4 (11.2%) |
| **Linguistic** | Padding | 4.4% | Claude 4 Opus (9.3%) |
| **Structural** | Roleplay | 3.5% | Claude 4 Sonnet (8.7%) |
| **Technical** | Base64 Encoding | 2.7% | Azure GPT-4 (4.3%) |

## 🤖 Discord Bot Integration

Add the PrompTrend Bot to your Discord server for automated vulnerability monitoring:

👉 **[Add PrompTrend Bot](https://discord.com/oauth2/authorize?client_id=1371893848418029598&scope=bot&permissions=274877991936)**

*Note: Requires "Manage Server" permissions*

## 🔗 Related Work

Our framework builds upon and extends:

- **Static Benchmarks**: HarmBench, HELM Safety, AdvBench
- **Red Teaming**: Rainbow Teaming, automated adversarial generation
- **Community Analysis**: Social media threat intelligence platforms

## 📈 Results Summary

- **91% transferability gap** - Only 16.9% of attacks transfer across model families
- **Platform-specific patterns** - Discord vulnerabilities 2.4x more effective against Claude
- **Temporal tracking** - Framework designed for longitudinal analysis (future work)
- **PVAF validation** - 78% classification accuracy, AUC 0.72

## ⚠️ Limitations

- Current analysis is cross-sectional; longitudinal tracking in future deployments
- English-language forums only
- Black-box testing without access to model internals
- Community vulnerabilities peak at PVAF 47 (no high-severity in dataset)

## 🔮 Future Work

- Deploy longitudinal tracking capabilities
- Expand to non-English communities
- Integrate with real-time defense systems
- Develop predictive vulnerability models

## 📚 Citation

If you use PrompTrend in your research, please cite:

```bibtex
@article{promptrend2025,
  title={PrompTrend: Continuous Community-Driven Vulnerability Discovery and Assessment for Large Language Models},
  author={Gasmi, Tarek and Guesmi, Ramzi and Aloui, Mootez and Bennaceur, Jihene},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

## 📄 License

This project is licensed for academic research use. See [LICENSE](LICENSE) for full terms. For commercial inquiries, contact the authors.

## 🙏 Acknowledgments

We thank the security research community and the platforms that enable responsible vulnerability disclosure and research.

---

<p align="center">
  <a href="https://github.com/yourusername/promptrend">
    <img src="https://img.shields.io/badge/PrompTrend-LLM%20Security%20Research-blue?style=for-the-badge" alt="PrompTrend">
  </a>
</p>
