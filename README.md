# 🤖 An LLM-Agentic Workflow for Multi-Objective Optimization  
**From Toy Image Reconstruction to Cu–Mg CALPHAD Assessment**

---

## 🌐 Overview

This repository provides the code implementation for our paper:

**“An LLM-Agentic Workflow for Multi-Objective Optimization: From Toy Image Reconstruction to Cu–Mg CALPHAD Assessment.”**

We introduce **Auto-Weighter**, a novel hybrid optimization system that integrates large language model (LLM) agents into a multi-objective evolutionary algorithm. The method significantly improves the quality and efficiency of continuous optimization, particularly in data-scarce, high-dimensional domains.

---

## 🚀 Key Features

- **LLM-Embedded Optimization**  
  GPT-driven agents participate directly in the optimization loop to intelligently modify parameters and control diversity.

- **Outperforms Human Experts**  
  Achieves superior results compared to expert-designed dataset weighting in a 22-objective CALPHAD task.

- **Surrogate-Accelerated Evaluation**  
  A residual MLP model emulates a Bayesian inference-based physics engine (ESPEI), reducing evaluation time from 8 hours to ~1 minute.

- **Two Intelligent Agents**  
  - `Repair Agent`: Proposes edits or ε-threshold changes based on population statistics.
  - `Diversity Agent`: Actively prevents early collapse by perturbing over-converged parameters.

---

## 📁 Project Structure
├── agents/
│ ├── repair_agent.py # LLM-driven correction logic
│ ├── diversity_agent.py # Maintains parameter spread
│
├── optimizer/
│ ├── variation.py # SBX crossover + Gaussian mutation
│ ├── pareto_sort.py # ε-dominance-based sorting
│
├── surrogate/
│ ├── mlp_model.py # Residual MLP for fast evaluation
│ └── train_surrogate.py # Surrogate training from ESPEI traces
│
├── examples/
│ ├── toy_image/ # Toy demo task (3D input/output)
│ └── calphad_cu_mg/ # Cu–Mg CALPHAD optimization setup
│
├── utils/
│ ├── prompt_utils.py # Prompt formatting helpers
│ ├── data_utils.py # Loaders and pre/postprocessing
│
├── README.md
└── requirements.txt




