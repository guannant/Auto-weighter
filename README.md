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


- **Two Intelligent Agents**  
  - `Repair Agent`: Proposes edits or ε-threshold changes based on population statistics.
  - `Diversity Agent`: Actively prevents early collapse by perturbing over-converged parameters.

---



