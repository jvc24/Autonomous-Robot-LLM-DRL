# Autonomous-Robot-LLM-DRL



Autonomous robotic manipulation framework combining teleoperation data, deep reinforcement learning, and Large Language Models (LLMs) to enable sequential and multitask execution from natural language instructions in household kitchen environments.

> **Note:** Due to licensing restrictions, LLaMA model weights are not distributed with this repository.

---

## Project Overview

This project presents a complete pipeline for language-guided robotic manipulation in household environments. The main idea is to first train individual atomic manipulation skills from teleoperation data using deep reinforcement learning, and then combine these learned skills for sequential and multitask execution.

The framework supports:

- learning atomic tasks such as microwave, hinge, cabinet, and other household manipulation actions
- combining individual task policies into multitask execution pipelines
- using sentence embeddings or Large Language Models (LLMs) to map natural language instructions into executable task sequences
- enabling flexible and interactive robotic behavior from human instructions

The overall pipeline can be summarized as:

**human demonstrations → policy learning with SAC → task sequencing → language-guided execution**

---
![Pipeline](Overall_flowchart.png)
---
## Repository Structure

### Atomic Task Folders
The individual tasks are organized in separate folders such as:

- `microwave/`
- `hinge/`
- `cabinet/`

and other task-specific folders.

These folders contain the training and evaluation code for the corresponding atomic manipulation skills.

### LLM Folder
The `LLM/` folder contains the code for training and fine-tuning the LLaMA model using **LoRA fine-tuning**.

### Sentence Embedding Folder
The `sentence_embedding/` folder contains the code for training the **sentence transformer model** used for instruction embedding and task retrieval.

---

## Inference Options

There are **three main ways** to run multitask execution in this project.

### 1. LLaMA-based multitask inference
The main final inference file for multitask execution with **fine-tuned LLaMA + SAC** is:

```bash
python test_multitask_1b.py
```

### 2. Sentence embedding-based multitask inference
The sentence embedding-based multitask file uses the trained sentence transformer model and does not require any LLaMA license or gated model access.

```bash
python test_embedding.py
```

### 2. Predefined multitask execution
The following file runs multitask execution using task sequences already defined inside the file:

```bash
python test_multitask.py
```
