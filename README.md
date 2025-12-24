# 🕌 Arabic Diacritization System  
### 🔤 Multi-Pipeline Automatic Tashkeel for Arabic Text

This project presents a comprehensive **Arabic diacritization (Tashkeel) system** that restores missing diacritical marks in undiacritized Arabic text.  
Instead of relying on a single approach, the system implements and evaluates **multiple diacritization pipelines**, enabling structured comparison between classical, neural, and hybrid methods.

The project is designed for **research, benchmarking, and real-world NLP integration**, with an emphasis on linguistic correctness, modular experimentation, and reproducibility.

---

## 📌 Problem Definition

Arabic text is typically written without diacritics, leading to substantial lexical and grammatical ambiguity. Accurate diacritization is essential for:

- 🗣️ Text-to-Speech (TTS)
- 🎙️ Automatic Speech Recognition (ASR)
- 🧠 Syntactic and semantic parsing
- 📚 Language learning and corpus annotation

The task is formulated as a **sequence labeling problem**, where each character is assigned its appropriate diacritic based on contextual information.

---

## 🏗️ System Overview

All pipelines share a common preprocessing and evaluation backbone while differing in **feature representation and modeling strategy**.



---

## 🧹 Common Preprocessing Pipeline

Applied consistently across all variants:

- Unicode normalization  
- Diacritic stripping and separation  
- Character-level tokenization  
- Sentence segmentation  
- Dataset splitting (train / validation / test)  

---

## 🔄 Implemented Pipeline Variants

### 🟦 1. Rule-Based Baseline Pipeline

A lightweight baseline using deterministic linguistic heuristics.

**Characteristics**
- Hand-crafted diacritic rules  
- Minimal contextual awareness  
- Extremely fast inference  

**Purpose**
- Establish a lower-bound baseline  
- Analyze ambiguity patterns in Arabic text  

---

### 🟨 2. Statistical Character-Level Pipeline

A classical machine learning approach modeling diacritization as character classification.

**Features**
- Sliding character context windows  
- N-gram-based representations  
- Frequency-driven diacritic selection  

**Models**
- Naive Bayes / Logistic Regression (or equivalent)

**Purpose**
- Evaluate statistical context modeling  
- Compare classical vs neural approaches  

---

### 🟩 3. BiLSTM Sequence Labeling Pipeline

A neural sequence model capturing bidirectional character dependencies.

**Architecture**
- Character embeddings  
- Bidirectional LSTM layers  
- Softmax output per character  

**Strengths**
- Long-range contextual modeling  
- Robust handling of syntactic ambiguity  

---

### 🟪 4. BiLSTM-CRF Pipeline

An enhanced neural pipeline combining representation learning with structured prediction.

**Architecture**
- Character embeddings  
- BiLSTM encoder  
- Conditional Random Field (CRF) decoding  

**Advantages**
- Enforces valid diacritic transitions  
- Improves sequence-level consistency  
- Reduces illegal diacritic combinations  

---

### 🟫 5. Hybrid Linguistic-Neural Pipeline

A hybrid approach combining linguistic cues with neural modeling.

**Features**
- Character embeddings  
- Word-level contextual features  
- Optional morphological indicators  
- Neural sequence modeling  

**Purpose**
- Inject linguistic bias  
- Improve generalization on morphologically rich forms  

---

## 🛠️ Post-processing

- Diacritic reconstruction  
- Output formatting  
- Error inspection and analysis utilities  

---

## 📊 Evaluation Strategy

All pipelines are evaluated using identical datasets and metrics to ensure fair comparison:

- 🎯 Character-level accuracy  
- 📉 Diacritic Error Rate (DER)  
- 🧩 Word-level accuracy  

---

## ⚙️ Technology Stack

- 🐍 **Language:** Python  
- 🔥 **Deep Learning:** PyTorch  
- 🧠 **Sequence Models:** BiLSTM, CRF  
- 📦 **Data Processing:** NumPy, Pandas  
- 📈 **Evaluation:** Custom metric scripts  



