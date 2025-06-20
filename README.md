<div align="center">
    <h1>
        JiuZhou: Open Foundation Language Models for Geoscience
    </h1>
</div>



\[ English | [中文](README_zh.md) \]

## 🎉 News
- **[2025-05]** Paper [*TagRouter: Learning Route to LLMs through Tags for Open-Domain Text Generation Tasks*](https://arxiv.org/abs/2506.12473) has been accepted by the top NLP conference *ACL*. [Model Download](https://huggingface.co/itpossible/TagGenerator).
- **[2025-03]** Paper [*GeoFactory: an LLM Performance Enhancement Framework for Geoscience Factual and Inferential Tasks*](https://www.tandfonline.com/doi/full/10.1080/20964471.2025.2506291) has been accepted by the journal *Big Earth Data*. [Data Download](https://huggingface.co/datasets/itpossible/WikiRAG).
- **[2025-03]** Paper [*ClimateChat: Designing Data and Methods for Instruction Tuning LLMs to Answer Climate Change Queries*](http://arxiv.org/abs/2506.13796) has been accepted by the International Conference on Learning Representations (*ICLR*). [Model Download](https://huggingface.co/itpossible/ClimateChat).
- **[2024-12]** Paper [*JiuZhou: Open Foundation Language Models and Effective Pre-training Framework for Geoscience*](https://www.tandfonline.com/doi/full/10.1080/17538947.2025.2449708) has been accepted by the *International Journal of Digital Earth*. [Model Introduction](https://deepwiki.com/THU-ESIS/JiuZhou). [Project Repository](https://github.com/THU-ESIS/JiuZhou).
- **[2024-09]** Released chat model [ClimateChat](https://huggingface.co/itpossible/ClimateChat).
- **[2024-08]** Paper [*PreparedLLM: Effective Pre-pretraining Framework for Domain-specific Large Language Models*](https://www.tandfonline.com/doi/full/10.1080/20964471.2024.2396159) has been accepted by the journal *Big Earth Data*. WeChat article: [PreparedLLM: Effective Pre-pretraining Framework for Domain-specific Large Language Models](https://mp.weixin.qq.com/s/ugJQ9tbp6Y87xA3TOWteqw). [Model Download](https://huggingface.co/itpossible/Prepared-Llama).
- **[2024-08]** Released chat model [Chinese-Mistral-7B-Instruct-v0.2](https://huggingface.co/itpossible/Chinese-Mistral-7B-Instruct-v0.2), featuring significantly improved language understanding and multi-turn conversation capabilities.
- **[2024-06]** Released chat model [JiuZhou-Instruct-v0.2](https://huggingface.co/itpossible/JiuZhou-Instruct-v0.2), with significantly enhanced language understanding and multi-turn conversation capabilities.
- **[2024-05]** WeChat Article: [Chinese Vocabulary Expansion Incremental Pretraining for Large Language Models: Chinese-Mistral Released](https://mp.weixin.qq.com/s/PMQmRCZMWosWMfgKRBjLlQ).
- **[2024-03]** Released base model [Chinese-Mistral-7B-v0.1](https://huggingface.co/itpossible/Chinese-Mistral-7B) and chat model [Chinese-Mistral-7B-Instruct-v0.1](https://huggingface.co/itpossible/Chinese-Mistral-7B-Instruct-v0.1). [Model Introduction](https://deepwiki.com/THU-ESIS/Chinese-Mistral). [Project Repository](https://huggingface.co/itpossible/Chinese-Mistral).
- **[2024-03]** Released JiuZhou's base version [JiuZhou-base](https://huggingface.co/itpossible/JiuZhou-base), instruct version [JiuZhou-instruct-v0.1](https://huggingface.co/itpossible/JiuZhou-Instruct-v0.1), and [intermediate checkpoints](https://huggingface.co/itpossible). [Model Introduction](https://deepwiki.com/THU-ESIS/JiuZhou). [Project Repository](https://github.com/THU-ESIS/JiuZhou).
- **[2024-01]** Completed training of Chinese-Mistral and JiuZhou, and commenced model evaluation.



## Table of Contents

- [Introduction](#introduction)
- [Download](#download)
- [Inference](#inference)
- [Model Performance](#model-performance)
- [Model Training Process](#model-training-process)
- [Model Training Code](#model-training-code)
- [Citations](#citations)
- [Acknowledgments](#acknowledgments)

## Introduction
The field of geoscience has amassed a vast amount of data, necessitating the extraction and integration of diverse knowledge from this data to address global change challenges, promote sustainable development, and accelerate scientific discovery. Foundation language models initially learn and integrate knowledge autonomously through self-supervised pre-training on extensive text data. Subsequently, they acquire the capability to solve geoscience problems through instruction tuning. However, when the foundational language models lack sufficient geoscience expertise, instruction tuning with relevant data can lead to the generation of content that is inconsistent with established facts. To improve the model's accuracy and practicality, a robust geoscience foundational language model is urgently needed.<br>

This study uses [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) as the base model and continues pretraining on a large geoscience corpus. It also incorporates the [domain-specific large language model *pre*-pretraining framework (PreparedLLM)](https://www.tandfonline.com/doi/full/10.1080/20964471.2024.2396159) and the "two-stage pre-adaptation pre-training" algorithm to build the geoscience large language model, JiuZhou.


## Download

| **Model Series**      | **Model**                           | **Download Link**                                           | **Description**                                                  |
|-----------------------|-------------------------------------|------------------------------------------------------------|------------------------------------------------------------------|
| **JiuZhou**           | JiuZhou-base                        | [Huggingface](https://huggingface.co/itpossible/JiuZhou-base) | Base model (Rich in geoscience knowledge)                     |
| **JiuZhou**           | JiuZhou-Instruct-v0.1               | [Huggingface](https://huggingface.co/itpossible/Chinese-Mistral-7B-Instruct-v0.1) | Instruct model (Instruction alignment caused a loss of some geoscience knowledge, but it has instruction-following ability) <br> LoRA fine-tuned on Alpaca_GPT4 in both Chinese and English and GeoSignal |
| **JiuZhou**           | JiuZhou-Instruct-v0.2               | [HuggingFace](https://huggingface.co/itpossible/Chinese-Mistral-7B-Instruct-v0.2)<br>[Wisemodel](https://wisemodel.cn/models/itpossible/Chinese-Mistral-7B-Instruct-v0.2) | Instruct model (Instruction alignment caused a loss of some geoscience knowledge, but it has instruction-following ability) <br> Fine-tuned with high-quality general instruction data |
| **ClimateChat**       | ClimateChat                         | [HuggingFace](https://huggingface.co/itpossible/ClimateChat)<br>[Wisemodel](https://wisemodel.cn/models/itpossible/ClimateChat) | Instruct model <br> Fine-tuned on JiuZhou-base for instruction following |
| **Chinese-Mistral**   | Chinese-Mistral-7B                  | [HuggingFace](https://huggingface.co/itpossible/Chinese-Mistral-7B-v0.1)<br>[Wisemodel](https://wisemodel.cn/models/itpossible/Chinese-Mistral-7B-v0.1)<br>[ModelScope](https://www.modelscope.cn/models/itpossible/Chinese-Mistral-7B-v0.1) | Base model                                                      |
| **Chinese-Mistral**   | Chinese-Mistral-7B-Instruct-v0.1    | [HuggingFace](https://huggingface.co/itpossible/Chinese-Mistral-7B-Instruct-v0.1)<br>[Wisemodel](https://wisemodel.cn/models/itpossible/Chinese-Mistral-7B-Instruct-v0.1)<br>[ModelScope](https://www.modelscope.cn/models/itpossible/Chinese-Mistral-7B-Instruct-v0.1) | Instruct model <br> LoRA fine-tuned with Alpaca_GPT4 in both Chinese and English |
| **Chinese-Mistral**   | Chinese-Mistral-7B-Instruct-v0.2    | [HuggingFace](https://huggingface.co/itpossible/Chinese-Mistral-7B-Instruct-v0.2)<br>[Wisemodel](https://wisemodel.cn/models/itpossible/Chinese-Mistral-7B-Instruct-v0.2) | Instruct model <br> LoRA fine-tuned with a million high-quality instructions |
| **PreparedLLM**       | Prepared-Llama                      | [Huggingface](https://huggingface.co/itpossible/Prepared-Llama)<br>[Wisemodel](https://wisemodel.cn/models/itpossible/PREPARED-Llama) | Base model <br> Continual pretraining with a small number of geoscience data <br> Recommended to use JiuZhou |


## Inference
Below is an example of inference code using JiuZhou-Instruct-v0.2.
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model_path = "itpossible/JiuZhou-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)

text = "What is geoscience?"
messages = [{"role": "user", "content": text}]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
outputs_id = model.generate(inputs, max_new_tokens=600, do_sample=True)
outputs = tokenizer.batch_decode(outputs_id, skip_special_tokens=True)[0]
print(outputs)
```

## Model Performance

### Geoscience Ability
We evaluate the performance of JiuZhou using the GeoBench benchmark.<br>
JiuZhou outperforms GPT-3.5 in objective tasks:
<p align="center">
    <br>
    <img src="image/objective_score.png" width="800"/>
    <br>
</p>

JiuZhou also scores higher than baselines across six criteria in subjective tasks:
<p align="center">
    <br>
    <img src="image/subjective_score.png" width="800"/>
    <br>
</p>

### General Ability

We evaluate the performance of JiuZhou using three benchmark datasets: C-Eval, CMMLU, and MMLU.<br>
Compared to other variants of Llama and Mistral models, JiuZhou shows outstanding performance:
<p align="center">
    <br>
    <img src="image/general_score.png" width="800"/>
    <br>
</p>

## Model Training Process

### Training Corpus
The corpus consists of 50 million general documents and 3.4 million geoscience-related documents.
<p align="center">
    <br>
    <img src="image/JiuZhou-Corpus.png" width="800"/>
    <br>
</p>

### Training Framework
We use the JiuZhou-Framework proposed in this study.
<p align="center">
    <br>
    <img src="image/JiuZhou-Framework.png" width="800"/>
    <br>
</p>

### Two-stage Pre-adaptation Pre-training (TSPT)
TSPT improves the efficiency of using limited geoscience data and overcomes some of the technical bottlenecks in continual pretraining for LLMs.<br>
The difference between TSPT and single-stage training algorithms:
<p align="center">
    <br>
    <img src="image/TSPT.png" width="800"/>
    <br>
</p>
Comparison of TSPT and one-stage pre-training algorithm performance:
<p align="center">
    <br>
    <img src="image/TSPT_score.png" width="800"/>
    <br>
</p>


## Model Training Code
We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to fine-tune JiuZhou.

### Project Deployment
```bash
git clone https://github.com/THU-ESIS/JiuZhou.git
cd JiuZhou
pip install -e ".[torch,metrics]"
```
### Model Training
Pre-training：
```bash
llamafactory-cli train examples/train_lora/JiuZhou_pretrain_sft.yaml
```
Instruction-tuning：
```bash
llamafactory-cli train examples/train_lora/JiuZhou_lora_sft.yaml
```
Chat with the fine-tuned JiuZhou:：
```bash
llamafactory-cli chat examples/inference/JiuZhou_lora_sft.yaml
```
Merge the instruction-tuned LoRA weights with the original JiuZhou weights:
```bash
llamafactory-cli export examples/merge_lora/JiuZhou_lora_sft.yaml
```

## Citations
```bibtex
@article{chen2024preparedllm,
  title={PreparedLLM: Effective Pre-Pretraining Framework for Domain-Specific Large Language Models},
  author={Chen, Zhou and Lin, Ming and Wang, Zimeng and Zang, Mingrun and Bai, Yuqi},
  journal={Big Earth Data},
  volume={8},
  number={4},
  pages={649--672},
  year={2024},
  doi={10.1080/20964471.2024.2396159}
}

@article{chen2025jiuzhou,
  title={JiuZhou: Open Foundation Language Models and Effective Pre-Training Framework for Geoscience},
  author={Chen, Zhou and Lin, Ming and Zang, Mingrun and Wang, Zimeng and Li, Juanzi and Bai, Yuqi},
  journal={International Journal of Digital Earth},
  volume={18},
  number={1},
  year={2025},
  doi={10.1080/17538947.2025.2449708}
}

@article{chen2025geofactory,
  title={GeoFactory: An LLM Performance Enhancement Framework for Geoscience Factual and Inferential Tasks},
  author={Chen, Zhou and Wang, Xiao and Zhang, Xinan and Lin, Ming and Liao, Yuanhong and Li, Juanzi and Bai, Yuqi},
  journal={Big Earth Data},
  year={2025},
  month={May},
  pages={1--33},
  doi={10.1080/20964471.2025.2506291}
}

@inproceedings{chen2025tagrouter,
  title={TagRouter: Learning Route to LLMs through Tags for Open-Domain Text Generation Tasks},
  author={Chen, Zhou and Wei, Zhiqiang and Bai, Yuqi and Xiong, Xue and Wu, Jianmin},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)},
  year={2025},
  organization={Association for Computational Linguistics}
}

@inproceedings{chen2025climatechat,
  title={ClimateChat: Designing Data and Methods for Instruction Tuning LLMs to Answer Climate Change Queries},
  author={Chen, Zhou and Wang, Xiao and Liao, Yuanhong and Lin, Ming and Bai, Yuqi},
  booktitle={Proceedings of the 2025 International Conference on Learning Representations (ICLR) Workshop},
  year={2025},
  organization={International Conference on Learning Representations}
}
```

## Acknowledgments
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [OpenCompass](https://github.com/open-compass/opencompass)
- [K2](https://github.com/davendw49/k2)
- [GeoGalactica](https://github.com/geobrain-ai/geogalactica)
- [BB-GeoGPT](https://github.com/AGI-GIS/BB-GeoGPT)
