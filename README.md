## Neural Networks and Deep Learning
Please refer below articlae for detailed an explanation of neural networks and deep learning -
https://github.com/maneeshavinayak/A-simple-explanation-of-Neural-Networks

## Transformers and Attention Mechanism

## Tokenization

## Large Language Models

## Compute and Memory requirements to load and inference from a model
* M = [(P*4B)/(32/Q)]*1.2
* that is, Memory = [(no-of-parameters * 4B)/(32/no-of-bits)] * 1.2

* M = GPU Memory in Gigabytes, P = Number of parameters in the model, 4B = 4 bytes per parameters (since most models use FP32 or FP16 precision), Q = Number of bits for loading the model (16-bit for FP16, 8 bit for quantized models), 1.2 is the additional 20% memory overhead.

* Example1: Memory required to host a Llama2 7B parameter model in 16-bit (FP16 precision): 
M = [(7*4)/(32/16)]*1.2 = 16.8 GB

* Example2: Memory required to host a Llama2 7B parameter model in 8-bits (quantized version): 
M = [(7*4/(32/8)]*1.2 = 8.4 GB

## Compute and memory requirements to train a model

## Quantization
Quantization is a process aimed at simplifying data representation by reducing precision – the number of bits used. Quantization is a technique utilized within large language models (LLMs) to convert weights and activation values of high precision data, usually 32-bit floating point (FP32) or 16-bit floating point (FP16), to a lower-precision data, like 8-bit integer (INT8). This helps in faster inference, efficiency, lower power consumption, compatibility.
Types of Quantization Post-training Quantization (PTQ) - GPTQ, AWQ are classified as PTQ Quantization-aware training (QAT) GGUF(GGML)

## Inference Optimization

## Finetuning LLMs
* Pretraining --> Base LLM --> Finetuning using SFT, RLFH --> Chat Model
* Practical tips for finetuning LLMs - https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms

## Two techniques of Finetuning LLMs
Instruction tuning and preference alignment are essential techniques for adapting Large Language Models (LLMs) to specific tasks. Traditionally, this involves a multi-stage process: 1-Supervised Fine-Tuning (SFT) on instructions to adapt the model to the target domain, followed by 2-preference alignment methods like Reinforcement Learning with Human Feedback (RLHF) to increase the likelihood of generating preferred responses over rejected ones.

**Supervised Fine-Tuning (SFT)**
    Different ways to perform supervised finetuning - PEFT using LoRA, QLoRA, Prompt Tuning, Prefix Tuning etc

**Reinforcement Learning from Human Feedback(RLHF)** 
* Different RLHF techniques - PPO, DPO
* RLHF is performed in 3 steps- 1. Prepare a preference dataset. 2. Train a reward model, use a preference dataset to train a reward model in with supervised learning. 3. Use the reward model in an Reinforcement Learning loop to finetune the base LLM.

A New technique named **ORPO** to fine tune the base LLM -
ORPO is a new exciting fine-tuning technique that combines the traditional supervised fine-tuning and preference alignment stages into a single process. This reduces the computational resources and time required for training. Moreover, empirical results demonstrate that ORPO outperforms other alignment methods on various model sizes and benchmarks.

QLoRA (Quantized Low-Rank Adaptation) and ORPO (Odds Ratio Preference Optimization) are both techniques for fine-tuning large language models (LLMs), but they address different aspects: QLoRA focuses on efficient memory usage by quantizing model parameters and using LoRA adapters, while ORPO aims to streamline the alignment process by combining instruction tuning and preference alignment into a single step.

## Training Dataset Preparation for Supervised Finetuning

## Training Arguments for supervised finetuning a LLM
* QLoRA Paramaters
* Bitsandbytes parameters
* Supervised Finetuning parameters
* Training parameters

## Training Dataset Preparation for RLHF
## Tranining Arguments for RLHF

## Evaluation

## Serving the model

## Securing LLMs

## Conclusions

## References
