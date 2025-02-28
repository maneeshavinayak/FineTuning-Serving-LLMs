![image](https://github.com/user-attachments/assets/73c37a06-2f75-4bc2-a8b2-ede4f35595ec)


## Neural Networks and Deep Learning
Please refer below articlae for detailed an explanation of neural networks and deep learning -
https://github.com/maneeshavinayak/A-simple-explanation-of-Neural-Networks

## Transformers and Attention Mechanism

## Tokenization

## Large Language Models

## MLM

## Compute and Memory requirements to load and inference from a model
* M = [(P*4B)/(32/Q)]*1.2
* that is, Memory = [(No-of-Parameters * 4B)/(32/no-of-bits)] * 1.2

* M = GPU Memory in Gigabytes, P = Number of parameters in the model, 4B = 4 bytes per parameters (since most models use FP32 or FP16 precision), Q = Number of bits for loading the model (16-bit for FP16, 8 bit for quantized models), 1.2 is the additional 20% memory overhead.

* Example1: Memory required to host a Llama2 7B parameter model in 16-bit (FP16 precision): 
M = [(7*4)/(32/16)]*1.2 = 16.8 GB

* Example2: Memory required to host a Llama2 7B parameter model in 8-bits (quantized version): 
M = [(7*4/(32/8)]*1.2 = 8.4 GB

## LLM Optimization Techniques
- Quantization: Use quantization when you need to deploy a model on devices with limited computational resources. It helps in model size reduction, lower memory and compute requirements and faster inference. But it may come at cost of accuracy. Many quantized models are available on HuggingFace.
  
- PEFT (Parameter Efficient FineTuning) like LoRA (Low Rank Adaption) is an architectural optimization technique to reduce model size, improve efficiency, and enhance performance. This technique is used a lot for finetuning LLMs on task specific use-cases.

- Flash Attention is also an architectural optimization technique which aims at reducing computation complexity.

- Key-Value Cache - memory optimization technique which speeds up the token generation by storing some of the tensors in the attention head for use in subsequeny generation steps.

- Context Caching

- Speculative Decoding

- Distillation: Knowledge distillation involves transferring knowledge from a larger, more complex LLM (teacher) to a smaller LLM (student).

- Pruning

## Serving the model
* Local Deployment  - Local llm servers like Ollama, LM Studio etc
* Demo Deployment   - HuggingFace Spaces
* Server Deployment - HuggingFace's TGI, vLLM etc
* Edge Deployment

## Inference on Cloud
will be one another article

## Evaluation

## Securing LLMs
* Prompt Hacking
* Backdoors
* Defensive measures

## Real-time AI Application Architecture
will be one another article

## LLM Quantization
Quantization is a process aimed at simplifying data representation by reducing precision – the number of bits used. Quantization is a technique utilized within large language models (LLMs) to convert weights and activation values of high precision data, usually 32-bit floating point (FP32) or 16-bit floating point (FP16), to a lower-precision data, like 8-bit integer (INT8). This helps in faster inference, efficiency, lower power consumption, compatibility.
Types of Quantization Post-training Quantization (PTQ) - GPTQ, AWQ are classified as PTQ Quantization-aware training (QAT) GGUF(GGML)

## LLM Finetuning
* Pretraining --> Base LLM --> Finetuning using SFT, RLFH --> Chat Model
* Practical tips for finetuning LLMs - https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms

## Two techniques of Finetuning LLMs
Instruction tuning and preference alignment are essential techniques for adapting Large Language Models (LLMs) to specific tasks. Traditionally, this involves a multi-stage process: 1-Supervised Fine-Tuning (SFT) on instructions to adapt the model to the target domain, followed by 2-preference alignment methods like Reinforcement Learning with Human Feedback (RLHF) to increase the likelihood of generating preferred responses over rejected ones.

**Supervised Fine-Tuning (SFT)**
Pretraining on large corpus of data produces an auto-regressive model, the transformer based model in case of LLMs are able to predict the next token in a sequence. Instruction or Supervised Finetuning is then employed to align their results/outpus/answers with what humans expect.
Different ways to perform supervised finetuning - PEFT using LoRA, QLoRA, Prompt Tuning, Prefix Tuning, LLama-Adapters etc. The detail will be covered in the training section.

**Reinforcement Learning From Human Feedback(RLHF)** 
Reinforcement Learning with hum feedback is used to tune the model to produce responses that are aligned to human preferences. RLHF is performed in 3 steps- 1. Prepare a preference dataset. 2. Train a reward model, use a preference dataset to train a reward model in with supervised learning. 3. Use the reward model in an Reinforcement Learning loop to finetune the base LLM. The details will be explained in the section on the training using Reinforcement Learning section in this article as we would need to understand the Reinforcement Learning before understanding the RLHF process in detail. For now RLHF looks something like this -
    ![RLHF](https://github.com/user-attachments/assets/4f5c83e8-bb53-4d33-b5e9-6fbc50cb2e5e)
    
A New technique named **ORPO** to fine tune the base LLM -
ORPO is a new exciting fine-tuning technique that combines the traditional supervised fine-tuning and preference alignment stages into a single process. This reduces the computational resources and time required for training. Moreover, empirical results demonstrate that ORPO outperforms other alignment methods on various model sizes and benchmarks.

QLoRA (Quantized Low-Rank Adaptation) and ORPO (Odds Ratio Preference Optimization) are both techniques for fine-tuning large language models (LLMs), but they address different aspects: QLoRA focuses on efficient memory usage by quantizing model parameters and using LoRA adapters, while ORPO aims to streamline the alignment process by combining instruction tuning and preference alignment into a single step.

## Training Dataset Preparation for Supervised Finetuning
Finetuning can be of different types - Eg., reasoning, routing, co-pilot, chat, agent etc.
Success at finetuning depends on what tasks we want to finetune. Also wnowing what a good output vs bad output vs better output looks like.

Fine-tuning dataset is much more structured and usually is prepared in the form of a prompt template.
Examples of two prompt templates:

    ###Question:
    ###Answer:

    ###Instrution:
    {instruction}
    ###Response:
    {response}

A sample from Alpaca dateset looks like below:
    ![alpaca_dataset sample](https://github.com/user-attachments/assets/ab21985d-b2fd-45cc-8e12-7cb6b410e280)

A sample from Finance dataset that I have used to finetune Llama3.2 model:
    ![image](https://github.com/user-attachments/assets/5ab68788-bb42-4103-b7f1-5a9ea2dd5aee)

* Steps to prepare the finetuning dataset:
    - Collect instruction-response pairs
    - Concatenate pairs
    - Tokenize: pad, truncate etc
    - Split into train/test
* One tip here: Use the same template used by the chat model which you want to finetune when preparing instruction dataset.
  
## Training Arguments for supervised finetuning a LLM
* QLoRA Paramaters
* Bitsandbytes parameters
* Supervised Finetuning parameters
* Training parameters

## Training LLM using Supervised Finetuning

## Training Dataset Preparation for RLHF
Preference dataset indicates a human labelers preference between two possible model outputs for the same input.
Use preference datset to train the reward model. Generally with RLHF and LLMs, this reward model in itself is another LLM. At inference time, we take this reward model to take any prompt and a completion and return a scalar value that tells how good that completion is for the given prompt.

## Tranining Arguments for RLHF

## Training LLM using RLHF
* Different RLHF techniques - PPO, DPO etc

## Compute and memory requirements to train a model

## Conclusions

## References
- https://huggingface.co/docs/optimum/en/concept_guides/quantization
- https://huggingface.co/docs/autotrain/en/llm_finetuning_params
- https://rentry.org/llm-training#low-rank-adaptation-lora_1
- https://lightning.ai/pages/community/tutorial/lora-llm/
- https://lightning.ai/pages/community/lora-insights/#toc5
- https://huggingface.co/blog/4bit-transformers-bitsandbytes
- https://lightning.ai/pages/community/article/what-is-quantization/
- https://generativeai.pub/practical-guide-of-llm-quantization-gptq-awq-bitsandbytes-and-unsloth-bdeaa2c0bbf6#255c
- https://learn.deeplearning.ai/search?q=Finetuning
- Severals articles on google.
