![image](https://github.com/user-attachments/assets/73c37a06-2f75-4bc2-a8b2-ede4f35595ec)


## Neural Networks and Deep Learning
Please refer below articlae for detailed an explanation of neural networks and deep learning -
https://github.com/maneeshavinayak/A-simple-explanation-of-Neural-Networks

## Attention Mechanism & Transformer Architecture
'Attention is all you need': This is the title of the research paper https://arxiv.org/abs/1706.03762 published by Google in 2017 on which all the language models are based. The original paper proposes a non-recurrent sequence-to-sequence encoder-decoder model which works in parallel to improve machine translation.

Attention mechanism is a technique used in deep learning that allows the model to selectively focus on specific parts of the input data when making predictions. This selective focus enables the model to weigh and priortize information adaptively, improving its capacity to detect relevant patterns and connections in the data.

Different types of Attention Mechanism:
- Self-attention
- Scaled-dot-product Attention
- Multi-Head Attention

**Transformer Architecture**

![Transformer architecture](https://github.com/user-attachments/assets/7741d3ce-50b4-4f2b-91de-e2907b818e8d)

## Tokenization & Embeddings

## Large Language Models

## Compute and Memory requirements to load and inference from a model
* M = [(P*4B)/(32/Q)]*1.2
* that is, Memory = [(No-of-Parameters * 4B)/(32/no-of-bits)] * 1.2

* M = GPU Memory in Gigabytes, P = Number of parameters in the model, 4B = 4 bytes per parameters (since most models use FP32 or FP16 precision), Q = Number of bits for loading the model (16-bit for FP16, 8 bit for quantized models), 1.2 is the additional 20% memory overhead.

* Example1: Memory required to host a Llama2 7B parameter model in 16-bit (FP16 precision): 
M = [(7*4)/(32/16)]*1.2 = 16.8 GB

* Example2: Memory required to host a Llama2 7B parameter model in 8-bits (quantized version): 
M = [(7*4/(32/8)]*1.2 = 8.4 GB

* Example3: Memory required to host a Llama3 1B (1.23B) parameter model in 16-bits(FP16 or BF16 precision):
M = [(1.23*4)/(32/16)]*1.2 = 3 GB or 4 GB approx

## LLM Model & Inference Optimization Techniques
* **Quantization** \
Use quantization when you need to deploy a model on devices with limited computational resources. It helps in model size reduction, lower memory and compute requirements and faster inference. But it may come at the cost of accuracy. Many quantized models are available on HuggingFace. Quantization is a process aimed at simplifying data representation by reducing precision – the number of bits used. Quantization is a technique utilized within large language models (LLMs) to convert weights and activation values of high precision data, usually 32-bit floating point (FP32) or 16-bit floating point (FP16), to a lower-precision data, like 8-bit integer (INT8) or even lower. This helps in faster inference, efficiency, lower power consumption, compatibility.
Types : Post-Training Quantization (PTQ): Applied after training, easier to implement but may have slight accuracy loss.
Quantization-Aware Training (QAT):  Incorporates quantization during training, more complex but generally better accuracy.
  
* **PEFT** (Parameter Efficient FineTuning)\
PEFT techniques like LoRA (Low Rank Adaption) are architectural optimization technique to reduce model size, improve efficiency, and enhance performance. This technique is used a lot for finetuning LLMs on task specific use-cases. Other techniques include QLoRA, Adapter Tuning, Prefix Tuning etc. \
LoRA: Adds trainable low-rank matrices in parallel to existing weight matrices in the transformer layers. Only these low-rank matrices are trained, while the original weights are frozen.

* **Flash Attention** \
An architectural optimization technique which aims at reducing computation complexity.

* **Key-Value Cache** \
Memory optimization technique which speeds up the token generation by storing some of the tensors in the attention head for use in subsequent generation steps.

* **Distillation** \
Knowledge distillation involves transferring knowledge from a larger, more complex LLM (teacher) to a smaller LLM (student).

* **Pruning** \
Removes less important connections (weights) in the neural network

* **Batching**
Processes multiple inference requests together in a single forward pass.

* **Continuous Batching** \
Dynamically groups incoming requests into batches as they arrive, maximizing GPU utilization without waiting for fixed batch sizes.

* **Asynchronous Inference** \
Decouples request processing from response delivery.

* **Prompt Caching** \
Caching the output for frequently used prompts to avoid re-computation. Useful for chatbots.

* **Speculative Decoding**

**FlashAttention-2** can be combined with other optimization techniques like quantization to further speedup inference.

## Serving the model
A detailed article will be covered on inference and serving the LLM models. This article is more about optimization and finetuning LLMs.
* Local Deployment  - Local llm servers like Ollama, LM Studio etc
* Demo Deployment   - HuggingFace Spaces
* Server Deployment - HuggingFace's TGI, vLLM etc
* Edge Deployment   - MLC

## Inference on Cloud/Real-time AI Application Architecture
will be one another article

## Evaluation Metrics
**1. LLM Specific Metrics**
* Perplexity / Fluency
  - Measures how well a probability model predicts a sample.
  - Lower perplexity indicates a better model.
  - Often used to evaluate language model fluency.
  - Limitations: Doesn't assess semantic correctness or factual accuracy.
* Accuracy
  - Measures how often an LLM produces correct response to prompts. It is typically expressed as a percentage of successful task completions.
  - Metric used widely for Classification tasks, question-answering tasks.
  - Limitations: 
* Fidelity/Factuality
  - Evaluates how accurately the LLM's output reflects the source material or real-world facts.
  - Requires external knowledge sources or fact-checking mechanisms.
  - Methods: 1. Fact-checking against knowledge bases. 2. Comparison with reference documents. 3. Using other LLMs to verify the output.
* Coherence/Consistency
  - Assesses the logical flow and consistency of the LLM's output.
  - Evaluates whether the generated text makes sense and avoids contradictions.
  - particularly important for longer texts, such as essays or articles where maintaining a consistent narrative is key.
* Relevance
  - Measures how well the LLM's output addresses the given prompt or task.
  - Subjective and often requires human evaluation.
* Safety/Toxicity
  - Evaluates whether the LLM generates harmful, biased, or toxic content.
  - Uses content moderation tools and human evaluation.
* Bias
  - Evaluates if the model produces biased results against certain groups of people.
* Code Execution Metrics
  - For LLMs that generate code, metrics that evaluate if the generated code executes, and if it produces the correct output are very important.

**2. Human Evaluation**
* Human Evaluation
  - Involves human raters assessing the LLM's output based on various criteria (e.g., fluency, relevance, accuracy).
  - Provides the most comprehensive and nuanced evaluation.
  - Can be time-consuming and expensive.
* Test Suites
  - Benchmark datasets designed to test specific capabilities of LLMs (e.g., reasoning, common sense).
  - Examples: MMLU (Massive Multitask Language Understanding, BIG-bench (Beyond limitation Game Benchmark, HellaSwag (commonsense reasoning)
* Elo Rankings
  - Used to compare the performance of different LLMs by having them compete against each other in pairwise comparisons.
  - Provides a relative ranking of LLMs.
 
**3. Task specific Metrics** 
* BLEU (Bilingual Evaluation Understudy): Used for machine translation tasks, comparing model responses to human-generated references.
  - Measures the overlap of n-grams (sequences of words) between the generated text and reference text.
  - Limitation: Doesn't capture semantic meaning well.
* ROUGE (Recall-Oriented Understudy for Gisting Evaluation): Commonly used for summarization tasks and machine translation.
  - Measures the overlap of n-grams or word sequences between the generated text and reference text, focusing on recall.
  - Variants: ROUGE-L (longest common subsequence), ROUGE-N (n-gram overlap).
* METEOR (Metric for Evaluation of Translation with Explicit Ordering)
  - Considers synonyms and paraphrases, providing a more semantically aware evaluation than BLEU.
  - Combines precision and recall.
  
**Key Considerations**
* **Task-Specific Metrics**       - The choice of evaluation metrics depends on the specific task for which the LLM is being used.
* **Subjectivity**                - Many aspects of LLM evaluation are subjective, requiring human judgment.
* **Comprehensive Evaluation**    - A comprehensive evaluation should involve a combination of automatic metrics and human evaluation.
* **Compliance with Regulations** - Ensure LLM evaluation benchmarks align with data protection laws (eg., GDPR, CCPA etc),
                                  Industry specific regulations (eg., HIPAA), Ethical AI guidelines from recognized bodies or institutions.
* **Ethical Considerations**      - Evaluation should also consider the ethical implications of the LLM's outputs,
                                  such as bias and toxicity, hate speech or discreminatory language, misinformation or conspiracy theories.
* **Responsible AI**              - Fairness, reliability and safety, privacy and security, inclusiveness           
* **LLM as a Judge**              - LLMs are increasingly being used to judge other LLMs. This is a rapidly advancing field.

By using a combination of these metrics, researchers and developers can gain a more complete understanding of LLM performance.

**Tools and Libraries to use for evaluation metrics**
* DeepEval: offers 20+ metrics out of the box. This open-source framework offers a wide range of evaluation metrics for various LLM tasks, including RAG and fine-tuning. It emphasizes detailed assessments, focusing on accuracy, ethical considerations, and performance.
* RAGAS:
* MLflow:
* DeepChecks:
* TruLens: Focuses on transparency and interpretability in LLM evaluation, Good for understanding and explaining LLM decision-making.
* LangSmith (Anthropic): Offers evaluation tools tailored for language models, with a focus on bias detection and safety.

## Securing LLMs
* Prompt Hacking/Injection
* Insecure Output Handling
* Training Data Poisoning
* Sensitive Information Disclosure
* Model denial of Service
* Supply Chain Vulnerabilties
* Excessive Agency
* Model Theft

**Key considerations when securing Applications**
* **OWASP** Top 10 list of security risks for LLM applications which provides a valuable framework for understanding and addressing potential vulnerabilities.
* **Open-Source vs Commercial Tools** - When choosing tools, it is important to evaluate needs and prioritize the risks that are most relevant to the application.
* How well the security tools integrate with our existing infrastructure and workflows.

**Different Tool Categories and Examples which are available to secure LLM Applications**
* **Prompt Injection and Input Validation**
    - **Proprietary Cloud Provider Guardrails**
        - AWS Guardrails (within Amazon Bedrock)
        - Google Cloud Vertex AI Safety
        - Azure AI Content Safety
    - NVIDIA NeMo Guardrails
  - **Open Source and Community Driven**
    * LLM Guard (Protect AI): Focuses on detecting and neutralizing prompt injection attacks, as well as filtering harmful content and preventing data leakage.
    * Rebuff: Specifically designed to prevent prompt injection attacks, using heuristics, LLM analysis, and vector databases.
    * Purple Llama (Meta): Includes "Llama Guard" and "Prompt Guard" which are tools designed to moderate inputs and outputs, and detect malicious prompts.
    * PromptTools
    * Outlines
* **Output Validation and Content Moderation**
    - WhyLabs: Provides a security and observability platform for AI applications, including real-time threat detection, bias detection, and customizable security guardrails.
    - Granica Screen: Focuses on privacy and safety, detecting sensitive information, bias, and toxicity in data and model outputs.
* **API Security and Monitoring** \
Pynt: Enhances API discovery and identifies LLM-based APIs, monitoring their usage and detecting vulnerabilities. 
* **Adversarial Robustness** \
Adversarial Robustness Toolbox (ART): A Python library for adversarial training, helping LLMs defend against various attacks.
* **Data Privacy and Security** \
Private AI: Provides data minimization solutions, identifying and anonymizing sensitive information.

## LLM Finetuning
* Pretraining --> Base LLM --> Finetuning using Supervised FineTuning, Reinforcement Learning From Human Feedback --> Chat Model
* Any of the techniques SFT or RLHF can be used to align the base LLM or they can be used in combination as well.
* Practical tips for finetuning LLMs - https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms

## Techniques for Finetuning LLMs
Instruction tuning and preference alignment are essential techniques for adapting Large Language Models (LLMs) to specific tasks. Traditionally, this involves a multi-stage process: 1-Supervised Fine-Tuning (SFT) on instructions to adapt the model to the target domain, followed by 2-preference alignment methods like Reinforcement Learning From Human Feedback (RLHF) to increase the likelihood of generating preferred responses over rejected ones.

**Supervised Fine-Tuning (SFT)** \
    Different ways to perform supervised finetuning - PEFT using LoRA, QLoRA, Prompt Tuning, Prefix Tuning, LLama-Adapters etc

**Reinforcement Learning From Human Feedback(RLHF)** \ 
    Reinforcement Learning with human feedback is used to tune the model to produce responses that are aligned to human preferences. RLHF is performed in 3 steps- 1. Prepare a preference dataset. 2. Train a reward model, use a preference dataset to train a reward model in with supervised learning. 3. Use the reward model in an Reinforcement Learning loop to finetune the base LLM. The details will be explained in the section on the training using Reinforcement Learning section in this article as we would need to understand the Reinforcement Learning before understanding the RLHF process in detail. For now RLHF looks something like this -
    ![RLHF](https://github.com/user-attachments/assets/4f5c83e8-bb53-4d33-b5e9-6fbc50cb2e5e)
    
**ORPO**
A New technique named **ORPO** to fine tune the base LLM -
ORPO is a new exciting fine-tuning technique that combines the traditional supervised fine-tuning and preference alignment stages into a single process. This reduces the computational resources and time required for training. Moreover, empirical results demonstrate that ORPO outperforms other alignment methods on various model sizes and benchmarks.

**QLoRA** (Quantized Low-Rank Adaptation) and **ORPO** (Odds Ratio Preference Optimization) are both techniques for fine-tuning large language models (LLMs), but they address different aspects: QLoRA focuses on efficient memory usage by quantizing model parameters and using LoRA adapters, while ORPO aims to streamline the alignment process by combining instruction tuning and preference alignment into a single step.

## Training Dataset Preparation for Supervised Finetuning
Finetuning can be of different types - Eg., reasoning, routing, co-pilot, chat, agent etc.
Success at finetuning depends on what tasks we want to finetune. Also wnowing what a good output vs bad output vs better output looks like.

Fine-tuning dataset is much more structured and usually is prepared in the form of a prompt template.
Examples of two prompt templates:

    ###Question:
    ###Answer:

    ###Instruction:
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
LoRA: Adds trainable low-rank matrices in parallel to existing weight matrices in the transformer layers. Only these low-rank matrices are trained, while the original weights are frozen.
In simple words - When we train a fully connected layers in the neural network, the weight matrices usually have full ranks, which is a technical term meaning that a matrix does not have any linearly dependent (ie., redundant) rows and columns. In contrast to full rank, low rank means that the matrix has redundant rows and columns. So, while the weights of a pretrained model have full rank on the pretrained tasks, the LoRA authors point out that pretreined large language models have a low "intrinsic dimension" when they are adapted to a new task. This means that we can decompose the new weight matrix for the adapted task into lower-dimensional(smaller) matrices without loosing too much important information.

* QLoRA: QLoRA builds upon LoRA by adding quantization to further reduce memory footprint.
  
**LoRA Paramaters** 
* r (LoRA Rank): rank of the low-rank matrices used for adaptation. A smaller r leads to a simpler low-rank matrix, which results in fewer parameters to learn during adaptation. This can lead to faster training and potentially reduced computational requirements. However, with a smaller r, the capacity of the low-rank matrix to capture task-specific information decreases. This may result in lower adaptation quality, and the model might not perform as well on the new task compared to a higher r. The original LoRA paper recommends a rank of 8 (r = 8) as the minimum amount. Keep in mind that higher ranks lead to better results and higher compute requirements. The more complex your dataset, the higher your rank will need to be.
* LoRA Alpha: A scaling factor that controls how much the LoRA adapters influence the original pre-trained model. It scales the learned low-rank matrices before they are added to the original weights. Often set as 2 * r(LoRA Rank).
* LoRA Dropout: Dropout probibility applied to the LoRA adapter layers. Dropout is a regularization technique that helps prevent overfitting.
* Target Modules: specifies which modules or layers in the pre-trained model will have LoRA adapters injected.

**Bitsandbytes parameters for QLoRA** 
* bits: specifies the number of bits to which the pre-trained model weights are quantized. primarily 4-bit, 8-bit can also be seen.
* bnb_4bit_quant_type: specifies the quantization type used for 4-bit quantization. Typical value nf4 (as NormalFloat4 is designed to be more information preserving than standard float), fp4 also present as option.
* bnb_4bit_compute_dtype: specifies the data type used for computation during matrix multiplications in the quantized layers. Often set to higher precision. Typical values are bf16 (bfloat16) or fp16 (float16) for faster and more memory-efficient computation. fp32 (float32) can be used for higher precision but is slower and more memory-intensive.
* double quantization: applies a second quantization step.

Use bitsandbytes library parameters \
use_4bits = True \
bnb_4bit_quant_type = 'nf4' \
bnb_4bit_compute_dtype = 'float16' \
use_nested_quant = False (for small models, use it incase you want to use for very large models)

**Training parameters** 
* Number of training epochs. The number of times the entire training dataset is passed through the model. Task-dependent often 1-5-10 epochs are sufficient.
num_train_epochs = 1

* Batch Size: The number of training examples processed in each iteration. Larger batch size can improve tranining speed but require more memory.
    Batch size per GPU for training. It is hardware depemdent, larger is often faster. 
    per_device_train_batch_size = 1

    Batch size per GPU for evaluation
    per_device_eval_batch_size = 1

* Gradient Accumulation: is a technique to train models with large batch sizes than would otherwise fit into GPU memory. Instead of computing gradients and updating weights after each mini-batch, gradient accumulation involves processing several mini-batches(called accumulation steps) and accumulating (summing or averaging) the gradients calculated from each mini-batch. Then the models weights are updated, simulating a larger batch size. This results in weights being updated less frequently.

    number of steps before performing a backward/update pass. 
    gradient_accumulation_steps = 1

* Enable gradient checkpointing. use gradient checkpointing to save memory
    gradient_checkpointing = True

* Maximum gradient normal (gradient clipping). Set the maximum norm for gradient clipping, which is critical for preventing gradients from exploding during backpropagation. Default is 1.0.
max_grad_norm = 0.3

* Define the weight decay rate for regularization, which helps prevent overfitting by penalizing larger weights. Default is 0.0. Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

* Optimizer to use
optim = "paged_adamw_32bit"

* Initial learning rate (AdamW optimizer). The step size taken during optimization.
learning_rate = 2e-4

* Learning rate scheduler (constant a bit better than cosine for LoRA/QLoRA)  ---> not sure
lr_scheduler_type = "constant"

* Number of training steps (overrides num_train_epochs)
max_steps = -1

* Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

* Group sequences into batches with same length. Saves memory and speeds up training considerably
group_by_length = True

* Save checkpoint every X updates steps
save_steps = 25

* Log every X updates steps
logging_steps = 25

* Enable fp16/bf16 training (set bf16 to True with an A100 GPU)
fp16 = True
bf16 = False

**Supervised Finetuning parameters**
* max sequence length:
* packing:
* device_map: load the entire model on GPU or not

## Training LLM using Supervised Finetuning
I have used QLoRA to finetune the Llama model over Google Colab as QLoRA has significantly reduced the memory and compute requirements that we can finetune the model using free T4-GPU on Google Colab. A-100 GPU is also availble on Google colab but will incur some additional cost.
QLoRA improves over LoRA by quantizing the transformer model to 4-bit precision and using pages optimizers to handle memory spikes.

One thing to note of finetuning using LoRA/QLoRA is that it performs well on original source domain as the goal of LoRA (low-rank adaptation) is not substantially modify all model parameters but updating fewer parameters.

Thus choose a pre-trained model of the domain which you want to fine-tune.

## Training Process using Reinforcement Learning From Human Feedback

A RLHF process/pipeline looks something like below:

![RLHF Pipeline](https://github.com/user-attachments/assets/283e5ce4-c103-44ab-aff5-09c71df20863)

Preference dataset indicates a human labelers preference between two possible model outputs for the same input. It can be the trickiest part of the proces. 

Use preference datset to train the reward model. Generally with RLHF and LLMs, this reward model in itself is another LLM. At inference time, we take this reward model to take any prompt and a completion and return a scalar value that tells how good that completion is for the given prompt.

Once we have trained the reward model, we will use it in the final step of the process where Reinforcement Learning of RLHF come into play. Our goal here is to tune the base LLM to produce completion that maximize the reward given by the reward model. so if the base model produces the completions that better align with the preferences of the people who labeled the data, then it will recieve higher rewards from the reward model.

To do this, we introduce a second dataset known as prompt dataset which is just a dataset of prompts, no completions. ....more to be written...

Different RLHF techniques - PPO, DPO etc

## Training Dataset Preparation for RLHF
* Preference Dataset
* Prompt Dataset

## Tranining Arguments for RLHF

## Compute and memory requirements to train a model
**Estimated Memory Requirements for Training Llama-3.2-1B**
* Full Finetuning (Standard approach)
  - GPU Memory (VRAM): For full fine-tuning in full precision (FP32), you would likely need around 20GB to 30GB of VRAM.
  - System RAM: You would also need a reasonable amount of system RAM, estimated to be at least 32GB to 64GB to handle data loading and processing effectively.
  - While full FP32 fine-tuning is possible on higher-end consumer GPUs, it's less memory-efficient.
* Full Finetuning with Mixed Precision
  - GPU Memory (VRAM): Using mixed precision training (FP16 or BF16) significantly reduces VRAM usage. You can likely fine-tune a 1.2B Llama model with around 10GB to 15GB of VRAM when using mixed precision.
  - System RAM: System RAM requirements are reduced compared to FP32, but having 32GB to 64GB would still be beneficial for smoother data handling.
* PEFT Techniques - LoRA, Adapters
  - GPU Memory (VRAM): PEFT methods like LoRA are highly effective for reducing VRAM. For a 1.2B Llama model with LoRA, you could likely fine-tune with around 6GB to 10GB of VRAM or even less, depending on the LoRA rank (r) and other configurations.
  - System RAM: System RAM requirements are further lowered with PEFT, and 16GB to 32GB of RAM should be sufficient for most datasets.
  - LoRA makes fine-tuning 1.2B models extremely accessible, even on GPUs with 8GB to 12GB of VRAM, like some of the mid-range to high-end consumer cards.
* QLoRA
  - GPU Memory (VRAM): QLoRA maximizes memory efficiency. You can likely fine-tune a 1.2B Llama model with as little as 4GB to 8GB of VRAM or even less (potentially as low as ~4GB). This is exceptionally memory-friendly and opens up fine-tuning to GPUs with even limited VRAM.
  - System RAM: System RAM requirements are also minimal with QLoRA, and 16GB to 32GB of RAM should be more than enough.
  - QLoRA makes fine-tuning 1.2B models incredibly accessible, potentially even on some older or lower-end GPUs with 8GB or even 4GB of VRAM (though 4GB might be very constrained).

## Conclusions

## References
  - 'Attention Is All You Need' - https://arxiv.org/pdf/1706.03762
  - https://huggingface.co/docs/optimum/en/concept_guides/quantization
  - https://huggingface.co/docs/autotrain/en/llm_finetuning_params
  - https://rentry.org/llm-training#low-rank-adaptation-lora_1
  - https://lightning.ai/pages/community/tutorial/lora-llm/
  - https://github.com/microsoft/LoRA
  - https://lightning.ai/pages/community/lora-insights/#toc5
  - https://huggingface.co/blog/4bit-transformers-bitsandbytes
  - https://lightning.ai/pages/community/article/what-is-quantization/
  - https://generativeai.pub/practical-guide-of-llm-quantization-gptq-awq-bitsandbytes-and-unsloth-bdeaa2c0bbf6#255c
  - https://learn.deeplearning.ai/search?q=Finetuning
  - https://lightning.ai/blog/gradient-accumulation/
  - https://learn.deeplearning.ai/search?q=Reinforcement+Learning+From+Human+Feedback
  - https://sebastianraschka.com/blog/2024/llm-research-insights-instruction.html
  - https://modelbench.ai/blogs/llm-evaluation-benchmarks-a-comprehensive-guide
  - https://www.datacamp.com/blog/llm-evaluation
  - https://developer.nvidia.com/blog/mastering-llm-techniques-evaluation/
  - https://ieeexplore.ieee.org/document/10556082
  - https://docs.nvidia.com/nemo/guardrails/latest/index.html
  - https://docs.pynt.io/documentation
  - https://github.com/protectai/rebuff
  - https://github.com/meta-llama/PurpleLlama
  - https://www.llama.com/trust-and-safety/
  - https://adversarial-robustness-toolbox.readthedocs.io/en/latest/
  - https://github.com/Trusted-AI/adversarial-robustness-toolbox
  - https://docs.ragas.io/en/latest/
  - https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/
  
  - https://www.google.com/url?sa=E&source=gmail&q=https://developer.nvidia.com/blog/inference-optimization-techniques-deep-learning/
  - arXiv. https://arxiv.org/abs/1712.05877
  - arXiv. https://arxiv.org/abs/2401.13654
  - https://blog.mlc.ai/2023/04/27/faster-and-cheaper-transformer-inference-using-speculative-decoding/
  - arXiv. https://arxiv.org/abs/2309.06180
  
  - https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#:~:text=Half%20precision%20(also%20known%20as,network%2C%20allowing%20training%20and%20deployment
  - https://heidloff.net/article/efficient-fine-tuning-lora/#:~:text=PEFT%20methods%20only%20fine%2Dtune,because%20fine%2Dtuning%20large%2Dscale
  - https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/lora-qlora#:~:text=QLoRA%20uses%20less%20GPU%20memory,support%20higher%20max%20sequence%20lengths.&Both%20methods%20offer%20similar%20accuracy%20improvements.
  - https://www.databricks.com/blog/llama-finetuning#:~:text=As%20the%20sequence%20length%20of,potentially%20exceeding%20GPU%20memory%20limits.
  - https://arxiv.org/html/2406.02290v1#:~:text=3.1%20Gradient%20Checkpointing,-Report%20issue%20for&It%20instead%20recalculates%20many%20activations,which%20helps%20conserve%20GPU%20memory.
  - https://huggingface.co/learn/nlp-course/chapter11/4#:~:text=Using%20TRL%20with%20PEFT,tuning%20to%20reduce%20memory%20requirements.
  - Severals articles on google.
