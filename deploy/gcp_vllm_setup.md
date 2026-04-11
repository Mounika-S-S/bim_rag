# GCP vLLM Deployment Guide

This guide outlines the steps needed to deploy the Fine-Tuned Llama 3.1 model on Google Cloud Platform using `vLLM` on a dedicated Compute Node.

## 1. Create a VM Instance with NVIDIA GPUs

We recommend an L4 or A100 instance depending on budget. L4 is heavily cost-optimized for inference.

Using `gcloud`:
```bash
gcloud compute instances create vllm-inference-node \
    --project=llm-bim-rag \
    --zone=us-central1-c \
    --machine-type=g2-standard-12 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=200GB
```

## 2. Install vLLM

SSH into the newly created instance and install vLLM alongside HuggingFace Hub:
```bash
gcloud compute ssh vllm-inference-node --zone=us-central1-a

# inside the VM:
pip install vllm huggingface_hub
```

## 3. Upload the Fine-tuned LoRA weights
After running `src/finetuning/qlora_finetune.py`, you will have a directory `models/llama3.1-bim-rag-lora`. You should merge these weights with the base Llama 3.1 model and upload them to a GCP Cloud Storage Bucket.

```bash
gsutil cp -r models/llama3.1-bim-rag-lora gs://YOUR_BUCKET_NAME/model/
```

Then on the Compute Node, download them:
```bash
gsutil cp -r gs://YOUR_BUCKET_NAME/model/ /home/user/model/
```

## 4. Run the vLLM OpenAI-Compatible Server

vLLM provides an API server natively compatible with the OpenAI spec. We will run it, attaching our fine-tuned LoRA adapters dynamically:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules bim-rag=/home/user/model/llama3.1-bim-rag-lora \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192
```

## 5. Configure Your Backend Environment

To connect the BIM RAG pipeline to the newly deployed customized LLM, update the `.env` file on your RAG backend server:

```env
# Previous Grok API Key
GROQ_API_KEY=your_groq_api_key

# NEW GCP Configs
CUSTOM_LLM_URL=http://<VM_EXTERNAL_IP>:8000/v1
LLM_MODEL_NAME=bim-rag
```

Now, the `llm_client.py` will route all reasoning prompts to your self-hosted API Server.
