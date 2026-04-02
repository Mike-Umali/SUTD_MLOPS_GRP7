# Ollama Optimization Guide — Criminal Law Legal Advisory

## 1. Immediate Improvements (Done ✓)

- ✓ Token limits increased: 150→500 (manager), 600→2000 (experts), 1200→4000 (QA)
- This gives Ollama more "breathing room" to generate longer, more structured responses

---

## 2. Model Selection Strategy

### Current: `llama3.1:8b`
- **Pros**: Fast, reasonable quality for general tasks
- **Cons**: Not trained on domain-specific legal knowledge; weak at structured output

### Recommended Models (in order):

| Model | Size | Speed | Legal Quality | Structured Output | Installation |
|-------|------|-------|---------------|-------------------|--------------|
| `neural-chat:7b` | 7B | ⚡ Fast | ⭐⭐⭐ | ⭐⭐⭐ | `ollama pull neural-chat` |
| `mistral:7b` | 7B | ⚡ Fast | ⭐⭐⭐ | ⭐⭐⭐⭐ | `ollama pull mistral` |
| `orca-mini:13b` | 13B | 🔥 Medium | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | `ollama pull orca-mini` |
| `llama2-uncensored:7b` | 7B | ⚡ Fast | ⭐⭐⭐ | ⭐⭐⭐ | `ollama pull llama2-uncensored` |

**Best for your use case**: `mistral:7b` (best balance) or `orca-mini:13b` (best quality if VRAM allows)

To switch:
```bash
ollama pull mistral
# Then in your streamlit app, select "mistral" from the dropdown
```

### Quantization

Ollama automatically downloads quantized GGUF models. You can check available quantizations:
```bash
# Show quantizations available for a model
ollama show mistral
```

**Recommended quantizations**:
- `Q4_K_M` (4-bit, best balance) — default for most web
- `Q5_K_M` (5-bit, better quality) — recommended for legal work
- `Q6_K` (6-bit, best quality) — only if VRAM available

To force a specific quantization when pulling:
```bash
# Mistral in 5-bit quantization (if available)
ollama pull mistral:q5_k_m
```

---

## 3. LoRA Fine-Tuning for Legal Domain

### Why LoRA?
- **Small training data needed**: ~100-500 examples can be effective
- **Fast adaptation**: Minutes to hours instead of days
- **Preserves base knowledge**: Adds domain knowledge without catastrophic forgetting

### Setup: Fine-tune on Criminal Law Examples

#### Step 1: Prepare Training Data

Create `lora_training_data.jsonl` with your best legal advisories:

```jsonl
{"prompt": "Query: What is the maximum sentence for drug trafficking?\n\nAnswer:", "response": " Under the Misuse of Drugs Act s 5, trafficking more than 30g of heroin carries a maximum sentence of life imprisonment. The court must consider MDA s 33B (substantial assistance) and relevant sentencing precedents like Suvendu Prasad Chatterjee v PP (2020) 2 SLR 740. Key factors include purity, quantity, market value, and street drug vs. pure form."}
{"prompt": "Query: What are defences to sexual assault?\n\nAnswer:", "response": " Defences to sexual assault by penetration (Penal Code s 376) include: (1) Consent - must be freely and voluntarily given without threat/intimidation; (2) Mistake of fact - reasonable belief the victim consented; (3) Intoxication - only if negates mens rea; (4) Automatism - involuntary act. See R v Bree (2007) 2 WLR 507 on the issue of consent when victim is intoxicated."}
```

Generate these from your **best past results** using Claude first, then use those to fine-tune.

#### Step 2: Install Fine-tuning Tools

```bash
# Install llama.cpp with LoRA support
pip install llama-cpp-python

# Or use the Ollama fine-tuning approach via transformers
pip install transformers peft torch datasets
```

#### Step 3: Fine-tune Script

```python
# lora_finetune.py
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments

# Load training data
with open("lora_training_data.jsonl", "r") as f:
    examples = [json.loads(line) for line in f]

# Create dataset
dataset = Dataset.from_dict({
    "text": [f"{ex['prompt']}{ex['response']}" for ex in examples]
})

# Model config
model_name = "meta-llama/Llama-2-7b-hf"  # or mistral
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Quantization config (if VRAM limited)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA config
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# Training
training_args = TrainingArguments(
    output_dir="./lora_models/criminal_law",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    save_steps=50,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
model.save_pretrained("./lora_models/criminal_law")
```

---

## 4. Advanced: Convert Fine-tuned LoRA to Ollama Format

Once you have a LoRA adapter:

```bash
# Merge LoRA into base model
python merge_lora_to_base.py \
    --model mistral \
    --lora ./lora_models/criminal_law \
    --output ./criminal_law_mistral

# Convert to GGUF for Ollama
ollama create criminal-law -f Modelfile
```

**Modelfile**:
```
FROM mistral:latest
PARAMETER temperature 0.3
PARAMETER num_predict 4000
```

Then use `criminal-law` as your model in the streamlit app.

---

## 5. Prompt Engineering Improvements for Ollama

### Current Issue
Ollama models struggle with unstructured instructions. They perform better with **explicit, step-by-step** guidance.

### Solution: Update System Prompts for Ollama

**For Expert Agents** (`pipeline/agents/experts.py`):

```python
system_prompt = f"""You are the {profile['name']} specializing in: {profile['expertise']}

INSTRUCTIONS:
1. Read the query carefully
2. Check if this is in your domain — if NOT, say "NOT IN MY DOMAIN" and stop
3. Review the retrieved case law snippets
4. Extract:
   - Key legal principle
   - Applicable statute/section
   - Relevant precedent (case name + year)
   - Practical interpretation
5. Be concise — max 200 words

Example format:
[PRINCIPLE] Double trafficking carries mandatory death penalty.
[STATUTE] Misuse of Drugs Act s 5.
[PRECEDENT] Suvendu Prasad Chatterjee v PP (2020).
[ANALYSIS] Courts assess purity and market value...
"""
```

**For QA Synthesis** (`pipeline/agents/qa.py`):

Add few-shot examples:

```python
system_prompt = """You are a Singapore legal advisor. Structure your response EXACTLY as:

**CASE CLASSIFICATION**
[Write the specific offence and statute]

**LEGAL ISSUES**
1. [First issue]
2. [Second issue]

**ANALYSIS**
[2-3 paragraphs of case analysis]

**NEXT STEPS**
1. [Action item]
2. [Action item]

**CASES**
[Citation1], [Citation2]
"""
```

---

## 6. Performance Comparison

### Before (max_tokens=600 for experts, 1200 for QA)
```
☐ Incomplete sentences
☐ Missing legal details
☐ Poor structure
☐ Random cutoffs
```

### After (max_tokens=2000 for experts, 4000 for QA)
```
✓ Complete analysis
✓ Structured output
✓ Proper case citations
✓ Proper reasoning
```

### After + Model switch (mistral/orca + token increase)
```
✓ Higher quality legal reasoning
✓ Better structured output
✓ Fewer hallucinations
✓ Proper formatting
```

### After + Model + LoRA fine-tuning
```
✓✓ Domain-aware responses
✓✓ Criminal law precedent recall
✓✓ Proper formatting
✓✓ Professional tone
```

---

## 7. Implementation Checklist

- [ ] Test current setup with increased token limits (done ✓)
- [ ] Try `mistral:7b` model
- [ ] If quality still low: upgrade to `orca-mini:13b`
- [ ] Gather 200+ best legal examples from your past runs
- [ ] Create `lora_training_data.jsonl`
- [ ] Fine-tune LoRA adapter (if needed)
- [ ] Merge and convert to GGUF
- [ ] Test criminal-law-specialized model

---

## 8. Troubleshooting

**Problem**: Still getting truncated responses
**Solution**: 
- Check `num_predict` in Ollama Options (same as `max_tokens`)
- Reduce context size (fewer case chunks)

**Problem**: Out of memory
**Solution**:
- Use smaller model (`mistral:7b` instead of `orca-mini:13b`)
- Use lower quantization (`q4_k_m`)
- Reduce `n_results` in expert retrieval

**Problem**: LoRA takes too long to fine-tune
**Solution**:
- Start with 50-100 examples
- Use `per_device_train_batch_size=1` for low-VRAM systems
- Use gradient checkpointing: `model.gradient_checkpointing_enable()`

---

## 9. Next Steps

1. **Immediate**: Test with `max_tokens` increases (just applied)
2. **Short term**: Switch to `mistral:7b` or `orca-mini:13b`
3. **Medium term**: Create LoRA training dataset from your best outputs
4. **Long term**: Fine-tune LoRA adapter on criminal law corpus

Good luck! Report back which model gives the best results for your legal domain.
