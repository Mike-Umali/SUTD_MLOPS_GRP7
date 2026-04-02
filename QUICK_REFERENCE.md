# Quick Implementation Guide — Ollama Optimization

## What Was Changed ✓

### 1. Token Limits (DONE)
```
Manager Agent:    150 → 500 tokens
Expert Agents:    600 → 2000 tokens  
QA Synthesis:   1200 → 4000 tokens
```
This gives Ollama models room to generate complete, structured responses without truncation.

### 2. Temperature Control (DONE)
Added `temperature` parameter to `ollama_chat()` in `pipeline/llm.py`:
- Default: `0.3` (for structured legal analysis)
- Lower = more deterministic/consistent
- Higher = more creative (0.7-0.9 if needed)

---

## Immediate Next Steps (15 minutes)

### Step 1: Test Current Setup
```bash
cd c:\Users\jesly\OneDrive\SUTD_MLOPS_GRP7
streamlit run app.py
```
Select "Ollama (local)" and test a legal query. You should see longer, more complete responses.

### Step 2: Try a Better Model
```bash
ollama pull mistral
ollama pull neural-chat
```

Test with `mistral:7b` in the Streamlit app dropdown. Should be noticeably better than `llama3.1:8b`.

---

## Phase 1: Model Optimization (1-2 hours)

### Try These Models (in order)
1. **`mistral:7b`** ← Start here
   ```bash
   ollama pull mistral
   ```
   Then select in Streamlit dropdown

2. **`neural-chat:7b`** ← Good balance
   ```bash
   ollama pull neural-chat
   ```

3. **`orca-mini:13b`** ← Best if you have VRAM
   ```bash
   ollama pull orca-mini
   ```

### Which one is best?
- **Fast machine with low VRAM**: `mistral:7b` ⭐⭐⭐
- **Balanced**: `neural-chat:7b` or `mistral:7b` ⭐⭐⭐⭐
- **High VRAM (16GB+)**: `orca-mini:13b` ⭐⭐⭐⭐⭐

Run tests and compare quality.

---

## Phase 2: Prompt Engineering (30 minutes)

### Option A: Quick Improvement
Copy the improved prompts from `ollama_prompts.py` into your agents:

1. Edit `pipeline/agents/manager.py`
   - Replace the system prompt for `_run_manager_ollama()` with `MANAGER_PROMPT_OLLAMA`

2. Edit `pipeline/agents/experts.py`
   - Replace system_prompt for Ollama with domain-specific prompts

3. Edit `pipeline/agents/qa.py`
   - Replace SYSTEM_PROMPT for Ollama with `QA_PROMPT_OLLAMA`

### Option B: Gradual Testing
Test each agent separately first. Only update if a change improves quality.

---

## Phase 3: LoRA Fine-tuning (2-4 hours)

### When to Use LoRA
- If Phase 1 & 2 still don't meet your quality bar
- Have 100+ good examples from past runs
- Want to specialize the model on criminal law

### Quick Start
```bash
# 1. Install dependencies
pip install transformers peft torch datasets

# 2. Prepare training data
# Create lora_training_data.jsonl with examples from your best past outputs
# (template provided in lora_training_data.jsonl)

# 3. Run fine-tuning
python lora_finetune.py \
    --data lora_training_data.jsonl \
    --model mistral \
    --output ./lora_models/criminal_law \
    --epochs 3 \
    --batch-size 2 \
    --lr 1e-4
```

### After Fine-tuning
- LoRA adapter saves to `./lora_models/criminal_law/lora_adapter`
- Merge with base model (requires additional tools)
- Convert to GGUF for Ollama

---

## Testing Checklist

- [ ] Test current setup with higher token limits
- [ ] Try `mistral:7b` model
- [ ] Compare output quality vs. Claude
- [ ] If still poor: try `orca-mini:13b`
- [ ] If better but not perfect: update prompts from `ollama_prompts.py`
- [ ] If nearly perfect: collect examples and fine-tune LoRA
- [ ] If still lacking: consider Claude only (cost vs. local trade-off)

---

## Files Created

| File | Purpose |
|------|---------|
| `OLLAMA_OPTIMIZATION.md` | Comprehensive optimization guide (quantization, LoRA, model selection) |
| `ollama_prompts.py` | Specialized prompts for Ollama (domain-specific expert prompts) |
| `lora_finetune.py` | LoRA fine-tuning script with full argument parsing |
| `lora_training_data.jsonl` | Example training data (criminal law domain) |
| `QUICK_REFERENCE.md` | This file! |

---

## Troubleshooting

**Q: Still getting truncated responses?**
A: Check that token limits are increased (grep for `max_tokens` in code). If yes, model is hitting its context limit. Reduce `n_results` in retrieval.

**Q: Model is slow?**
A: Using a too-large model for your system. Try `mistral:7b` (faster) or check VRAM usage.

**Q: Output is nonsensical?**
A: Temperature too high, or model not right for task. Try `temperature=0.1` or different model.

**Q: Out of memory errors?**
A: Your system can't run Ollama + retriever together. 
Solutions:
- Use smaller model (`mistral` instead of `orca-mini`)
- Reduce `n_results` from 5 to 2-3
- Run Ollama on separate machine
- Use Claude API instead

---

## Expected Results

### Before (original setup)
- Token limit: 600 tokens for experts
- Output: Incomplete, truncated, disjointed
- Quality: ⭐⭐ (poor)

### After Phase 1 (token increase only)
- Token limit: 2000 tokens for experts
- Output: More complete, but still generic
- Quality: ⭐⭐⭐ (acceptable)

### After Phase 2 (better model + improved prompts)
- Model: `mistral:7b` or `orca-mini:13b`
- Output: Structured, legal citations, clear reasoning
- Quality: ⭐⭐⭐⭐ (good)

### After Phase 3 (LoRA fine-tuning)
- Fine-tuned on criminal law corpus
- Output: Domain-aware, proper precedent recall
- Quality: ⭐⭐⭐⭐⭐ (excellent)

---

## Cost Comparison

| Approach | Cost | Speed | Quality | Effort |
|----------|------|-------|---------|--------|
| Claude API (paid) | $$ per query | ⚡ | ⭐⭐⭐⭐⭐ | 0 (just use it) |
| Ollama (Phase 1) | Free | 🔥 Slow | ⭐⭐⭐ | 15 min |
| Ollama + better model | Free | 🔥 Slow | ⭐⭐⭐⭐ | 1 hour |
| Ollama + prompts | Free | 🔥 Slow | ⭐⭐⭐⭐ | 30 min |
| Ollama + LoRA | Free | 🔥 Slow | ⭐⭐⭐⭐⭐ | 4 hours |

**Recommendation**: Start with Phase 1 + `mistral:7b` (1.5 hours total). If that's good enough, stop. If not, do LoRA.

---

## Questions?

- See `OLLAMA_OPTIMIZATION.md` for detailed info on quantization and LoRA
- Check `ollama_prompts.py` for prompt examples
- Run `python lora_finetune.py --help` for fine-tuning options
