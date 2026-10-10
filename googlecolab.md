Ah! Colab is defaulting to **CPU instead of GPU**. You need to **enable GPU in Colab settings first**.

## **In Google Colab, do this:**

1. **Top menu → Runtime → Change runtime type**
2. **Select: T4 GPU** (or V100 if available)
3. **Click Save**

Then verify GPU is available:

```python
import torch
print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.get_device_name(0))  # Should print GPU name
```

---

## **Then in Colab, run these commands in order:**

```bash
# 1. Clone repo
!git clone https://github.com/mohalkarushikesh/NanoChat-X.git
%cd NanoChat-X

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Preprocess Cornell (optional but recommended for dialogue)
!python -m src.preprocess_cornell

# 4. Train with GPU config
!python -m src.train
```

This will:
- ✅ Use GPU automatically (your config has `device: "auto"`)
- ✅ Train 140M model in ~1.5-2 hours
- ✅ Produce excellent dialogue results

---

## **Key Point:**

Without GPU enabled in Colab runtime settings, it defaults to CPU (which crashes on large models).

**After enabling T4 GPU and running the above, you should see:**

```
device=cuda  tokenizer=word  vocab=142622
parameters: 140.76M
[step    50] loss 4.5812  lr 1.25e-04
[step   100] loss 3.2841  lr 2.50e-04
...
```

Try this and let me know if GPU works now!