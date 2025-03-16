from datasets import load_dataset

ds = load_dataset("iberbench/lm-eval-results")
print(ds["train"])
for row in ds["train"]:
    print(row)