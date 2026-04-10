import json

def subsample(input_file, output_file, num_samples=100):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    subset = dict(list(data.items())[:num_samples])
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
    print(f"Subsampled {num_samples} items from {input_file} to {output_file}")

if __name__ == "__main__":
    data_dir = "../../../data/vimedcss"
    
    print("Tao du lieu con (Subset Data) de chay thu (Fast Debug)...")
    subsample(f"{data_dir}/train.json", f"{data_dir}/train_sample.json", 100)
    subsample(f"{data_dir}/valid.json", f"{data_dir}/valid_sample.json", 20)
    
    print("\n[XONG] Ban co the mo file `train_xlsr_ctc.yaml`, sua thanh:")
    print("train_annotation: !ref <data_folder>/train_sample.json")
    print("valid_annotation: !ref <data_folder>/valid_sample.json")
    print("Va set number_of_epochs: 3 de test cuc nhanh.")
