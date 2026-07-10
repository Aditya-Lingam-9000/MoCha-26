import pandas as pd
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def assign_sites():
    print("Mapping subjects to clinical sites...")
    data_dir = ROOT / "CARE-PD" / "Canonicalized_SMPL_pickles"
    pkl_files = list(data_dir.glob("*.pkl"))
    
    subject_to_site = {}
    for pkl_file in pkl_files:
        site_name = pkl_file.stem.split('_')[0]  # e.g., '3DGait'
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            for subject_id in data.keys():
                subject_to_site[subject_id] = site_name
                
    print("Updating fusion CSV with site labels...")
    df = pd.read_csv(ROOT / "train_features_fusion.csv")
    # Using .astype(str) in case subject_ids are mixed
    df['site'] = df['subject_id'].astype(str).map(lambda x: subject_to_site.get(x, "Unknown"))
    
    df.to_csv(ROOT / "train_features_fusion.csv", index=False)
    print("Done! Site labels added.")
    print("Site distribution:")
    print(df['site'].value_counts())

if __name__ == "__main__":
    assign_sites()
