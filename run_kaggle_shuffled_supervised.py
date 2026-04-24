import subprocess
from pathlib import Path


WORKDIR = Path("/kaggle/working/DepthEnhance")
SHUFFLE_SCRIPT = WORKDIR / "shuffle_data.py"
SEED = 42


SHUFFLE_JOBS = [
    {
        "input_dir": "/kaggle/input/datasets/akiyanguyen/polypdataset/polypDataset_final1/kvasir_SEG/Train/images",
        "output_json": "/kaggle/working/kvasir_SEG_seed_42.json",
    },
    {
        "input_dir": "/kaggle/input/datasets/akiyanguyen/polypdataset/polypDataset_final1/CVC_ClinicDB/Train/images",
        "output_json": "/kaggle/working/CVC_ClinicDB_seed_42.json",
    },
    {
        "input_dir": "/kaggle/input/datasets/akiyanguyen/polypdepth-dataset/TrainDataset/images",
        "output_json": "/kaggle/working/kvasir_SEG_CVC_ClinicDB_seed_42.json",
    },
]

def run_shuffle_jobs():
    for job in SHUFFLE_JOBS:
        cmd = [
            "python",
            str(SHUFFLE_SCRIPT),
            "--input-dir",
            job["input_dir"],
            "--seed",
            str(SEED),
            "--output-json",
            job["output_json"],
        ]
        print(f"[shuffle] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=WORKDIR, check=True)




def main():
    run_shuffle_jobs()

if __name__ == "__main__":
    main()
