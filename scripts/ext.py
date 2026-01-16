import json
from pathlib import Path

save_dir = Path("alzheimer_papers")
metadata_file = save_dir / "metadata.json"
output_file = save_dir / "pmcids.txt"

with open(metadata_file, 'r', encoding='utf-8') as f:
    papers_data = json.load(f)

pmcids = [paper.get('pmcid') for paper in papers_data if paper.get('pmcid')]

with open(output_file, 'w', encoding='utf-8') as f:
    for pmcid in pmcids:
        f.write(f"{pmcid}\n")

print(f"Extracted {len(pmcids)} PMCIDs to {output_file}")