import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

class TextEntityMerger:

    """
    Merges extracted texts with NER entities
    """
    
    def __init__(
        self,
        texts_dir: str,
        entities_dir: str,
        output_dir: str,
        llm_cleaned: bool = False
    ):
        self.texts_dir = Path(texts_dir)
        self.entities_dir = Path(entities_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.llm_cleaned = llm_cleaned
        
    def load_text(self, article_id: str) -> Optional[Dict]:

        """
        Loads article text from JSON file
        """

        possible_names = [
            f"{article_id}_text.json",
            f"{article_id}_cleaned.json",
            f"{article_id}.json"
        ]
        
        for filename in possible_names:
            filepath = self.texts_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"⚠️  Error loading {filename}: {e}")
                    continue
        
        return None
    
    def load_entities(self, article_id: str) -> Optional[Dict]:

        """
        Loads entities for article
        """

        filepath = self.entities_dir / f"{article_id}.json"
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading entities for {article_id}: {e}")
            return None
    
    def merge_single_article(self, article_id: str) -> Optional[Dict]:

        """
        Merges text and entities for single article
        """

        text_data = self.load_text(article_id)
        if not text_data:
            print(f"❌ Text not found: {article_id}")
            return None
        
        # Загружаем entities
        entities_data = self.load_entities(article_id)
        if not entities_data:
            print(f"⚠️  Entities not found: {article_id}, using empty list")
            entities_data = {"article_title": article_id, "entities": []}
        
        # Объединяем
        merged = {
            "article_id": article_id,
            "full_text": text_data.get("full_text", ""),
            "entities": entities_data.get("entities", []),
            "metadata": {
                "char_count": text_data.get("char_count", len(text_data.get("full_text", ""))),
                "page_count": text_data.get("page_count", 0),
                "entity_count": len(entities_data.get("entities", [])),
                "llm_cleaned": self.llm_cleaned
            }
        }
        
        return merged
    
    def merge_all(self) -> List[Dict]:

        """
        Merges all articles
        """
  
        entity_files = list(self.entities_dir.glob("*.json"))
        
        if not entity_files:
            print(f"❌ No entity files found in {self.entities_dir}")
            return []
        
        print(f"\n📚 Found {len(entity_files)} entity files")
        print(f"📁 Texts directory: {self.texts_dir}")
        print(f"📁 Entities directory: {self.entities_dir}")
        print(f"💾 Output directory: {self.output_dir}\n")
        
        merged_data = []
        failed = []
        
        for entity_file in tqdm(entity_files, desc="Merging articles"):
            article_id = entity_file.stem  
            
            merged = self.merge_single_article(article_id)
            
            if merged:
                merged_data.append(merged)
                
                output_file = self.output_dir / f"{article_id}_merged.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(merged, f, indent=2, ensure_ascii=False)
            else:
                failed.append(article_id)
        
        suffix = "llm" if self.llm_cleaned else "basic"
        summary_file = self.output_dir / f"all_merged_{suffix}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*70)
        print("📊 MERGE SUMMARY")
        print("="*70)
        print(f"✅ Successfully merged: {len(merged_data)}/{len(entity_files)}")
        print(f"❌ Failed: {len(failed)}")
        
        if merged_data:
            avg_chars = sum(d['metadata']['char_count'] for d in merged_data) / len(merged_data)
            avg_entities = sum(d['metadata']['entity_count'] for d in merged_data) / len(merged_data)
            total_entities = sum(d['metadata']['entity_count'] for d in merged_data)
            
            print(f"📈 Average chars per article: {avg_chars:.0f}")
            print(f"🧬 Average entities per article: {avg_entities:.1f}")
            print(f"🧬 Total unique entities: {total_entities}")
        
        if failed:
            print(f"\n⚠️  Failed articles: {', '.join(failed)}")
        
        print(f"\n💾 Saved to: {summary_file}")
        print("="*70)
        
        return merged_data


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent 
    DATA_DIR = BASE_DIR / "data_processing" / "data"
    print("\n" + "="*70)
    print("MERGING: LLM-CLEANED TEXTS + ENTITIES")
    print("="*70)
    
    merger_llm = TextEntityMerger(
        texts_dir = DATA_DIR / "texts" / "texts_clean_w_llm",
        entities_dir = DATA_DIR / "entities",   
        output_dir = DATA_DIR / "merged" / "with_llm",
        llm_cleaned = True
    )
    
    results_llm = merger_llm.merge_all()
    
    print("\n" + "="*70)
    print("MERGING: BASIC-CLEANED TEXTS + ENTITIES")
    print("="*70)
    
    merger_basic = TextEntityMerger(
        texts_dir = DATA_DIR / "texts" / "texts_clean_no_llm",
        entities_dir = DATA_DIR / "entities",
        output_dir = DATA_DIR / "merged" / "no_llm",
        llm_cleaned = False
    )
    
    results_basic = merger_basic.merge_all()
    
    print("\n" + "="*70)
    print("✅ MERGE COMPLETE!")
    print("="*70)
    print(f"📊 LLM-cleaned: {len(results_llm)} articles")
    print(f"📊 Basic-cleaned: {len(results_basic)} articles")
