import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import pdfplumber
from tqdm import tqdm
import requests
import time
from dotenv import load_dotenv
load_dotenv()


class PDFTextExtractor:
    """
    Извлекает текст из PDF файлов и опционально очищает через LLM
    """
    
    def __init__(
        self, 
        pdf_dir: str = "data/pdfs",
        output_dir: str = "data/extracted_texts",
        min_text_length: int = 100,
        use_llm_cleaning: bool = True,
        openrouter_api_key: Optional[str] = os.getenv('OPENROUTER_API_KEY')
    ):
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.min_text_length = min_text_length
        self.use_llm_cleaning = use_llm_cleaning
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        
        if self.use_llm_cleaning and not self.openrouter_api_key:
            raise ValueError("OpenRouter API key required for LLM cleaning")

    def split_into_chunks(self, text: str, chunk_size: int = 8000) -> List[str]:
        """
        Делит текст на чанки по chunk_size символов
        
        Args:
            text: исходный текст
            chunk_size: размер чанка в символах
            
        Returns:
            list: список чанков
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks

    def clean_with_llm(self, text: str) -> str:
        """
        Очищает текст через LLM (разбивает на чанки по 8000 символов)
        """
        chunks = self.split_into_chunks(text, chunk_size=8000)
        
        if len(chunks) == 1:
            return self._clean_chunk_with_llm(chunks[0])
        
        cleaned_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"  🤖 LLM cleaning chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            cleaned = self._clean_chunk_with_llm(chunk)
            cleaned_chunks.append(cleaned)
            
            if i < len(chunks) - 1:
                time.sleep(5)
        
        return ' '.join(cleaned_chunks)
    
    def _clean_chunk_with_llm(self, text: str) -> str:
        """
        Очищает один чанк текста через OpenRouter API
        """
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        prompt = f"""You are a specialized scientific text editor.
        Task: Restore readability of the provided text chunk from a PDF.

        STRICT RULES:
        1. Fix broken hyphenation (e.g., "Alz- heimer" -> "Alzheimer").
        2. Remove random line breaks within sentences.
        3. Fix OCR errors but DO NOT change chemical formulas or protein names (e.g., "BACE1" must stay "BACE1").
        4. DO NOT summarize or rewrite the content. Preserver original wording.
        5. If you see a table or data that is impossible to format as text, keep it as is or format as a Markdown table.
        6. Return ONLY the raw cleaned text. No "Here is the output" preamble.

        Text to clean:
        {text}"""
        
        payload = {
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.0,
            "max_tokens": 4000
        }
        
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            cleaned_text = result['choices'][0]['message']['content'].strip()
            return cleaned_text
            
        except Exception as e:
            print(f"  ⚠️  LLM cleaning failed: {str(e)[:100]}")
            return text  
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Optional[Dict[str, str]]:
        """
        Извлекает текст из одного PDF файла
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = []
                
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text.append(text)
                
                combined_text = "\n\n".join(full_text)
    
                combined_text = self._basic_clean(combined_text)
                
                if len(combined_text) < self.min_text_length:
                    print(f"⚠️  {pdf_path.stem}: text too short ({len(combined_text)} chars)")
                    return None
        
                if self.use_llm_cleaning:
                    print(f"🤖 LLM cleaning: {pdf_path.stem}")
                    combined_text = self.clean_with_llm(combined_text)
                
                return {
                    'filename': pdf_path.stem,
                    'full_text': combined_text,
                    'page_count': len(pdf.pages),
                    'char_count': len(combined_text),
                    'llm_cleaned': self.use_llm_cleaning
                }
                
        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {str(e)}")
            return None
    
    def _basic_clean(self, text: str) -> str:
        """
        Базовая очистка текста (без LLM)
        """
        lines = text.split('\n')
        cleaned_lines = [' '.join(line.split()) for line in lines if line.strip()]
        text = '\n'.join(cleaned_lines)
        return text.strip()
    
    def extract_all_pdfs(self, save_individual: bool = True) -> List[Dict]:
        """
        Извлекает текст из всех PDF файлов
        """
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ No PDF files found in {self.pdf_dir}")
            return []
        
        print(f"\n📄 Found {len(pdf_files)} PDF files")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🤖 LLM cleaning: {'ENABLED' if self.use_llm_cleaning else 'DISABLED'}\n")
        
        extracted_data = []
        failed_files = []
        
        for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
            result = self.extract_text_from_pdf(pdf_path)
            
            if result:
                extracted_data.append(result)
                
                if save_individual:
                    suffix = "_cleaned" if self.use_llm_cleaning else "_text"
                    output_file = self.output_dir / f"{result['filename']}{suffix}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                failed_files.append(pdf_path.name)
        
        summary_filename = "all_texts_cleaned.json" if self.use_llm_cleaning else "all_texts.json"
        summary_file = self.output_dir / summary_filename
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        
        # Статистика
        print("\n" + "="*70)
        print("📊 EXTRACTION SUMMARY")
        print("="*70)
        print(f"✅ Successfully extracted: {len(extracted_data)}/{len(pdf_files)}")
        print(f"❌ Failed: {len(failed_files)}")
        
        if extracted_data:
            avg_chars = sum(d['char_count'] for d in extracted_data) / len(extracted_data)
            avg_pages = sum(d['page_count'] for d in extracted_data) / len(extracted_data)
            total_chars = sum(d['char_count'] for d in extracted_data)
            print(f"📈 Average characters per document: {avg_chars:.0f}")
            print(f"📄 Average pages per document: {avg_pages:.1f}")
            print(f"📚 Total characters extracted: {total_chars:,}")
        
        if failed_files:
            print(f"\n⚠️  Failed files: {', '.join(failed_files)}")
        
        print(f"\n💾 Saved to: {summary_file}")
        print("="*70)
        
        return extracted_data


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent 
    DATA_DIR = BASE_DIR / "data_processing" / "data"
    # Вариант 1: БЕЗ LLM (быстро)
    # extractor = PDFTextExtractor(
    #     pdf_dir=r"C:\Users\User\Desktop\BIOCAD\dataprep\data\pdfs_clean",
    #     output_dir=r"C:\Users\User\Desktop\BIOCAD\dataprep\data\texts",
    #     use_llm_cleaning=False
    # )
    
    extractor = PDFTextExtractor(
        pdf_dir = DATA_DIR / "pdfs_clean",
        output_dir = DATA_DIR / "texts",
        use_llm_cleaning = True,
        openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    )
    
    results = extractor.extract_all_pdfs(save_individual=True)
    
    if results:
        print("\n📝 Sample from first document:")
        print(f"Filename: {results[0]['filename']}")
        print(f"First 500 chars:\n{results[0]['full_text'][:500]}...")