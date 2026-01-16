import requests
import json
from pathlib import Path
import time
import re
from tqdm import tqdm

class PubMedCollector:
    def __init__(self, save_dir="data\alzheimer_papers"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.search_api_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        self.oa_api_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
        self.metadata_file = self.save_dir / "metadata.json"
        self.papers_data = []
    
    def search_papers_by_keywords(self, keywords, max_pages=5, page_size=100, sleep_time=1, use_oa_filter=True, min_relevance=0.0, verbose=True):
        
        """
        Args:
            keywords: list of keywords
            max_pages: maximum result pages
            page_size: results per page
            sleep_time: pause between requests
            use_oa_filter: using "open access[filter]"
            min_relevance: minimal threshold of relevane (0-1)
            verbose: output detailed information
        
        Returns:
            list: list of articles ith PMCID
        """

        collected = {}
        keywords_lower = [k.lower() for k in keywords]
        
        for keyword_idx, keyword in enumerate(keywords, 1):
            if verbose:
                print(f"\n[{keyword_idx}/{len(keywords)}] Searching: {keyword}")
            
            for page in range(1, max_pages + 1):
                if use_oa_filter:
                    query = f'{keyword} open access[filter]'
                else:
                    query = keyword
                
                params = {
                    "query": query,
                    "pageSize": min(page_size, 1000),
                    "pageNumber": page,
                    "format": "json",
                    "sortBy": "RELEVANCE",  
                    "resultType": "core"
                }
                
                try:
                    response = requests.get(self.search_api_url, params=params, timeout=20)
                    response.raise_for_status()
                except requests.RequestException as e:
                    if verbose:
                        print(f"Error on page {page}: {e}")
                    break
                
                data = response.json()
                papers = data.get('resultList', {}).get('result', [])
                
                if not papers:
                    if verbose:
                        print("No more results")
                    break
                
                page_matched = 0
                
                for paper in papers:
                    pmcid = paper.get('pmcid')
                    abstract = (paper.get('abstractText') or "").lower()
                    title = (paper.get('title') or "").lower()
                    
                    if not pmcid:
                        continue
                    
                    if not abstract:
                        continue
                    
                    matched_keywords = []
                    for k in keywords_lower:
                        words = k.split()  
                        text = abstract + " " + title  
                        core_words = [w for w in words if w not in ['and', 'the', 'for', 'with', 'from']]
                        if any(word in text for word in core_words):
                            matched_keywords.append(k)
                    
                    if not matched_keywords:
                        continue
                    
                    relevance_score = len(matched_keywords) / len(keywords_lower)
                    if relevance_score < min_relevance:
                        continue
                    
                    if pmcid not in collected:
                        paper['matched_keywords'] = matched_keywords
                        paper['relevance_score'] = relevance_score
                        collected[pmcid] = paper
                        page_matched += 1
                
                if verbose:
                    print(f"Page {page}: {len(papers)} results, matched: {page_matched}, total: {len(collected)}")
                
                time.sleep(sleep_time)
        
        self.papers_data = list(collected.values())
        
        if verbose:
            print("\nSearch summary:")
            print(f"Total papers with PMCID: {len(self.papers_data)}")
            if self.papers_data:
                avg_relevance = sum(p.get('relevance_score', 0) for p in self.papers_data) / len(self.papers_data)
                print(f"Average relevance: {avg_relevance:.2f}")
        
        return self.papers_data

    def get_oa_resources(self, pmcid):

        """
        Get information about OA resources through PMC OA API
        
        Args:
            pmcid: PubMed Central ID (without PMC prefix)
        
        Returns:
            dict: {'pdf_url': str, 'tgz_url': str, 'license': str, 'is_oa': bool}
        """

        try:
            params = {'id': f'PMC{pmcid}'}
            response = requests.get(self.oa_api_url, params=params, timeout=15)
            response.raise_for_status()
            
            xml_text = response.text
            
            resources = {
                'pdf_url': None,
                'tgz_url': None,
                'license': None,
                'is_oa': False
            }
            
            pdf_match = re.search(r'<link format="pdf"[^>]*href="([^"]+)"', xml_text)
            if pdf_match:
                resources['pdf_url'] = pdf_match.group(1)
                resources['is_oa'] = True

            tgz_match = re.search(r'<link format="tgz"[^>]*href="([^"]+)"', xml_text)
            if tgz_match:
                resources['tgz_url'] = tgz_match.group(1)
            
            license_match = re.search(r'license="([^"]+)"', xml_text)
            if license_match:
                resources['license'] = license_match.group(1)
            
            return resources

        except Exception as e:
            print(f"  Warning: Error getting OA resources for PMC{pmcid}: {str(e)[:50]}")
            return {
                'pdf_url': None,
                'tgz_url': None,
                'license': None,
                'is_oa': False
            }

    def save_metadata(self):

        """
        Save metadata to JSON with OA resource information
        
        Enriches each article with PDF/TGZ links and license information
        """
        
        unique_papers = {}
        oa_count = 0
        
        print("\nEnriching metadata with OA resource information...")
        
        for paper in tqdm(self.papers_data):
            pmcid = paper.get('pmcid')
            if pmcid and pmcid not in unique_papers:
                oa_resources = self.get_oa_resources(pmcid)
                paper['oa_resources'] = oa_resources
                
                if oa_resources['is_oa']:
                    oa_count += 1
                    license_info = oa_resources.get('license', 'unknown')
                    print(f"PMC{pmcid}: OA {license_info}")
                
                unique_papers[pmcid] = paper
        
        self.papers_data = list(unique_papers.values())
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.papers_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nMetadata saved: {self.metadata_file}")
        print(f"Total papers: {len(self.papers_data)}")
        print(f"Papers with OA PDFs: {oa_count}")

if __name__ == "__main__":
    collector = PubMedCollector("alzheimer_papers")
    
    keywords = [
        "Alzheimer's disease targets",
        "Alzheimer therapeutic targets",
        "Alzheimer drug targets"
    ]
    
    print("="*70)
    print("STEP 1: SEARCHING PAPERS (Europe PubMed Central API)")
    print("="*70)
    collector.search_papers_by_keywords(keywords, max_pages=2, use_oa_filter=False)
    
    print("\n" + "="*70)
    print("STEP 2: ENRICHING METADATA (PMC OA API)")
    print("="*70)
    collector.save_metadata()
        
    print("\n" + "="*70)
    print("✅ METADATA GENERATION COMPLETE!")
    print("="*70)