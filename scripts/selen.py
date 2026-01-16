import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

pmcids_file = 'alzheimer_papers/pmcids.txt'  # Input file with PMCIDs, one per line
download_dir = os.path.abspath('data/pdfs')  
os.makedirs(download_dir, exist_ok=True)

with open(pmcids_file, 'r', encoding='utf-8') as f:
    pmcids = [line.strip() for line in f if line.strip()]

chrome_options = Options()
chrome_options.add_experimental_option('prefs', {
    'download.default_directory': download_dir,
    'download.prompt_for_download': False,
    'download.directory_upgrade': True,
    'plugins.always_open_pdf_externally': True  
})

driver = webdriver.Chrome(options=chrome_options)  # Or: webdriver.Chrome(executable_path='/path/to/chromedriver', options=chrome_options)

try:
    for idx, pmcid in enumerate(pmcids, start=1):
        article_url = f'https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf'
        print(f'Processing {idx}/{len(pmcids)}: {pmcid}: navigating to {article_url}')
        
        driver.get(article_url)
        
        try:
            pdf_link = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.int-view[href$="/pdf/"]')) 
            )
            pdf_link.click()
            time.sleep(5)  

            print(f'Download initiated for {pmcid}')
        
        except Exception as e:
            print(f'Error processing {pmcid}: {str(e)}')
    
finally:
    driver.quit()
    print('All processing complete. PDFs saved in:', download_dir)