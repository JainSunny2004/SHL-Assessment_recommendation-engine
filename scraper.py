import requests
import json
import os
import time
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin


class SHLScraper:
    def __init__(self):
        self.base_url = "https://www.shl.com"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.shl.com/products/product-catalog/",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.data = []
        self.checkpoint_file = "shl_scraper_checkpoint.json"
        self.processed_urls = set()  # Track URLs we've already processed
        
    def load_checkpoint(self):
        """Load checkpoint if it exists"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                self.data = checkpoint_data.get('data', [])
                # Rebuild processed URLs set from loaded data
                self.processed_urls = set(item['url'] for item in self.data if 'url' in item)
                print(f"Loaded checkpoint with {len(self.data)} products and {len(self.processed_urls)} processed URLs")
                return checkpoint_data.get('next_table', 1), checkpoint_data.get('next_page', 0)
        return 1, 0
        
    def save_checkpoint(self, table_num, page_num):
        """Save current progress to checkpoint file"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'data': self.data,
                'next_table': table_num,
                'next_page': page_num,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f)
        
    def save_data(self):
        """Save scraped data to JSON file"""
        with open('shl_products_data.json', 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"Data saved to shl_products_data.json ({len(self.data)} products)")
        
    def get_page(self, url):
        """Fetch page content with error handling and retry logic"""
        retries = 3
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Check if we got a valid page (not a redirect to login or error page)
                if "Please log in" in response.text or "Page not found" in response.text:
                    print(f"Invalid response from {url}: redirected to login or error page")
                    if attempt < retries - 1:
                        continue
                    else:
                        return None
                        
                return response.text
            except requests.exceptions.RequestException as e:
                print(f"Error fetching {url}: {e}")
                if attempt < retries - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Skipping this URL.")
                    return None
    
    def parse_test_type(self, product_keys):
        """Parse test type keys into their actual meanings"""
        key_mappings = {
            'A': 'Ability',
            'B': 'Behavior',
            'C': 'Cognitive',
            'P': 'Personality',
            'S': 'Situational',
            'K': 'Knowledge'
        }
        
        test_types = []
        for key in product_keys:
            if key in key_mappings:
                test_types.append(key_mappings[key])
        
        return test_types
                
    def parse_product_table(self, page_content, table_type):
        """Parse products from table on catalog page"""
        if not page_content:
            return []
            
        soup = BeautifulSoup(page_content, 'html.parser')
        products = []
        
        # Find all product rows in the table
        rows = soup.select('tr[data-course-id]')
        
        for row in rows:
            course_id = row.get('data-course-id')
            
            # Extract product name and URL
            name_cell = row.select_one('td.custom__table-heading__title a')
            if not name_cell:
                continue
                
            name = name_cell.text.strip()
            product_url = urljoin(self.base_url, name_cell.get('href', ''))
            
            # Extract remote testing support
            remote_cell = row.select_one('td.custom__table-heading__general:nth-of-type(2)')
            remote_support = bool(remote_cell and remote_cell.select_one('span.catalogue__circle.-yes'))
            
            # Extract adaptive/IRT support
            adaptive_cell = row.select_one('td.custom__table-heading__general:nth-of-type(3)')
            adaptive_support = bool(adaptive_cell and adaptive_cell.select_one('span.catalogue__circle.-yes'))
            
            # Extract test type keys
            test_type_cell = row.select_one('td.custom__table-heading__general.product-catalogue__keys')
            test_type_keys = []
            if test_type_cell:
                key_spans = test_type_cell.select('span.product-catalogue__key')
                test_type_keys = [span.text.strip() for span in key_spans]
            
            test_types = self.parse_test_type(test_type_keys)
            
            product = {
                'id': course_id,
                'name': name,
                'url': product_url,
                'remote_support': remote_support,
                'adaptive_irt': adaptive_support,
                'test_type_keys': test_type_keys,
                'test_types': test_types,
                'table_type': table_type,
                'description': None,
                'duration': None
            }
            
            products.append(product)
            
        return products
        
    def extract_product_details(self, product):
        """Visit product page and extract additional details"""
        print(f"Fetching details for: {product['name']}")
        
        page_content = self.get_page(product['url'])
        if not page_content:
            return product
            
        soup = BeautifulSoup(page_content, 'html.parser')
        
        # Extract description
        description_div = soup.select_one('div.product-catalogue-training-calendar__row.typ p')
        if description_div:
            product['description'] = description_div.text.strip()
        else:
            # Try alternate selectors for description
            alt_description = soup.select_one('div.product-catalogue-training-calendar__row p')
            if alt_description:
                product['description'] = alt_description.text.strip()
            
        # Extract duration
        duration_elem = soup.select_one('div.product-catalogue-training-calendar__row.typ h4:-soup-contains("Assessment length") + p')
        if duration_elem:
            duration_text = duration_elem.text.strip()
            # Try to extract the numeric duration if available
            duration_match = re.search(r'(\d+)', duration_text)
            if duration_match:
                product['duration'] = int(duration_match.group(1))
            else:
                product['duration'] = duration_text
        else:
            # Try alternate selectors for duration
            alt_duration = soup.select_one('div.product-catalogue-training-calendar__row h4:-soup-contains("Assessment length") + p')
            if alt_duration:
                duration_text = alt_duration.text.strip()
                duration_match = re.search(r'(\d+)', duration_text)
                if duration_match:
                    product['duration'] = int(duration_match.group(1))
                else:
                    product['duration'] = duration_text
                
        return product
    
    def try_url_variations(self, table_type, page_num):
        """Try different URL variations to find one that works"""
        url_variations = [
            # Standard format
            f"https://www.shl.com/products/product-catalog/?start={page_num}&type={table_type}",
            # Double type parameter
            f"https://www.shl.com/products/product-catalog/?start={page_num}&type={table_type}&type={table_type}",
            # Mixed type parameters
            f"https://www.shl.com/products/product-catalog/?start={page_num}&type={table_type}&type={3-table_type}"
        ]
        
        for url in url_variations:
            print(f"Trying URL: {url}")
            content = self.get_page(url)
            if content:
                soup = BeautifulSoup(content, 'html.parser')
                # Check if we have product rows
                if soup.select('tr[data-course-id]'):
                    print(f"Found valid content with URL: {url}")
                    return content
                else:
                    print(f"No product rows found with URL: {url}")
        
        print("All URL variations failed")
        return None
        
    def scrape_catalog(self):
        """Main scraping function"""
        # Load checkpoint if it exists
        next_table, next_page = self.load_checkpoint()
        
        # Define the tables to scrape
        tables = [
            {
                'pages': 32,
                'type': 1
            },
            {
                'pages': 12,
                'type': 2
            }
        ]
        
        # Start from checkpoint or beginning
        for table_idx in range(next_table-1, len(tables)):
            table = tables[table_idx]
            start_page = next_page if table_idx == next_table-1 else 0
            
            for page in range(start_page, table['pages']):
                page_offset = page * 12  # 12 items per page
                print(f"\nScraping table {table_idx+1}, page {page+1}/{table['pages']}")
                
                # Try different URL variations until we find one that works
                page_content = self.try_url_variations(table['type'], page_offset)
                if not page_content:
                    print(f"Skipping page {page+1} - could not get valid content")
                    continue
                    
                # Parse the product table
                products = self.parse_product_table(page_content, table['type'])
                print(f"Found {len(products)} products on this page")
                
                # Get details for each product
                for product in products:
                    try:
                        # Check if we've already processed this URL
                        if product['url'] in self.processed_urls:
                            print(f"Skipping already processed product: {product['name']}")
                            continue
                            
                        product = self.extract_product_details(product)
                        self.data.append(product)
                        self.processed_urls.add(product['url'])
                        
                        # Sleep briefly to avoid overloading the server
                        time.sleep(1)
                    except Exception as e:
                        print(f"Error processing product {product['name']}: {e}")
                
                # Save checkpoint after each page
                self.save_checkpoint(table_idx+1, page+1)
                print(f"Checkpoint saved. Products collected so far: {len(self.data)}")
                
                # Save data periodically
                if page % 5 == 0:
                    self.save_data()
                
                # Be nice to the server
                time.sleep(2)
        
        # Save final data
        self.save_data()
        
        # Remove checkpoint file after successful completion
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            print("Scraping completed successfully. Checkpoint file removed.")


if __name__ == "__main__":
    scraper = SHLScraper()
    scraper.scrape_catalog()