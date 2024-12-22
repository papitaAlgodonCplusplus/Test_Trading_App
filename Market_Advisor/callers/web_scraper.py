from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import re

def indicate_investing(frequency="5m"):
    url = "https://www.investing.com/currencies/eur-usd-technical"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        })
        try:
            page.goto(url, timeout=60000, wait_until="domcontentloaded")
            page.wait_for_selector(f'button[data-test="{frequency}"]', timeout=10000)
            page.click(f'button[data-test="{frequency}"]')
            selector = 'div.mb-6.mt-1.rounded-full.px-4.py-1\\.5.text-center'
            page.wait_for_selector(selector, timeout=10000)
            soup = BeautifulSoup(page.content(), "html.parser")
            div_content = soup.find("div", class_="mb-6 mt-1 rounded-full px-4 py-1.5 text-center -mt-2.5 font-semibold leading-5 text-white bg-positive-main") or \
                          soup.find("div", class_="mb-6 mt-1 rounded-full px-4 py-1.5 text-center -mt-2.5 font-semibold leading-5 text-white bg-[#5B616E]") or \
                          soup.find("div", class_="mb-6 mt-1 rounded-full px-4 py-1.5 text-center -mt-2.5 font-semibold leading-5 text-white bg-negative-main")
            if div_content:
                print(f"Div content in Investing: {div_content.text.strip()}")
            else:
                print("Div not found")
                return 0
            if div_content.text.strip() in ["Sell", "Strong Sell"]:
                return -1
            elif div_content.text.strip() in ["Buy", "Strong Buy"]:
                return 1
            else:
                return 0
        except Exception as e:
            print(f"Error occurred: {e}")
            return 0
        finally:
            browser.close()
            
def indicate_tradingview(frequency="5m"):
    url = "https://www.tradingview.com/symbols/EURUSD/technicals/"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        })
        try:
            page.goto(url, timeout=60000, wait_until="domcontentloaded")
            page.wait_for_selector(f'button[id="{frequency}"]', timeout=10000)
            page.click(f'button[id="{frequency}"]')
            selector = 'div[class*="container-"][class*="container-large-"]'
            page.wait_for_selector(selector, timeout=10000)
            soup = BeautifulSoup(page.content(), "html.parser")
            div_content = soup.find("div", class_=re.compile(f"container-vLbFM67a.*container-"))
            if div_content:
                match = re.search(r'container-(\w+)-vLbFM67a', str(div_content))
                if match:
                    result = match.group(1).replace("-", " ").capitalize()
                    print(f"Div content from Trading View: {result}")
                    div_content = result
            else:
                print("Div not found")
                return 0
            if div_content in ["Sell", "Strong sell"]:
                return -1
            elif div_content in ["Buy", "Strong buy"]:
                return 1
            else:
                return 0
        except Exception as e:
            print(f"Error occurred: {e}")
            return 0
        finally:
            browser.close()

def indicate_tradersunion(freq="m5"):
    url = "https://tradersunion.com/currencies/forecast/eur-usd/signals/"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        })

        try:
            result = "None"
            page.goto(url, timeout=60000, wait_until="domcontentloaded")
            page.wait_for_load_state("networkidle")  # Ensure the page has fully loaded

            # Handle multiple matching elements
            span_elements = page.query_selector_all(f'div[data-timeframe="{freq}"] span')
            if span_elements:
                for element in span_elements:
                    if element.is_visible():
                        element.scroll_into_view_if_needed()
                        page.wait_for_timeout(1000)  # Small wait for visibility
                        element.click(force=True)
                        break
            else:
                print("No visible elements found for the given selector")
            
            # Fetch the status element
            element = page.query_selector('div.arrow p[class*="status"]')
            if element:
                result = element.inner_text().strip()
                print(f"Div content in TradersUnion: {result}")
            else:
                print("Div not found")

            if result in ["Sell", "Strong Sell"]:
                return -1
            elif result in ["Buy", "Strong Buy"]:
                return 1
            else:
                return 0

        except Exception as e:
            print(f"Error occurred: {e}")
            return 0

        finally:
            browser.close()
