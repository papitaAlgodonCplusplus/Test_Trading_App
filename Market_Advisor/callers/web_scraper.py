from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import re

def indicate_investing(frequency="5m"):
    url = "https://www.investing.com/currencies/eur-usd-technical"
    print(f"Fetching data for frequency {frequency} from Investing")
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
    frequency = "4h" if frequency == "5h" else frequency
    url = "https://www.tradingview.com/symbols/EURUSD/technicals/"
    print(f"Fetching data for {frequency} timeframe")
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
            div_content = soup.find("div", class_=re.compile(r"container-vLbFM67a.*"))
            if div_content:
                match = re.search(r'container-(\w+)-vLbFM67a', ' '.join(div_content['class']))
                if match:
                    result = match.group(1).replace("-", " ").capitalize()
                    print(f"Div content from Trading View: {result}")
                    div_content = result
            else:
                print("Div not found from Trading View")
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
    freq = "30m" if freq == "m15" else freq == "d1" if freq == "h5" else freq
    url = "https://tradersunion.com/currencies/forecast/eur-usd/signals/"
    print(f"Fetching data for {freq} from TradersUnion")
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
            soup = BeautifulSoup(page.content(), "html.parser")
            element = soup.find("a", href="/brokers/forex/redirect/roboforex/")
            if element:
                result = element.text.strip()
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

def indicate_fxstreet():
    url = "https://www.fxstreet.com/rates-charts/eurusd"
    print("Fetching data from FXStreet")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        })

        try:
            page.goto(url, timeout=60000, wait_until="domcontentloaded")
            page.wait_for_timeout(8000) 
            soup = BeautifulSoup(page.content(), "html.parser")
            div_content = soup.find("span", class_=re.compile(r"fxs_index_value fxs_txt_.*"))
            if div_content:
                result = div_content.text.strip()
                print(f"Div content from FXStreet: {result}")

                if "Bearish" in result:
                    return -1
                elif "Bullish" in result:
                    return 1
                else:
                    return 0
            else:
                print("Div not found from FXStreet")
                return 0
        
        except Exception as e:
            print(f"Error occurred: {e}")
            return 0
        
        finally:
            browser.close()

def indicate_fxleaders(frequency="5m"):
    url = "https://www.fxleaders.com/live-rates/eur-usd/"
    print(f"Fetching data for {frequency} timeframe from FXLeaders")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        })

        try:
            page.goto(url, timeout=60000, wait_until="domcontentloaded")
            
            # Determine the button text based on frequency
            button_text = ""
            if frequency == "5m":
                button_text = "5 min"
            elif frequency == "15m":
                button_text = "15 min"
            elif frequency == "30m":
                button_text = "30 min"
            else:
                button_text = "Hourly"

            font_selector = f"button:has-text('{button_text}')"
            page.wait_for_selector(font_selector, timeout=10000)
            page.click(font_selector)

            # Parse the page content
            page.wait_for_timeout(3000)  # Wait for 3 seconds to ensure the page loads completely
            soup = BeautifulSoup(page.content(), "html.parser")
            font_content = soup.find(
                "div",
                class_=re.compile(r"font-bold text-(red|green) ng-scope")
            )
            if font_content:
                result = font_content.text.strip()
                print(f"Font content from FXLeaders: {result}")

                if "Buy" in result:
                    return 1
                elif "Sell" in result:
                    return -1
                else:
                    return 0
            else:
                print("Font content not found from FXLeaders")
                return 0

        except Exception as e:
            print(f"Error occurred: {e}")
            return 0

        finally:
            browser.close()
