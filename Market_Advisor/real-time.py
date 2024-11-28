from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import csv
from datetime import datetime

# Function to initialize the CSV file if it doesn't exist
def initialize_csv(file_name):
    try:
        with open(file_name, "x", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
    except FileExistsError:
        pass  # File already exists, no need to initialize

def fetch_data_with_chrome(file_name):
    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode (no GUI)
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Path to ChromeDriver
    service = Service("C:/Users/Alex/Desktop/forex/Deep-Reinforcement-Stock-Trading/Market_Advisor/chromedriver.exe")    
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        url = "https://es.tradingview.com/chart/?symbol=FX%3AEURUSD"
        driver.get(url)

        # Wait for the page to load (increase time if necessary)
        time.sleep(5)

        # Locate the div containing values
        values_wrapper = driver.find_element(By.CLASS_NAME, "valuesAdditionalWrapper-l31H9iuA")
        value_items = values_wrapper.find_elements(By.CLASS_NAME, "valueItem-l31H9iuA")

        # Initialize variables to store data
        data = {"Open": None, "High": None, "Low": None, "Close": None, "Volume": 0}  # Set Volume to 0 by default

        # Extract data
        for item in value_items:
            try:
                title = item.find_element(By.CLASS_NAME, "valueTitle-l31H9iuA").text.strip()
                value = item.find_element(By.CLASS_NAME, "valueValue-l31H9iuA").text.strip()
                print(f"{title}: {value}")

                # Replace , to . in value
                value = value.replace(",", ".")
                
                if title in ["O", "H", "L", "C"]:
                    key = {
                        "O": "Open",
                        "H": "High",
                        "L": "Low",
                        "C": "Close"
                    }[title]
                    data[key] = value
            except Exception as e:
                # Skip items that don't have the expected structure
                pass

        # Get the current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data["Date"] = current_time

        # Append data to the CSV file
        print("Data: ", data)
        with open(file_name, "a", newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)  # Prevent quoting around numeric values
            writer.writerow([data["Date"], data["Open"].replace(",", ""), data["High"].replace(",", ""), 
                            data["Low"].replace(",", ""), data["Close"].replace(",", ""), data["Volume"]])

        print(f"Data saved: {data}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()  # Ensure the browser is closed

def main():
    file_name = "real_time_data.csv"
    initialize_csv(file_name)

    while True:
        print("Fetching data...")
        fetch_data_with_chrome(file_name)
        print("Waiting for 1 minute...\n")
        time.sleep(60)

if __name__ == "__main__":
    main()
