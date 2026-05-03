import time
import pandas as pd
from datetime import datetime

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By

# =========================
# CONFIG
# =========================
URL = "https://bc.game/game/crash"
SAVE_FILE = "crash_data.csv"

# =========================
# INIT DRIVER
# =========================
options = uc.ChromeOptions()
options.add_argument("--disable-blink-features=AutomationControlled")

driver = uc.Chrome(options=options)
driver.get(URL)

time.sleep(10)  # allow full load

# =========================
# STORAGE
# =========================
data = []

def save_data(row):
    df = pd.DataFrame([row])
    df.to_csv(SAVE_FILE, mode='a', header=not pd.io.common.file_exists(SAVE_FILE), index=False)

# =========================
# COLLECTOR LOOP
# =========================
print("🚀 Collecting crash data...")

seen = set()

while True:
    try:
        # Find crash values (update selector if UI changes)
        elements = driver.find_elements(By.CSS_SELECTOR, "tbody tr td")

        for el in elements:
            value = el.text.replace("x", "").strip()

            if value and value not in seen:
                seen.add(value)

                row = {
                    "multiplier": float(value),
                    "timestamp": datetime.now()
                }

                print(row)

                data.append(row)
                save_data(row)

        time.sleep(2)

    except Exception as e:
        print("Error:", e)
        time.sleep(5)
