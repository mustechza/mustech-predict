import random
from collections import Counter
import requests
from bs4 import BeautifulSoup
import streamlit as st

# === Function to fetch latest UK49s results ===
def fetch_latest_results():
    url = 'https://www.uk49sresults.co.uk/'  # Example site
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    past_results = []
    draws = soup.select('.results .draw')[:10]  # Get last 10 draws

    for draw in draws:
        numbers = draw.select_one('.winning').text.strip().split()
        numbers = [int(n) for n in numbers if n.isdigit()]
        if len(numbers) >= 6:
            past_results.append(numbers[:6])

    return past_results


# === Fetch results with error handling ===
try:
    past_results = fetch_latest_results()
except Exception as e:
    st.error(f"Failed to fetch results online: {e}")
    # Fallback to static sample results
    past_results = [
        [5, 12, 23, 34, 45, 48],
        [1, 14, 22, 33, 39, 44],
        [7, 9, 16, 29, 36, 40],
        [3, 18, 21, 30, 42, 49],
        [6, 13, 27, 31, 38, 47],
        [2, 10, 20, 32, 41, 46],
        [4, 11, 19, 28, 35, 43],
        [8, 15, 17, 24, 25, 37],
    ]

# === Count frequency ===
all_numbers = [num for draw in past_results for num in draw]
number_counts = Counter(all_numbers)

# Sort numbers by frequency
hot_numbers = [num for num, count in number_counts.most_common()]
cold_numbers = [num for num, count in number_counts.most_common()][::-1]

print("Hot Numbers:", hot_numbers[:6])
print("Cold Numbers:", cold_numbers[:6])


# === Generate prediction ===
def generate_prediction():
    prediction = set()

    # Step 1: Pick 2 hot numbers
    prediction.update(random.sample(hot_numbers[:15], 2))

    # Step 2: Pick 2 cold numbers
    prediction.update(random.sample(cold_numbers[:20], 2))

    # Step 3: Fill the rest (ensuring odd/even balance and high/low balance)
    while len(prediction) < 6:
        candidate = random.randint(1, 49)

        temp = list(prediction) + [candidate]
        odd_count = sum(1 for n in temp if n % 2 != 0)
        even_count = sum(1 for n in temp if n % 2 == 0)

        low_count = sum(1 for n in temp if n <= 24)
        high_count = sum(1 for n in temp if n >= 25)

        # Keep balance 2-4 or 3-3 (odd-even and high-low)
        if odd_count <= 4 and even_count <= 4 and low_count <= 4 and high_count <= 4:
            prediction.add(candidate)

    return sorted(prediction)


# === Generate 3 predictions ===
print("\nPredicted Combinations:")
for _ in range(3):
    print(generate_prediction())
