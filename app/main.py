import cv2
import numpy as np

# Ścieżki do plików
SCREEN_PATH = 'screenshot2.png'
TEMPLATE_PATH = 'claim_template.png'
OUTPUT_PATH = 'wynik.png'

# Próg pewności (0.8 = 80% podobieństwa)
THRESHOLD = 0.8

# 1. Wczytanie obrazów
screen = cv2.imread(SCREEN_PATH)
template = cv2.imread(TEMPLATE_PATH)

if screen is None or template is None:
    raise FileNotFoundError("Błąd wczytywania: upewnij się, że pliki istnieją w podanej ścieżce.")

# 2. Pobranie wymiarów wzorca (wysokość, szerokość)
template_height, template_width = template.shape[:2]

# 3. Wyszukiwanie wzorca na screenie (normalizowana korelacja krzyżowa)
result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# 4. Sprawdzenie progu i zamalowanie
if max_val >= THRESHOLD:
    top_left = max_loc
    bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

    # OpenCV używa formatu BGR: zielony to (0, 255, 0)
    # thickness=-1 wypełnia cały prostokąt kolorem
    cv2.rectangle(screen, top_left, bottom_right, (0, 255, 0), thickness=-1)

    # 5. Zapis wyniku
    cv2.imwrite(OUTPUT_PATH, screen)
    print(f"Trafienie! Dopasowanie: {max_val * 100:.1f}%. Zapisano do: {OUTPUT_PATH}")
else:
    print(f"Brak dopasowania. Najwyższy wynik to {max_val * 100:.1f}%, a wymagany próg to {THRESHOLD * 100:.1f}%.")