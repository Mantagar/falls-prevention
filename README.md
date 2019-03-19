# Przetwarzanie danych
Zmień nazwę katalogu, z oryginalnymi matlabowymi danymi (z subfolderem "No finding") na "Mat"
```
mkdir Processed\ data/Synkope -f
mkdir Processed\ data/Nosynkope
python mat_to_csv.py
```
To chwilę potrwa. Komunikaty na ekranie pokazują, którzy pacjenci zostali odrzuceni lub gdzie zostali zapisani.

# Wizualizacja przeprocesowanych danych
```
python preview.py
```

# Podział przeprocesowanych danych na zbiory
```
python divide.py 70 15 15
```
Spowoduje to utworzenie trzech plików w "Processed data" z ścieżkami do danych.
Następuje tutaj balansowanie danych, aby je wyłączyć, należy usunąć linijkę z divide.py (towarzyszy jej odpowiedni komentarz).

# Uruchamianie wsadowych zadań na Prometeuszu
```
sbatch template.sh
```
Szablon pokazuje jak powinien wyglądać skrypt wsadowy.