# Pandas importieren
import pandas as pd

# Dateipfad 
Datei_Pfad = r"C:\Users\MarcH\Videos\yelp_academic_dataset_business.json"
Datei_Ausgabe = r"C:\Users\MarcH\Videos\missing_values.csv"

# Speicherung der Zeilen mit fehlenden Werten
missing_values = []

# Datei Zeilenweise lesen
try:
    with open(Datei_Pfad, "r", encoding="utf-8") as file:
        for line in file:
            try:
                # JSON-Zeile in Dictionary umwandeln
                business = eval(line.strip())  
                
                # Prüfe, ob eine der Spalten leer oder None ist
                for key, value in business.items():
                    if value is None or value == "":
                        missing_values.append(business) 
                        break  # Sobald ein fehlender Wert gefunden wurde, breche die Schleife für diese Zeile ab

# Falls fehlende Werte existieren, speichere sie in einer CSV-Datei
    if missing_values:
        df_missing = pd.DataFrame(missing_values)
        df_missing.to_csv(Datei_Ausgabe, index=False, encoding="utf-8")
        print("Missings gespeichert")
    else:
        print("Keine fehlenden Werte gefunden!")