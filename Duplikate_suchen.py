# Pandas importieren 
import pandas as pd

# Dateipfad 
Datei_Pfad = r"M:\Data\yelp_academic_dataset_business.json"
Datei_Ausgabe = r"M:\Data\duplicates_found.csv"

# Speicherung von IDs & Duplikaten
unique = set()
duplicates = []

# Datei Zeilenweise lesen
try: 
    with open(Datei_Pfad, "r", encoding="utf-8") as file: 
        for line in file: 
            try: 
                # JSON-Zeile in Dictionary umwandeln 
                business = eval(line.strip())  
                business_id = business.get("business_id")  

                # Basierend auf business_id nach Duplikaten Prüfen (zuerst falls Duplikat speichern, als zweites die Prüfung bei neu Speichern.)
                if business_id:
                    if business_id in unique:
                        duplicates.append(business) 
                    else: 
                        unique.add(business_id)


# Duplikate in separater Datei speichern
    if duplicates:
        df_duplicates = pd.DataFrame(duplicates)
        df_duplicates.to_csv(Datei_Ausgabe, index=False, encoding="utf-8")  
        print("Duplikate gespeichert")
    else:
        print("Keine Duplikate gefunden")


