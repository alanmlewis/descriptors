from bs4 import BeautifulSoup
import os 
import csv

## Files
Xdml_Files = [r"C:\Users\cathe\OneDrive\Documents\MChem_Project\compounds_HTMLs\compounds_enthalpy_of_vapourization.xml",r"C:\Users\cathe\OneDrive\Documents\MChem_Project\compounds_HTMLs\compounds_Boiling_Point.xml",
              r"C:\Users\cathe\OneDrive\Documents\MChem_Project\compounds_HTMLs\compounds_flash_point.xml",r"C:\Users\cathe\OneDrive\Documents\MChem_Project\compounds_HTMLs\compounds_lower_flammabiltiy.xml",
              r"C:\Users\cathe\OneDrive\Documents\MChem_Project\compounds_HTMLs\compounds_viscosity_as_LogVis.xml"]

##Xdml_Files = []

for file in Xdml_Files: 
    root_only = os.path.splitext(file)[0] ## only want the root, changing the extentsion
    output_file = f'{root_only}_InChI_Id.txt'
    ## end file

    with open(file, 'r', encoding="utf-8") as f:
        soup = BeautifulSoup(f, "xml")
        finding_InChI = soup.find_all('InChI')
        finding_Compound_ID = soup.find_all('Id')

    with open (output_file, 'w', encoding='utf-8') as output:
        ## added header
        output.write("InChI\tCompound_ID\n")

        for inchi, compound_id in zip(finding_InChI, finding_Compound_ID):
            InChI_as_text = inchi.get_text(strip=True)
            Compound_Id_as_text = compound_id.get_text(strip=True)
            output.write(f"{InChI_as_text}\t{Compound_Id_as_text}\n")

print('All files saved in Folder')        

## function to extract compound ID and InChI. Match it to BP and compound ID doc. then exclude the InChI correspodning to those ocmpound IDs.
# make one document with all? mismatch though 

## Files
file_a = r"C:\Users\cathe\OneDrive\Documents\MChem_Project\InChI_Id\compounds_Boiling_Point_InChI_Id.txt"
file_b = r"C:\Users\cathe\OneDrive\Documents\MChem_Project\Just_Properties_with_ID\values_boiling_point"    
output_InChI_BP = r"C:\Users\cathe\OneDrive\Documents\MChem_Project\InChI_Prop\InChI_BP.tsv"

##check the headers
with open(file_a, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    print("Headers in file_a:", reader.fieldnames)

with open(file_b, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    print("Headers in file_b:", reader.fieldnames)

##compound Id to InChI
id_to_inchi = {}
with open(file_a, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        compound_id = row["Compound_ID"]
        inchi = row["InChI"]
        id_to_inchi[compound_id] = inchi

## InChI to BP
with open(file_b, "r", encoding="utf-8") as f_in, open(output_InChI_BP, "w", encoding="utf-8", newline="") as f_out:
    reader = csv.DictReader(f_in, delimiter="\t")
    writer = csv.writer(f_out, delimiter="\t")
    
    ## headers
    writer.writerow(["InChI", "Boiling_Point"])
    
    for row in reader:
        compound_id = row["Compound Id"]
        bp = row["Normal boiling point"].strip()
        
        if compound_id in id_to_inchi and bp:  # only if InChI exists and BP is not empty
            writer.writerow([id_to_inchi[compound_id], bp])

print(f'InChI BP file saved to {output_InChI_BP}')
        