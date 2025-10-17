import os
from rdkit import Chem

import os
from rdkit import Chem

# path to InChI and property file
input_path = r"C:\Users\cathe\OneDrive\Documents\MChem_Project\InChI_Prop\InChI_BP.tsv"

# create output file name with _SMILES
output_path = r"C:\Users\cathe\OneDrive\Documents\MChem_Project\Descriptors_taken_from_datasets\BP_InChI_SMILES_fixed.txt"

print(f"Now converting {input_path} to SMILES")

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    header = next(infile)
    ## start is 2 as header is line 1
    for line_num, line in enumerate(infile, start=2):
        parts = line.strip().split("\t")  ##split columns
        if len(parts) < 1:
            print(f"Skipped line {line_num} as it was blank or malformed")
            continue

        ##first column is InChI    
        inchi = parts[0] 

        if not inchi:
            print(f"Skipped line {line_num} as InChI was blank")
            continue

        try:
            mol = Chem.MolFromInchi(inchi)
            if mol:
                smiles = Chem.MolToSmiles(mol)
                outfile.write(smiles + "\n")
            else:
                print(f"Invalid InChI at line {line_num}, skipped.")
        except Exception as e:
            print(f"Error on line {line_num}: {e}")

print(f"Saved SMILES to {output_path}")

# #list all txt files to ensure they exist there
# for filename in os.listdir(input_folder):
#     if filename.endswith(".txt"):
#         input_path = os.path.join(input_folder, filename)

#         #create file name with SMILES on end
#         root = os.path.splitext(filename)[0]
#         output_filename = f"{root}_SMILES.txt"
#         output_path = os.path.join(input_folder, output_filename)

#         print(f" Now converting {filename} to SMILES")

#         with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
#             for line_num, line in enumerate(infile, start=1):
#                 inchi = line.strip()

#                 if not inchi:
#                     print(f"Skipped line {line_num} as it was blank")
#                     continue  ## in case theres blank lines, skip

#                 try:
#                     mol = Chem.MolFromInchi(inchi)
#                     if mol:
#                         smiles = Chem.MolToSmiles(mol)
#                         outfile.write(smiles + "\n")
#                     else:
#                         outfile.write(f"sorry line {line_num} had Invalid InChI\n")
#                         print("check file, invalid InChI")
#                 except Exception as e:
#                     outfile.write(f"line {line_num} had an error: {e}\n")

#         print(f" Saved to {output_filename}\n")
