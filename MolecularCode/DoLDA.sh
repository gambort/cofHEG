Opts="Basis=cc-pvdz Pre=10. Ignore"

for Sys in glyoxal tetrazine benzoquinone; do
    ./LDA_Spectra.py ./Mols/${Sys}.xyz NO ${Opts}  > LDA_Outputs/${Sys}_NO_ELDA.out
done

#cp ./LDA_Outputs/*.out /mnt/c/Collabs/Ensemble/HEG-EDFA/HEG/NVCode/LDA_Outputs/
