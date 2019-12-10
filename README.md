# PhD_project_1

This repository contains all the coding docs required to de the data setup and analysis of the first project of my PhD project. 

Up-to-date content:
1. Data_setup_wp1.py				Reads all the necessary data from the call reports and combines variables. Output: RAW_DATA
2. Add_vars_to_df.py				Adds variables from the summary of deposits (uses a.) to the RAW_DATA and removes all non_uS, non-commercial banks
3. Clean_data_wp1_v2.py				Does some further data cleaning based on plots and summary statistics. Output: CLEAN_DATA
4. Table_num_sec-nonsec.py			Input: CLEAN_DATA. Output: Table about the structure of the data
5. tables_descriptives_wp_v3.py		Input: CLEAN_DATA. Output: Summary statistics tables of all the samples
6. corr_matrix.py					Input: CLEAN_DATA. Output: Correlation matrices of all the samples
7. FD_baseline.py					Input: CLEAN_DATA. Output: Excel with the estimates of a FD baseline model for all samples and dependent variables
8. FDIV_v4.py						Input: CLEAN_DATA. Output: Excel with the estimates of a FDIV model for all samples and dependent variables, includes tests
9. Figures_wp1.py					Input: CLEAN_DATA. Output: Plots with figures used in the paper
10. FDIV_alt.py						Alternative spcification of FDIV_v4

Other content needed (from the help_function folder):
a. Proxies_org_complex_banks.py 	Is needed in Add_vars_to_df.py
