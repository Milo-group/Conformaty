import re
import pandas as pd
import numpy as np
import os
from enum import Enum
from typing import List, Optional
import re
import pandas as pd

class ReExpressions(Enum):
    FLOAT = r'[-+]?[0-9]+\.[0-9]+'
    FLOAT_POL= r'-?\d*\.\d*'
    FLOATS_ONLY= "[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?"
    BONDS= r'R\(\d+.\d+'
    FREQUENCY= r'\-{19}'
    CHARGES=r'^-?[0-9]+([.][0-9]+)?$'

class FileFlags(Enum):
    DIPOLE_START='Dipole moment'
    DIPOLE_END='Quadrupole moment'
    MAIN_CUTOFF= r'\-{69}'
    STANDARD_ORIENTATION_START='Standard orientation:'
    POL_START='iso'
    POL_END='xx'
    CHARGE_START='Summary of'
    CHARGE_END='====='
    FREQUENCY_START='Harmonic frequencies'
    FREQUENCY_END='Thermochemistry'
    HIRSH_CHARGE_START='Hirshfeld'
    HIRSH_CHARGE_END='Hirshfeld charges with hydrogens '

class Names(Enum):
    
    DIPOLE_COLUMNS=['dip_x','dip_y','dip_z','total_dipole']
    STANDARD_ORIENTATION_COLUMNS=['atom','x','y','z']
    DF_LIST=['standard_orientation_df', 'dipole_df', 'pol_df', 'atype_df','charge_df', 'bonds_df', 'info_df','energy_value']


class GeneralConstants(Enum):    
    ATOMIC_NUMBERS ={
     '1':'H', '5':'B', '6':'C', '7':'N', '8':'O', '9':'F', '14':'Si',
              '15':'P', '16':'S', '17':'Cl', '35':'Br', '53':'I', '27':'Co', '28':'Ni'}
     

    


def search_phrase_in_text(text_lines: str, key_phrase: str) : 
        """
        Searches for a key phrase in a list of text lines.
        
        Parameters:
            text_lines: text string to search through.
            key_phrase: The phrase to search for.
            
        Returns:
            The index of the first line where the key phrase is found, or None if not found.
        """   
        search_result=re.compile(key_phrase).search(text_lines)
        # print(f"Found key phrase '{key_phrase}' at line {search_result.start()}") if search_result else print(f"Key phrase '{key_phrase}' not found")
        return search_result

def extract_lines_from_text(text_lines, re_expression):
    selected_lines=re.findall(re_expression, text_lines)

    return selected_lines

def find_all_matches(log_file_lines, key_phrase):
    return [match for match in re.finditer(key_phrase, log_file_lines)]


def process_gaussian_charge_text(log_file_lines):
    charges_start =find_all_matches(log_file_lines, key_phrase=FileFlags.CHARGE_START.value)
    charges_end = find_all_matches(log_file_lines, key_phrase=FileFlags.CHARGE_END.value)
    if charges_start and charges_end:
        charges_start = charges_start[-1].end()
        charges_end = charges_end[-1].start()
        text=log_file_lines[charges_start:charges_end]
        selected_lines = extract_lines_from_text(text, re_expression=ReExpressions.FLOATS_ONLY.value)
    # selected_lines = (extract_lines_from_text(log_file_lines[charges_start.start():charges_end.start()],
    #                                          re_expression=ReExpressions.FLOATS_ONLY.value))
    charge_array = np.array(selected_lines)
    charge_array=charge_array[1::6]
    return pd.DataFrame(charge_array, columns=['charge'])

## This function is not used in the current version of the code- takes the last instance of dipole moment from log file

def process_gaussian_dipole_text(log_file_lines):
    # Find all occurrences of the start and end markers
    dipole_starts = find_all_matches(log_file_lines, FileFlags.DIPOLE_START.value)
    dipole_ends = find_all_matches(log_file_lines, FileFlags.DIPOLE_END.value)

    if dipole_starts and dipole_ends:
        last_dipole_start = dipole_starts[-1].end()
        last_dipole_end = dipole_ends[-1].start()
       
        text_section = log_file_lines[last_dipole_start:last_dipole_end]
        selected_lines = extract_lines_from_text(text_section, re_expression=ReExpressions.FLOAT.value)
        selected_lines = selected_lines[0:4] 
        dipole_df = pd.DataFrame(selected_lines, index=Names.DIPOLE_COLUMNS.value).T
        
        return dipole_df
    


def gauss_first_split(log_file_lines):
    first_split=re.split(FileFlags.STANDARD_ORIENTATION_START.value,log_file_lines)[-1]
    gauss_data=re.split(FileFlags.MAIN_CUTOFF.value,first_split)
    return gauss_data

def process_gaussian_standard_orientation_text(log_file_lines):
    standard_orientation_lines=(extract_lines_from_text(log_file_lines, ReExpressions.FLOATS_ONLY.value ))
    standard_orientation=np.array(standard_orientation_lines).reshape(-1,6)
    standard_orientation=np.delete(standard_orientation,(0,2),1)
    standard_orientation_df=pd.DataFrame(standard_orientation,columns=Names.STANDARD_ORIENTATION_COLUMNS.value)
    standard_orientation_df['atom'].replace(GeneralConstants.ATOMIC_NUMBERS.value, inplace=True)
    standard_orientation_df[['x', 'y', 'z']] = standard_orientation_df[['x', 'y', 'z']].astype(float)
    ## add 2 columns on top and add the .shape[0] to the first row 'atom' column
    # Create a new DataFrame with two rows
    new_rows = pd.DataFrame(np.nan, index=[0, 1], columns=standard_orientation_df.columns)
    # Set the 'atom' column in the first row to the number of rows in df
    new_rows.loc[0, 'atom'] = standard_orientation_df.shape[0]
    # Concatenate the new rows with df
    
    df = pd.concat([new_rows, standard_orientation_df], ignore_index=True)

    return df

# def process_gaussian_pol(log_file_lines):
#     pol_start=search_phrase_in_text(log_file_lines, key_phrase=FileFlags.POL_START.value)
#     pol_end=search_phrase_in_text(log_file_lines, key_phrase=FileFlags.POL_END.value)
#     pol=extract_lines_from_text(log_file_lines[pol_start.start():pol_end.start()], re_expression=ReExpressions.FLOAT.value)
#     pol_df=pd.DataFrame([float(pol[0])*1000,float(pol[3])*1000],index=['iso','aniso'],dtype=float)
#     return pol_df

def search_phrase_in_text_pol(text: str, key_phrase: str) -> int:
    """
    Search for a key phrase in a text string and return the start position of the match.

    Parameters:
        text (str): The text to search within.
        key_phrase (str): The phrase to search for.

    Returns:
        int: The start index of the first match or None if not found.
    """
    pattern = re.compile(rf"{re.escape(str(key_phrase))}\s")
    match = pattern.search(text)
    if match:
        return match.start()
    return None

def process_hirshfeld_charges(log_file_lines):
   
    charges_start = find_all_matches(log_file_lines, key_phrase=FileFlags.HIRSH_CHARGE_START.value)
    charges_end = find_all_matches(log_file_lines, key_phrase=FileFlags.HIRSH_CHARGE_END.value)

    if charges_start and charges_end:
        charges_start = charges_start[-2].end()
        charges_end = charges_end[-1].start()
        
        lines=log_file_lines[charges_start:charges_end]
    
    selected_lines = (extract_lines_from_text(lines,
                                             re_expression=ReExpressions.FLOAT.value))
   
    charge_array = np.array(selected_lines).reshape(-1,6)
    # print(lines)
  
    hirsh_charge_array=charge_array[:,0]
    cm5_charge_array=charge_array[:,5]

    

    return pd.DataFrame(hirsh_charge_array, columns=['hirsh_charge']),pd.DataFrame(cm5_charge_array, columns=['cm5_charge'])



def process_gaussian_pol_text(log_file_lines: List[str]) -> Optional[pd.DataFrame]:
    """
    Processes Gaussian polarization data and returns a DataFrame.
    
    Parameters:
        log_file_lines: List of lines from the Gaussian log file.
        
    Returns:
        A DataFrame containing polarization data or None if not found.
    """
    # If log_file_lines is a list, convert it to a single string
    if isinstance(log_file_lines, list):
        log_file_text = '\n'.join(log_file_lines)
    else:
        log_file_text = log_file_lines

    pol_start = search_phrase_in_text_pol(log_file_text, key_phrase=FileFlags.POL_START.value)
    pol_end = search_phrase_in_text_pol(log_file_text, key_phrase=FileFlags.POL_END.value)
    
    if pol_start is not None and pol_end is not None:
        pol = extract_lines_from_text(log_file_lines[pol_start:pol_end], re_expression=ReExpressions.FLOAT_POL.value)
        pol_df = pd.DataFrame([float(pol[0])*1000, float(pol[5])*1000], index=['iso', 'aniso'], dtype=float).T
        return pol_df
    else:
        # print("Failed to create.")
        pol_df = pd.DataFrame([100, 100], index=['iso', 'aniso'], dtype=float).T
        return pol_df




def process_gaussian_bonds(log_file_lines):
    bonds=extract_lines_from_text(log_file_lines, re_expression=ReExpressions.BONDS.value)
    bonds_text=[re.sub(r'R\(','',line).split(',') for line in bonds]
    return bonds_text


def remove_floats_until_first_int(input_list):
    output_list = []
    encountered_integer = False
    for item in input_list:
        try:
            # Try converting the string to an integer
            int_item = int(item)
            encountered_integer = True
            output_list.append(item)
        except ValueError:
            # If conversion to integer fails, try float
            try:
                float_item = float(item)
                if encountered_integer:
                    output_list.append(item)
            except ValueError:
                # If conversion to both integer and float fails, keep the item
                output_list.append(item)
    return output_list


def process_gaussian_vibs_string(log_file_lines):
    pattern = re.compile(rf'{FileFlags.FREQUENCY_START.value}([\s\S]*?){FileFlags.FREQUENCY_END.value}')
    match = pattern.search(log_file_lines)
    if match:
        vibration_section = match.group(1).strip()

        # Further slicing the obtained text based on 'Frequencies --'
        frequencies_blocks = re.split(r'Frequencies --', vibration_section)

        # Removing the last part that contains '------'
        final_blocks = [block.split('-------------------')[0].strip() for block in frequencies_blocks[1:]]

        return final_blocks
    
def process_gaussian_info(frequency_string):
    ir,frequency=[],[]
    for data in frequency_string:
        match=extract_lines_from_text(data, re_expression=ReExpressions.FLOATS_ONLY.value)
        frequency.append((match[0:3]))
        ir.append(match[6:9])
    info_df=pd.DataFrame()
    info_df['Frequency']=[(item) for sublist in frequency for item in sublist]
    info_df['IR']=[(item) for sublist in ir for item in sublist] 
    info_df=info_df.astype(float)

    return info_df

def vib_array_list_to_df(array_list):
    array_list_df=[]

    for array in array_list:
        new_array=np.delete(array,[0,1],axis=1).reshape(-1,3)
        new_df=pd.DataFrame(new_array)
        ## reverse the order of the columns
        # new_df=new_df.iloc[:, ::-1]  ## correct but not like shahar's
        array_list_df.append(new_df)
    
    vibs_df=pd.concat(array_list_df,axis=1)
    vibs_df=vibs_df.astype(float)
    return vibs_df


def process_gaussian_frequency_string(final_blocks):
    vibs_list=[]
    short_list=[]
    lenght=[]
    for i,data in enumerate(final_blocks):
            match=re.findall(ReExpressions.FLOATS_ONLY.value,data)
            # if i == len(final_blocks) - 1:
            #     del match[0:4]
            # else:
            del match[0:12] 
            match=np.array(match)
           ## need to find a way to deal with last match which is not a multiple of 11, planar vibrations
            # match=remove_floats_until_first_int(match)
            match=np.array(match)
            if i==1:
                match=remove_floats_until_first_int(match)
            match=np.array(match)
            lenght.append(len(match))
            try:
                # print(match.reshape(-1,11))
                vibs_list.append(match.reshape(-1,11))  
            except ValueError:   
                    try:
                        match_try=np.delete(match,[-1],0)
                        # print(match_try)
                        vibs_list.append(np.array(match_try).reshape(-1,11))
                    except Exception as e:
                        try: 
                            
                            match_r=np.delete(match,[-1,-2,-3],0)
                            # print(match_r)
                            vibs_list.append(np.array(match_r).reshape(-1,11))
                        except Exception as e:
                            raise ValueError(f"Error processing vibrations: {e}")
                                
    vibs=np.vstack(vibs_list)
    
    np.set_printoptions(threshold=np.inf)
    final_atom=vibs[-1][0]
    ordered_vibs=[]
    for i in range(1,int(final_atom)+1):
        ordered_vibs.append(vibs[vibs[:,0]==str(i)])
    # print(ordered_vibs)
    # ordered_vibs_list=[ordered_vibs[i:i + len(vibs_list)] for i in range(0, len(ordered_vibs), len(vibs_list))]
    # print(ordered_vibs_list)
    vibs_df=vib_array_list_to_df(ordered_vibs) if ordered_vibs else None
    # vibs_df=vib_array_list_to_df_5(ordered_vibs_list,new_arrays) if ordered_vibs_list else None
    
    return vibs_df


def df_list_to_dict(df_list):
    my_dict={}
    for name,df in zip(Names.DF_LIST.value,df_list):
        my_dict[name]=df
    return my_dict



def process_gaussian_energy_text(energy_string):
    cut=re.split('SCF Done', energy_string)[1]
    energy=extract_lines_from_text(cut, re_expression=ReExpressions.FLOAT.value)
    data = np.array([[energy[0]]])
    df = pd.DataFrame(data, columns=['energy'])
    df=df.astype(float)

    return df





def gauss_file_handler(gauss_filename, export=False):
    string_report=''
    
    
    with open(os.path.abspath(gauss_filename)) as f:
        log_file_lines = f.read()

    # try:
    #     energy_df = process_gaussian_energy_text(log_file_lines)
    #     energy_df=energy_df.astype(float)
    # except IndexError:
    #     energy_df = pd.DataFrame()
    #     print("{gauss_filename}: Error processing energy.")
    #     string_report+="{gauss_filename}: Error processing energy.\n"

    try:
        charge_df = process_gaussian_charge_text(log_file_lines)
        charge_df=charge_df.astype(float)
    except Exception as e:
        charge_df = pd.DataFrame()
        print(f"{gauss_filename}: Error processing charge: {e}")
        string_report+=f"{gauss_filename}: Error processing charge: {e}\n"
    try:
        dipole_df = process_gaussian_dipole_text(log_file_lines)
        dipole_df=dipole_df.astype(float)
    except Exception as e:
        dipole_df = pd.DataFrame()
        print(f"{gauss_filename}: Error processing dipole: {e}")
        string_report+=f"{gauss_filename}: Error processing dipole: {e}\n"
    try:
        pol_df = process_gaussian_pol_text(log_file_lines)
        pol_df=pol_df.astype(float)
    except Exception as e:
        pol_df = pd.DataFrame()
        print(f"{gauss_filename}: Error processing polarization: {e}")
        string_report+=f"{gauss_filename}: Error processing polarization: {e}\n"
    try:
        gauss_data = gauss_first_split(log_file_lines)
        standard_orientation_df = process_gaussian_standard_orientation_text(gauss_data[2])
        print(standard_orientation_df)
        
    except Exception as e:
        standard_orientation_df = pd.DataFrame()
        print(f"{gauss_filename}: Error processing standard orientation: {e}")
        string_report+=f"{gauss_filename}: Error processing standard orientation: {e}\n"
    try:
        frequency_str = process_gaussian_vibs_string(log_file_lines)
    except Exception as e:
        frequency_str = pd.DataFrame()
        print(f"{gauss_filename}: Error processing vibrations: {e}")
        string_report+=f"{gauss_filename}: Error processing vibrations: {e}\n"
    try:
        info_df = process_gaussian_info(frequency_str)
        info_df=info_df.astype(float)
        
    except Exception as e:
        info_df = pd.DataFrame()  # or some default DataFrame
        print(f"{gauss_filename}: Error processing info: {e}")
        string_report+=f"{gauss_filename}: Error processing info: {e}\n"

    try:
        vibs_df = process_gaussian_frequency_string(frequency_str)
    except Exception as e:
        vibs_df = pd.DataFrame() # or some default DataFrame
        print(f"{gauss_filename}: Error processing frequency: {e}")
        string_report+=f"{gauss_filename}: Error processing frequency: {e}\n"
    
    try:
        hirsh_charge_df, cm5_charge_df = process_hirshfeld_charges(log_file_lines)
        hirsh_charge_df=hirsh_charge_df.astype(float)
        cm5_charge_df=cm5_charge_df.astype(float)
        print(f"{gauss_filename}: NBO, Hirshfeld and CM5 charges")
        # charge_df={'nbo_charge':charge_df,'hirsh_charge':hirsh_charge_df,'cm5_charge':cm5_charge_df} ### need to edit this -- working

    except Exception as e:
        hirsh_charge_df = pd.DataFrame({'NaN_Column': [np.nan] * standard_orientation_df.shape[0]})
        cm5_charge_df = pd.DataFrame({'NaN_Column': [np.nan] * standard_orientation_df.shape[0]})
        print(f"{gauss_filename}: NBO charge only")
        string_report+=f"{gauss_filename}: NBO charge only\n"


    try:
        concatenated_df = pd.concat([standard_orientation_df, dipole_df, pol_df, charge_df, hirsh_charge_df,cm5_charge_df, info_df, vibs_df], axis=1)
    except Exception as e:
        concatenated_df = pd.DataFrame()  # or some default DataFrame
        print(f"{gauss_filename}: Error concatenating data: {e}")
        string_report+=f"{gauss_filename}: Error concatenating data: {e}\n"

    return concatenated_df, string_report


def save_to_feather(df, filename):
    # Create a DataFrame from the list of strings
    # Each string becomes a column. Since all columns must be of the same length,
    # you may need to handle this if your strings are of different lengths.
    # Define column names that correspond to the data in string_list
    # column_names = ['Standard_Orientation', 'Dipole', 'Polarizability', 'Frequency', 'Charge', 'Energy']

    df['atom'] = df['atom'].astype(str)
    feather_filename = filename + '.feather'
    # df=df.astype(str)
    # Set each column name to a string representation of its index
    df.columns = range(df.shape[1]) # [str(i) for i in range(df.shape[1])]
    df.columns = df.columns.map(str)
    df.to_feather(feather_filename)

    print(f"Data saved to {feather_filename}")
    string_report=f"Data saved to {feather_filename}\n"
    return string_report

def logs_to_feather(dir_path):
    string_report=''
    failed_files_string='Files with Errors created with missing DataFrames-\n'
    os.chdir(dir_path)
    if not os.path.exists('feather_files'):
        os.mkdir('feather_files')

    for file in os.listdir(dir_path):
        if file.endswith(".log"):
            # try:
            df, gauss_string_report = gauss_file_handler(file)
            string_report+=gauss_string_report
            # except Exception as e:
            #     print(f"Error processing file {file}: {e}")
            #     string_report+=f"Error processing file {file}: {e}\n"
            #     failed_files_string+=f"{file}\n"
            #     continue  # Skip to the next file

            os.chdir('feather_files')
            
            string_report+=save_to_feather(df, file.split('.')[0])  # Assuming you want to remove the .log extension
            os.chdir('..')
        else:
            continue
    string_report+=failed_files_string + 'Check the log files and reported errors for more information.'
    os.chdir(dir_path)
    
    print('Done!')
    return string_report

import os

def logs_to_dict(dir_path):
    """
    Processes Gaussian log files in the given directory, returning a dictionary of DataFrames.

    Parameters:
    dir_path (str): Path to the directory containing Gaussian log files.

    Returns:
    dict: A dictionary where keys are file names (without .log) and values are DataFrames.
    """
    file_dict = {}
    error_log = {}

    os.chdir(dir_path)

    for file in os.listdir(dir_path):
        if file.endswith(".log"):
            try:
                df, gauss_string_report = gauss_file_handler(file)
                file_dict[file.split('.')[0]] = df
                dict_list=df_file_handler(df)
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                error_log[file] = str(e)  # Store error message for debugging

    return dict_list, error_log


def df_list_to_dict(df_list):
    my_dict={}
    for name,df in zip(['standard_orientation_df', 'dipole_df', 'pol_df', 'info_df'],df_list):
        my_dict[name]=df
    return my_dict

def df_file_handler(df):
    # Read the feather file
    
    data = df
    
    data.columns=range(len(data.columns))
    ## change pandas options to display all columns
  
    xyz = data.iloc[:, 0:4].dropna()
    ## check if xyz rows were drop , if not drop the first 2


    
    # if not xyz.iloc[:2].isnull().any(axis=None):
    #  Drop the first two rows if they contain values
    #     xyz = xyz.iloc[2:]

    
    dipole_df = data.iloc[:, 4:8].dropna()
    pol_df = data.iloc[:, 8:10].dropna()
    nbo_charge_df = data.iloc[:, 10:11].dropna()
    ## if the whole column is NaN, it will be removed
    hirshfeld_charge_df = data.iloc[:,11:12].dropna().reset_index(drop=True)
    cm5_charge_df = data.iloc[:,12:13].dropna().reset_index(drop=True)
    info_df = data.iloc[:, 13:15].dropna()
    vectors = data.iloc[:, 15:].dropna()
    

    
        
    xyz.rename(columns={xyz.columns[0]: 'atom', xyz.columns[1]: 'x', xyz.columns[2]: 'y', xyz.columns[3]: 'z'}, inplace=True)
        # Remove the first two rows
    # xyz = xyz.iloc[2:]
    xyz = xyz.reset_index(drop=True)
   
    xyz[['x', 'y', 'z']] = xyz[['x', 'y', 'z']].astype(float)
 
    xyz=xyz.dropna()
    # Calculate the length of the DataFrame
    dipole_df.rename(columns={dipole_df.columns[0]: 'dip_x', dipole_df.columns[1]: 'dip_y', dipole_df.columns[2]: 'dip_z', dipole_df.columns[3]: 'total_dipole'}, inplace=True)
    dipole_df=dipole_df.astype(float)
    dipole_df=dipole_df.dropna()
    dipole_df=dipole_df.reset_index(drop=True)
    nbo_charge_df.rename(columns={nbo_charge_df.columns[0]: 'charge'}, inplace=True)
    nbo_charge_df=nbo_charge_df.astype(float)
    nbo_charge_df=nbo_charge_df.dropna()
    nbo_charge_df=nbo_charge_df.reset_index(drop=True)

    charge_dict={'nbo':nbo_charge_df, 'hirshfeld':hirshfeld_charge_df, 'cm5':cm5_charge_df}


    pol_df.rename(columns={pol_df.columns[0]: 'aniso', pol_df.columns[1]: 'iso'}, inplace=True)
    pol_df=pol_df.astype(float)
    pol_df=pol_df.dropna()
    pol_df=pol_df.reset_index(drop=True)
    info_df.rename(columns={info_df.columns[0]: 'Frequency', info_df.columns[1]: 'IR'}, inplace=True)
    info_df=info_df.astype(float)
    info_df=info_df.dropna()
    info_df=info_df.reset_index(drop=True)
    
    def split_to_dict(dataframe):
        
        num_columns = dataframe.shape[1]
        num_dfs = num_columns // 3  # Integer division to get the number of 3-column dataframes

        dfs_dict = {}
        for i in range(num_dfs):
            start_col = i * 3
            end_col = start_col + 3
            key = f'vibration_atom_{i + 1}'
            dfs_dict[key] = np.array(dataframe.iloc[:, start_col:end_col].values.astype(float))  # Storing as a NumPy array

        return dfs_dict
    
    vib_dict=split_to_dict(vectors)
    df_list=[xyz ,dipole_df, pol_df, info_df]
    df_list=[df.dropna(how='all') for df in df_list if not df.empty]
    df_dict=df_list_to_dict(df_list)
    dict_list=[df_dict,vib_dict,charge_dict]

    return dict_list

if __name__ == "__main__":

    path=r'C:\Users\Lenovo\OneDrive\Desktop\university\lab\Mol_align-main\log_files'
    dict_list=logs_to_feather(path)

    ## file dict is comprised of [df_dict,vib_dict,charge_dict]
    ## you need the vib_dict and df_dict['info_df'] to get the vibrational frequencies
    
