import numpy
import requests
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Crippen
from rdkit.Chem import MolFromSmiles

from ..Combos_prparation.prepare_combos import get_cid


def get_properities_dictionary(drug_name: str) -> dict | None:
    """
    This function retrieves the properties of a drug from PubChem using its name. This function uses the PubChem RESTful API to retrieve the properties.

    :param drug_name: The name of the drug.

    :return: A dictionary containing the properties of the drug. If the drug is not found, returns None.
    """
    drug_cid = get_cid(drug_name)
    if drug_cid:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{drug_cid}/property/CanonicalSMILES,MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,FeatureRingCount3D,RotatableBondCount/json"
        response = requests.get(url)
        if response.status_code == 200:
            drug_properities = response.json()["PropertyTable"]["Properties"][0]
            return drug_properities
        else:
            return None
    else:
        return None


def get_smiles(drug_name: str, drug_properties_dict: dict = None) -> str | None:
    """
    This function retrieves the canonical SMILES representation of a drug from a given dictionary of properties.
    If the dictionary is not provided, it will fetch the properties from PubChem using the drug name.

    :param drug_name: The name of the drug.
    :param drug_properties_dict: A dictionary containing the properties of the drug. Defaults to None.
    
    :return: The canonical SMILES representation of the drug. If the drug is not found or the properties are not available, returns None.
    """
    if drug_properties_dict is None:
        drug_properties_dict = get_properities_dictionary(drug_name)
    return drug_properties_dict.get("CanonicalSMILES")


def get_mwt(drug_name: str, drug_properties_dict: dict = None) -> int | None:
    """
    This function retrieves the molecular weight of a drug from a given dictionary of properties.
    If the dictionary is not provided, it will fetch the properties from PubChem using the drug name.

    :param drug_name: The name of the drug.
    :param drug_properties_dict: A dictionary containing the properties of the drug. Defaults to None.
    
    :return: The molecular weight of the drug. If the drug is not found or the properties are not available, returns None.
    """
    if drug_properties_dict is None:
        drug_properties_dict = get_properities_dictionary(drug_name)
    return drug_properties_dict.get("MolecularWeight")


def get_clogp(drug_name: str, drug_properties_dict: dict = None) -> float | None:
    """
    This function retrieves the ClogP (octanol-water partition coefficient) of a drug.
    If the ClogP value is available in the provided dictionary, it is returned.
    If not, the function calculates the ClogP using the Crippen's method.

    :param drug_name: _description_
    :param drug_properties_dict: A dictionary containing the properties of the drug. Defaults to None.
    
    :return: The ClogP value of the drug. If the drug is not found or the properties are not available, returns None.
    """
    if drug_properties_dict is None:
        drug_properties_dict = get_properities_dictionary(drug_name)
    if drug_properties_dict.get("XLogP"):
        return drug_properties_dict.get("XLogP")
    else:
        smiles = get_smiles(drug_name, drug_properties_dict)
        mol = Chem.MolFromSmiles(smiles)
        # Calculate the logP using Crippen's method
        logP = Crippen.MolLogP(mol)
        return logP


def get_tpsa(drug_name: str, drug_properties_dict: dict = None) -> float | None:
    """This function retrieves the topological polar surface area (TPSA) of a drug.
    If the TPSA value is available in the provided dictionary, it is returned.
    If not, the function fetches the properties from PubChem using the drug name and gets the TPSA.

    :param drug_name: The name of the drug.
    :param drug_properties_dict: A dictionary containing the properties of the drug. Defaults to None.
    
    :return: The TPSA value of the drug. If the drug is not found or the properties are not available, returns None.
    """
    if drug_properties_dict is None:
        drug_properties_dict = get_properities_dictionary(drug_name)
    return drug_properties_dict.get("TPSA")


def get_hdonor(drug_name: str, drug_properties_dict: dict = None) -> int|None:
    """
    This function retrieves the number of hydrogen bond donors of a drug.
    If the dictionary of properties is not provided, it will fetch the properties from PubChem using the drug name.

    :param drug_name: The name of the drug.
    :param drug_properties_dict: A dictionary containing the properties of the drug. Defaults to None.
    
    :return: The number of hydrogen bond donors of the drug. If the drug is not found or the properties are not available, returns None.
    """
    if drug_properties_dict is None:
        drug_properties_dict = get_properities_dictionary(drug_name)
    return drug_properties_dict.get("HBondDonorCount")


def get_hacceptor(drug_name: str, drug_properties_dict: dict = None) -> int | None:
    """
    This function retrieves the number of hydrogen bond acceptors of a drug.
    If the dictionary of properties is not provided, it will fetch the properties from PubChem using the drug name.

    :param drug_name: The name of the drug.
    :param drug_properties_dict: A dictionary containing the properties of the drug. Defaults to None.
    
    :return: The number of hydrogen bond acceptors of the drug. If the drug is not found or the properties are not available, returns None.
    """
    if drug_properties_dict is None:
        drug_properties_dict = get_properities_dictionary(drug_name)
    return drug_properties_dict.get("HBondAcceptorCount")


def get_rbond(drug_name: str, drug_properties_dict: dict = None) -> str:
    """
    This function retrieves the number of rotatable bonds of a drug.
    If the dictionary of properties is not provided, it will fetch the properties from PubChem using the drug name.

    :param drug_name: The name of the drug.
    :param drug_properties_dict: A dictionary containing the properties of the drug. Defaults to None.
    
    :return: The number of rotatable bonds of the drug. If the drug is not found or the properties are not available, returns None.
    """
    
    if drug_properties_dict is None:
        drug_properties_dict = get_properities_dictionary(drug_name)
    return drug_properties_dict.get("RotatableBondCount")


def generate_morgan_fingerprint(
    drug_name: str,
    drug_properties_dict: dict = None,
    radius: int = 6,
    nBits: int = 2048,
) -> numpy.array:
    """
    This function generates a Morgan fingerprint of a drug.
    If the dictionary of properties is not provided, it will fetch the properties from PubChem using the drug name.

    :param drug_name: The name of the drug.
    :param drug_properties_dict: A dictionary containing the properties of the drug. Defaults to None.
    :param radius: The radius of the Morgan fingerprint. Defaults to 6.
    :param nBits: The number of bits in the fingerprint. Defaults to 2048.
    
    :return: The Morgan fingerprint of the drug. If the drug is not found or the properties are not available, returns None.
    """
    if drug_properties_dict is None:
        drug_properties_dict = get_properities_dictionary(drug_name)
    smiles = get_smiles(drug_name, drug_properties_dict)
    mol = Chem.MolFromSmiles(smiles)
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    morgan_fp_np = numpy.zeros((1,))
    Chem.DataStructs.ConvertToNumpyArray(morgan_fp, morgan_fp_np)
    return morgan_fp_np


def calculate_tanimoto_coefficient(
    drug1_name: str,
    drug2_name: str,
    drug1_properties_dict: dict = None,
    drug2_properties_dict: dict = None,
    radius: int = 6,
    nBits: int = 2048,
) -> float:
    """
    This function calculates the Tanimoto coefficient between two drugs.
    If the dictionaries of properties are not provided, it will fetch the properties from PubChem using the drug names.

    :param drug1_name: The name of the first drug.
    :param drug2_name: The name of the second drug.
    :param drug1_properties_dict: A dictionary containing the properties of the first drug. Defaults to None.
    :param drug2_properties_dict: A dictionary containing the properties of the second drug. Defaults to None.
    :param radius: The radius of the Morgan fingerprint. Defaults to 6.
    :param nBits: The number of bits in the fingerprint. Defaults to 2048.
    
    :return: The Tanimoto coefficient between the two drugs. If the drugs are not found or the properties are not available, returns None.
    """

    if drug1_properties_dict is None:
        drug1_properties_dict = get_properities_dictionary(drug1_name)

    if drug2_properties_dict is None:
        drug2_properties_dict = get_properities_dictionary(drug2_name)

    smiles1 = get_smiles(drug1_name, drug1_properties_dict)
    smiles2 = get_smiles(drug2_name, drug2_properties_dict)

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits)

    # Calculate the Tanimoto coefficient
    tanimoto_coefficient = DataStructs.TanimotoSimilarity(fp1, fp2)
    return tanimoto_coefficient
