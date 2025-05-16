import pandas as pd
import os
from Bio.PDB import PDBParser, Polypeptide
import numpy as np
import torch
from torch_cluster import knn
from torch_geometric.data import Data
import warnings
warnings.filterwarnings("ignore")
import sys

# 项目根目录
project_root = '/public/home/wangxin5/NanoKGAT/NanoKGAT-main/NanoKGAT-main/NanoKGAT'
if project_root not in sys.path:
    sys.path.append(project_root)

from antiberty import AntiBERTyRunner

import warnings
warnings.filterwarnings("ignore")

NB_MAX_LENGTH = 140
NB_CHAIN_ID = "H"
BACKBONE_ATOMS = ["N", "CA", "C", "O", "CB"]
OUTPUT_SIZE = len(BACKBONE_ATOMS) * 3

def is_aa(aa):
    return Polypeptide.is_aa(aa, standard=True)

def get_seq_aa(pdb_file, chain_id):
    pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file)
    chain = structure[0][chain_id]

    aa_residues = []
    seq = ""

    for residue in chain.get_residues():
        aa = residue.get_resname()
        if not is_aa(aa) or not residue.has_id('CA'):
            continue
        elif aa == "UNK":
            seq += "X"
        else:
            seq += Polypeptide.three_to_one(aa)
        aa_residues.append(residue)

    return seq, aa_residues

def generate_coord(pdb_file, chain_id):
    print(f"Processing PDB: {pdb_file}, Chain ID: {chain_id}")
    pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file)
    chain = structure[0][chain_id]

    aa_residues = []
    seq = ""

    for residue in chain.get_residues():
        aa = residue.get_resname()
        if not is_aa(aa) or not residue.has_id('CA'):
            continue
        elif aa == "UNK":
            seq += "X"
        else:
            seq += Polypeptide.three_to_one(aa)
        aa_residues.append(residue)

    if len(seq) > NB_MAX_LENGTH:
        print(f"警告: 序列长度 {len(seq)} 超过最大长度 {NB_MAX_LENGTH}。将截断。")
        seq = seq[:NB_MAX_LENGTH]
        aa_residues = aa_residues[:NB_MAX_LENGTH]

    xyz_matrix = np.zeros((NB_MAX_LENGTH, OUTPUT_SIZE))
    print(aa_residues)
    for i in range(len(aa_residues)):
        for j, atom in enumerate(BACKBONE_ATOMS):
            if not (atom == "CB" and seq[i] == "G"):
                try:
                    xyz_matrix[i][3*j:3*j+3] = aa_residues[i][atom].get_coord()
                except Exception as e:
                    print(f"错误: 获取原子 {atom} 在残基 {i} 的坐标时出错: {e}")
                    xyz_matrix[i][3*j:3*j+3] = np.nan

    return xyz_matrix[:,3:6]

def set_coord(df, pdb_dir):
    empty_col = pd.Series('', index=df.index)
    df['coord'] = empty_col
    for index, row in df.iterrows():
        pdb_name = row['PDB']
        if not pdb_name.endswith('.pdb'):
            pdb_name_with_ext = f"{pdb_name}.pdb"
        else:
            pdb_name_with_ext = pdb_name

        print(f"Processing PDB: {pdb_name_with_ext}")
        chain_id = row['nano_chain']
        
        pdb_file_path = os.path.join(pdb_dir, pdb_name_with_ext)
        
        if not os.path.isfile(pdb_file_path):
            print(f"警告: 文件 {pdb_file_path} 不存在。跳过此行。")
            df.at[index, 'coord'] = np.nan
            continue

        try:
            pdb_coord = generate_coord(pdb_file_path, chain_id)
            df.at[index, 'coord'] = pdb_coord
        except Exception as e:
            print(f"错误: 处理文件 {pdb_file_path} 时出错: {e}")
            df.at[index, 'coord'] = np.nan

    return df

def make_data(df, k):
    dataset = []
    for index, row in df.iterrows():
        seq = list(row['sequence'])
        size = len(seq[0])
        label = torch.tensor(np.where(row['paratope_labels'] == 'P', 1, 0))
        coord = torch.tensor(row['coord'])

        antiberty = AntiBERTyRunner()
        embeddings = antiberty.embed(seq)[0][1:-1]
        if size < NB_MAX_LENGTH:
            pad = torch.zeros((NB_MAX_LENGTH - size, 512)).cuda()
            embeddings = torch.cat([embeddings, pad], dim=0)

        edge_index = knn(coord, coord, k=k)
        data = Data(x=embeddings, y=label, pos=coord, edge_index=edge_index, mask=size)
        dataset.append(data)

    return dataset

