#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, re, glob, tempfile, traceback
import pandas as pd
from typing import List, Tuple

# --- knobs ---
RECURSIVE = True                 # recurse into subfolders
USE_CRYSTAL_NN = True            # True: CrystalNN; False: CutOffDictNN
CUTOFF_METAL_O = 2.7             # Å for M–O/N/halides when using cutoff
MIN_COMPONENT_SIZE = 3           # ignore tiny fragments
NON_METALS = {"H","B","C","N","O","F","P","S","Cl","Se","Br","I","Si","Te"}
MIN_HEAVY_ATOMS = 8              # minimum number of heavy atoms, to filter small guest/solvent

def log(msg): 
    print(msg, flush=True)

def discover_cifs(path: str) -> List[str]:
    if os.path.isdir(path):
        pattern = "**/*.cif" if RECURSIVE else "*.cif"
        files = glob.glob(os.path.join(path, pattern), recursive=RECURSIVE)
    elif os.path.isfile(path) and path.lower().endswith(".cif"):
        files = [path]
    else:
        files = []
    return sorted(files)

def read_structure_clean(path: str):
    from pymatgen.core.structure import IStructure
    from pymatgen.io.cif import CifParser
    # 1) direct
    try:
        return IStructure.from_file(path)
    except Exception:
        pass
    # 2) ASE roundtrip
    try:
        from ase.io import read, write
        atoms = read(path)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".cif"); tmp.close()
        write(tmp.name, atoms)
        return IStructure.from_file(tmp.name)
    except Exception:
        pass
    # 3) tolerant CifParser + order disorder
    from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
    parser = CifParser(path, occupancy_tolerance=100.0)
    structs = parser.parse_structures(primitive=False)
    if not structs:
        raise ValueError("No structures parsed from CIF")
    struct = structs[0]
    try: 
        struct.remove_oxidation_states()
    except Exception: 
        pass
    if not struct.is_ordered:
        try: 
            struct = OrderDisorderedStructureTransformation().apply_transformation(struct)
        except Exception: 
            pass
    return struct

def structure_to_graph(struct):
    from pymatgen.analysis.graphs import StructureGraph
    if USE_CRYSTAL_NN:
        from pymatgen.analysis.local_env import CrystalNN
        return StructureGraph.with_local_env_strategy(struct, CrystalNN())
    else:
        from pymatgen.analysis.local_env import CutOffDictNN
        # default small cutoff
        species = list({str(s.specie) for s in struct.sites})
        cutoff = {(a,b): 1.9 for a in species for b in species}
        metals = [str(s.specie) for s in struct.sites if s.specie.is_metal]
        for m in metals:
            for x in ["O","N","Cl","F","Br","I","S"]:
                cutoff[(m,x)] = CUTOFF_METAL_O
                cutoff[(x,m)] = CUTOFF_METAL_O
        return StructureGraph.with_local_env_strategy(struct, CutOffDictNN(cutoff_dict=cutoff))

# original simple component splitting function (now no longer used in process_one, but kept for comparison)
def split_linker_components(struct, sgraph):
    import networkx as nx
    G = sgraph.graph.to_undirected()
    idx_is_metal = {i: struct[i].specie.is_metal for i in range(len(struct))}
    keep = [i for i, isM in idx_is_metal.items() if not isM]
    if not keep: 
        return []
    G2 = G.subgraph(keep).copy()
    comps = [sorted(c) for c in nx.connected_components(G2)]
    comps = [c for c in comps if len(c) >= MIN_COMPONENT_SIZE]
    return comps

def rdkit_smiles_from_component(struct, comp_idx: List[int]) -> str:
    """
    Build an RDKit molecule by distance-based bonding,
    then output canonical SMILES (no Open Babel needed).
    """
    from rdkit import Chem
    from rdkit.Chem import rdchem
    # Build a Molecule for the component (Cartesian coords not needed for SMILES)
    atoms = [struct[i].specie.symbol for i in comp_idx]
    emol = Chem.EditableMol(Chem.Mol())
    atom_map = {}
    for i, sym in enumerate(atoms):
        a = Chem.Atom(sym)
        idx = emol.AddAtom(a)
        atom_map[comp_idx[i]] = idx

    # Simple distance-based bonding among non-metals:
    coords = [struct[i].coords for i in comp_idx]
    symbols = atoms
    COV = {"H":0.31,"B":0.81,"C":0.76,"N":0.71,"O":0.66,"F":0.57,
           "Si":1.11,"P":1.07,"S":1.05,"Cl":1.02,"Se":1.20,"Br":1.20,"I":1.39}
    import math
    n = len(comp_idx)
    for i in range(n):
        for j in range(i+1, n):
            ri = COV.get(symbols[i], 0.8); rj = COV.get(symbols[j], 0.8)
            max_bond = 1.25*(ri+rj) + 0.2  # generous
            dx = coords[i][0]-coords[j][0]; dy = coords[i][1]-coords[j][1]; dz = coords[i][2]-coords[j][2]
            d = math.sqrt(dx*dx+dy*dy+dz*dz)
            if d <= max_bond:
                emol.AddBond(atom_map[comp_idx[i]], atom_map[comp_idx[j]], rdchem.BondType.SINGLE)

    mol = emol.GetMol()
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
    except Exception:
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
        except Exception:
            pass
    smi = Chem.MolToSmiles(mol, canonical=True)
    return smi

def is_big_enough(smi: str, min_heavy: int = MIN_HEAVY_ATOMS) -> bool:
    """
    count heavy atoms (non-H) using RDKit, filter out water/small anions/small solvents.
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    return mol.GetNumHeavyAtoms() >= min_heavy

def metals_from_structure(struct) -> List[str]:
    mets = []
    for s in struct:
        el = s.specie
        if el.is_metal: 
            mets.append(el.symbol)
    return sorted(set(mets))

def split_linker_components_with_metal_contacts(struct, sgraph):
    """
    和之前相比：
    - 仍然是删掉所有金属，只在非金属子图上做连通分量；
    - 但额外记录每个 component 是否有至少一个原子与金属相连。
    返回列表：[(component_indices, has_metal_contact), ...]
    """
    import networkx as nx
    G = sgraph.graph.to_undirected()

    idx_is_metal = {i: struct[i].specie.is_metal for i in range(len(struct))}
    # record: whether each non-metal atom is connected to a metal
    idx_contacts_metal = {i: False for i in range(len(struct))}
    for i, j in G.edges():
        if idx_is_metal[i] and not idx_is_metal[j]:
            idx_contacts_metal[j] = True
        if idx_is_metal[j] and not idx_is_metal[i]:
            idx_contacts_metal[i] = True

    # remove metals, get non-metal subgraph
    non_metal_nodes = [i for i, isM in idx_is_metal.items() if not isM]
    if not non_metal_nodes:
        return []
    Gnm = G.subgraph(non_metal_nodes).copy()

    comps = []
    for comp in nx.connected_components(Gnm):
        comp = sorted(list(comp))
        if len(comp) < MIN_COMPONENT_SIZE:
            continue
        has_metal_contact = any(idx_contacts_metal[i] for i in comp)
        comps.append((comp, has_metal_contact))

    return comps

def process_one(path: str) -> Tuple[List[str], List[str], str]:
    """
    new logic:
    - 用 split_linker_components_with_metal_contacts 拿到每个 component 以及是否接触金属；
    - 只对 has_metal_contact=True 的 component 转成 SMILES；
    - 再用 is_big_enough 过滤掉重原子太少的（比如水/NO3-/small solvent）。
    """
    try:
        struct = read_structure_clean(path)
        sgraph = structure_to_graph(struct)
        comps = split_linker_components_with_metal_contacts(struct, sgraph)

        smiles = []
        for comp_indices, has_metal_contact in comps:
            # only keep organic components that are truly connected to metals
            if not has_metal_contact:
                continue
            try:
                smi = rdkit_smiles_from_component(struct, comp_indices)
                if smi and is_big_enough(smi):
                    smiles.append(smi)
            except Exception:
                # keep going; record failure via status if none succeed, just log the error
                pass

        metals = metals_from_structure(struct)
        status = "ok" if smiles or metals else "ok_empty"
        return metals, smiles, status
    except Exception as e:
        return [], [], f"error: {e.__class__.__name__}: {e}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python moffragmentor_break.py <cif_file_or_folder>")
        sys.exit(1)
    inpath = sys.argv[1]
    files = discover_cifs(inpath)
    log(f"[info] found {len(files)} CIF(s) under: {inpath}")
    if not files:
        out_csv = os.path.join(os.getcwd(), "fragment_results.csv")
        pd.DataFrame([], columns=["file","metals","n_linkers","linker_smiles","status"]).to_csv(out_csv, index=False)
        log(f"[warn] no CIF files found. Wrote empty CSV: {out_csv}")
        sys.exit(0)

    rows = []
    for f in files:
        log(f"\n=== Processing: {f}")
        metals, smiles, status = process_one(f)
        log(f"Metals: {', '.join(metals) if metals else '(none)'}")
        if smiles:
            for i, s in enumerate(smiles, 1):
                log(f"L{i}: {s}")
        else:
            log(f"Linkers: (none)  -> status: {status}")
        rows.append({
            "file": f,
            "metals": ";".join(metals),
            "n_linkers": len(smiles),
            "linker_smiles": "|".join(smiles),
            "status": status
        })

    out_csv = os.path.join(os.getcwd(), "fragment_results.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log(f"\n[done] wrote: {out_csv}  (rows: {len(rows)})")

if __name__ == "__main__":
    main()