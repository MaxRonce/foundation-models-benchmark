#!/usr/bin/env python3
"""
Script d'investigation pour comprendre le problème de matching des object_id
entre les embeddings et le catalogue FITS.
"""
import sys
import torch
import numpy as np
from pathlib import Path
from astropy.io import fits

print("🔍 Investigation du problème de matching object_id")
print("=" * 80)

# Chemins (modifiables via arguments)
if len(sys.argv) >= 4:
    AION_EMB = sys.argv[1]
    ASTROPT_EMB = sys.argv[2]
    CATALOG = sys.argv[3]
else:
    AION_EMB = "/n03data/ronceray/embeddings/aion_embeddings_spec_image_with_retrained_codec_428529.pt"
    ASTROPT_EMB = "/n03data/ronceray/embeddings/astropt_embeddings.pt"
    CATALOG = "/home/ronceray/AION/DESI_DR1_Euclid_Q1_dataset_catalog_EM.fits"

print(f"\n📂 Fichiers:")
print(f"   AION: {AION_EMB}")
print(f"   AstroPT: {ASTROPT_EMB}")
print(f"   Catalogue: {CATALOG}")

# ============================================================================
# 1. ANALYSE DES EMBEDDINGS
# ============================================================================
print("\n" + "=" * 80)
print("1️⃣  ANALYSE DES EMBEDDINGS")
print("=" * 80)

print("\n📊 AION embeddings...")
aion_data = torch.load(AION_EMB, map_location="cpu")
print(f"   Nombre de records: {len(aion_data)}")
print(f"   Type: {type(aion_data)}")

if len(aion_data) > 0:
    print(f"\n   Keys du premier record: {list(aion_data[0].keys())}")
    print(f"\n   Exemples d'object_id (10 premiers):")
    for i in range(min(10, len(aion_data))):
        obj_id = aion_data[i].get("object_id")
        print(f"      [{i}] {obj_id} (type: {type(obj_id).__name__})")

print("\n📊 AstroPT embeddings...")
astropt_data = torch.load(ASTROPT_EMB, map_location="cpu")
print(f"   Nombre de records: {len(astropt_data)}")

if len(astropt_data) > 0:
    print(f"\n   Keys du premier record: {list(astropt_data[0].keys())}")
    print(f"\n   Exemples d'object_id (10 premiers):")
    for i in range(min(10, len(astropt_data))):
        obj_id = astropt_data[i].get("object_id")
        print(f"      [{i}] {obj_id} (type: {type(obj_id).__name__})")

# Comparer les IDs
aion_ids = {str(rec.get("object_id", "")) for rec in aion_data if rec.get("object_id")}
astropt_ids = {str(rec.get("object_id", "")) for rec in astropt_data if rec.get("object_id")}
common_ids = aion_ids & astropt_ids

print(f"\n   IDs uniques AION: {len(aion_ids)}")
print(f"   IDs uniques AstroPT: {len(astropt_ids)}")
print(f"   IDs en commun: {len(common_ids)}")

# ============================================================================
# 2. ANALYSE DU CATALOGUE FITS
# ============================================================================
print("\n" + "=" * 80)
print("2️⃣  ANALYSE DU CATALOGUE FITS")
print("=" * 80)

with fits.open(CATALOG) as hdul:
    print(f"\n📋 Structure du FITS:")
    print(f"   Nombre d'extensions: {len(hdul)}")
    for i, hdu in enumerate(hdul):
        print(f"   [{i}] {hdu.name}: {type(hdu).__name__}")
    
    # Analyser l'extension principale (généralement HDU 1)
    data = hdul[1].data
    columns = hdul[1].columns.names
    
    print(f"\n📊 Extension de données (HDU 1):")
    print(f"   Nombre de lignes: {len(data)}")
    print(f"   Nombre de colonnes: {len(columns)}")
    
    print(f"\n   📝 Toutes les colonnes:")
    for i, col in enumerate(columns):
        col_format = hdul[1].columns[col].format
        print(f"      [{i:2d}] {col:40s} (format: {col_format})")
    
    # Chercher des colonnes qui pourraient être des IDs
    print(f"\n   🔑 Colonnes potentielles pour object_id:")
    id_candidates = []
    for col in columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['id', 'target', 'object', 'source', 'spec', 'gaia']):
            id_candidates.append(col)
            # Montrer quelques valeurs
            sample_vals = [str(data[i][col]) for i in range(min(5, len(data)))]
            print(f"      ✓ {col}:")
            for j, val in enumerate(sample_vals):
                print(f"         [{j}] {val}")
    
    # Tester la correspondance avec chaque colonne candidate
    print(f"\n" + "=" * 80)
    print("3️⃣  TEST DE CORRESPONDANCE")
    print("=" * 80)
    
    # Préparer les IDs des embeddings
    embed_ids = aion_ids | astropt_ids
    print(f"\n   Total IDs dans embeddings: {len(embed_ids)}")
    print(f"   Exemples: {list(embed_ids)[:5]}")
    
    for col in id_candidates:
        print(f"\n   🧪 Test avec colonne '{col}':")
        catalog_ids = {str(row[col]) for row in data}
        
        # Tester la correspondance directe
        matches = embed_ids & catalog_ids
        match_rate = len(matches) / len(embed_ids) * 100 if embed_ids else 0
        
        print(f"      Total IDs dans catalogue: {len(catalog_ids)}")
        print(f"      Correspondances: {len(matches)}/{len(embed_ids)} ({match_rate:.1f}%)")
        
        if len(matches) > 0:
            print(f"      ✅ CORRESPONDANCE TROUVÉE !")
            print(f"      Exemples d'IDs matchés: {list(matches)[:5]}")
        else:
            # Montrer des exemples pour comparaison
            print(f"      ❌ Aucune correspondance")
            print(f"      Exemples catalogue: {list(catalog_ids)[:3]}")
            print(f"      Exemples embeddings: {list(embed_ids)[:3]}")
        
        # Tester des transformations communes
        if len(matches) == 0:
            print(f"\n      🔄 Test de transformations:")
            
            # Test: conversion int
            try:
                catalog_ids_int = {str(int(float(row[col]))) for row in data[:1000]}
                matches_int = embed_ids & catalog_ids_int
                if len(matches_int) > 0:
                    print(f"         ✅ Avec conversion int: {len(matches_int)} matches!")
            except:
                pass
            
            # Test: avec préfixe/suffixe
            for prefix in ['', '0', '00']:
                for suffix in ['', '0', '00']:
                    test_ids = {prefix + str(row[col]) + suffix for row in data[:1000]}
                    matches_test = embed_ids & test_ids
                    if len(matches_test) > 0:
                        print(f"         ✅ Avec prefix='{prefix}', suffix='{suffix}': {len(matches_test)} matches!")

print("\n" + "=" * 80)
print("✅ Investigation terminée")
print("=" * 80)
