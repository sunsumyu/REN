@echo off
if exist local_rag.db del /f /q local_rag.db
if exist local_rag_vector.index del /f /q local_rag_vector.index
python scripts/import_clinical_pathways.py --max_docs 30
