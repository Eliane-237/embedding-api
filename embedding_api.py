import logging
import time
from typing import List

# Importations FastAPI et validation
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Importations pour le traitement des embeddings
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from contextlib import asynccontextmanager

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Modèles Pydantic
class TextsRequest(BaseModel):
    texts: List[str]

# Configuration du cycle de vie de l'application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialisation globale
    global model, tokenizer, device
    
    # Chargement du modèle
    logger.info("Chargement du modèle BGE-M3...")
    
    # Déterminer le device (GPU ou CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Utilisation de: {device}")
    
    # Charger le modèle et le tokenizer
    model_name = "BAAI/bge-m3"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()  # Mettre le modèle en mode évaluation
        logger.info(f"Modèle {model_name} chargé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        raise RuntimeError(f"Impossible de charger le modèle {model_name}: {e}")
    
    yield

# Initialisation de l'API FastAPI
app = FastAPI(lifespan=lifespan)

# Variables globales pour le modèle et le tokenizer
model = None
tokenizer = None
device = None

# Fonction pour générer les embeddings
def generate_model_embeddings(texts: List[str]) -> List[List[float]]:
    global tokenizer, model, device
    
    if not texts:
        return []
    
    # S'assurer que le modèle est chargé
    if tokenizer is None or model is None:
        raise RuntimeError("Le modèle n'est pas chargé")
    
    try:
        # Tokenization et génération d'embeddings par lots
        embeddings = []
        batch_size = 8  # Ajuster selon les capacités de mémoire

        logger.info(f"Génération d'embeddings pour {len(texts)} textes")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Tokenization
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            # Génération d'embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)
                # Utiliser les embeddings de la couche [CLS]
                batch_embeddings = model_output.last_hidden_state[:, 0].cpu().numpy()

            # Normaliser les vecteurs (optionnel mais recommandé pour la similarité cosinus)
            batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)

            embeddings.extend(batch_embeddings.tolist())

            logger.info(f"Lot {i//batch_size + 1} traité: {len(batch_texts)} textes")

        return embeddings

    except Exception as e:
        logger.error(f"Erreur lors de la génération des embeddings: {e}")
        raise RuntimeError(f"Échec de la génération d'embeddings: {e}")

# Endpoint pour générer des embeddings
@app.post("/generate_embeddings/")
async def generate_embeddings(request: TextsRequest):
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="La liste de textes ne peut pas être vide")

        logger.info(f"Requête reçue pour {len(request.texts)} textes")
        
        # Générer les embeddings
        embeddings = generate_model_embeddings(request.texts)

        if not embeddings:
            raise HTTPException(status_code=500, detail="Échec de la génération d'embeddings")

        logger.info(f"Génération réussie: {len(embeddings)} embeddings")
        logger.info(f"Dimension des embeddings: {len(embeddings[0])}")

        return {"embeddings": embeddings}

    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint pour vérifier l'état du service
@app.get("/health")
async def health_check():
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return {"status": "error", "message": "Modèle non chargé"}
    
    return {
        "status": "ok", 
        "model": "BGE-M3", 
        "device": str(device),
        "dimension": generate_model_embeddings(["Test de connexion"])[0].shape[0]
    }

# Route racine
@app.get("/")
async def root():
    return {"message": "Service d'Embeddings BGE-M3 opérationnel"}

# Lancement de l'API si exécutée directement
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)