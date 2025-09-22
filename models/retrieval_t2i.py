import os
import json
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import BlipProcessor, BlipForImageTextRetrieval
from tqdm import tqdm
import open_clip
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageCorpusDataset(torch.utils.data.Dataset):
    """
    Dataset for loading and preprocessing images from the MSCOCO corpus.
    It expects a list of image file paths.
    """
    def __init__(self, image_paths, preprocessor, model_name):
        self.image_paths = image_paths
        self.preprocessor = preprocessor
        self.model_name = model_name
        # Create a mapping from image filename to index
        self.img_id_to_idx = {os.path.basename(path): i for i, path in enumerate(self.image_paths)}
        self.idx_to_img_id = {i: os.path.basename(path) for i, path in enumerate(self.image_paths)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image = Image.open(self.image_paths[i]).convert("RGB")
        
        if self.model_name == 'BLIP':
            # Preprocess the image for BLIP
            processed = self.preprocessor(images=image, return_tensors="pt")
            pixel_values = processed.pixel_values.squeeze(0)
        elif self.model_name == 'CLIP':
            # Preprocess the image for CLIP
            pixel_values = self.preprocessor(image)
        else:
            raise ValueError(f"Model {self.model_name} not supported in Dataset.")
            
        return {'pixel_values': pixel_values, 'img_id_index': i}

class RetrievalModel:
    """
    A simplified retrieval model for MSCOCO text-image retrieval.
    It handles embedding generation, retrieval, and orchestrates reranking.
    """
    def __init__(self, model_name, blip_itr_model_path, cache_dir='./cache'):
        assert model_name in ['BLIP', 'CLIP'], "Model must be either 'BLIP' or 'CLIP'."
        self.model_name = model_name
        self.device = "cuda:3" 
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        logging.info(f"Initializing RetrievalModel with {self.model_name} on {self.device}")

        # Load the specified vision-language model
        if self.model_name == 'BLIP':
            self.model = BlipForImageTextRetrieval.from_pretrained(blip_itr_model_path).to(self.device)
            self.processor = BlipProcessor.from_pretrained(blip_itr_model_path)
        elif self.model_name == 'CLIP':
            logging.info("Loading CLIP model: ViT-B-32 with laion2b_s34b_b79k weights...")
            self.model, _, self.processor = open_clip.create_model_and_transforms(
                'ViT-B-32',
                pretrained='laion2b_s34b_b79k',
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        self.img_embeddings = None
        self.text_embeddings = None
        self.img_id_to_idx = None
        self.idx_to_img_id = None

    def _get_embedding_path(self, type, dataset_name):
        return os.path.join(self.cache_dir, f"{self.model_name}_{dataset_name}_{type}_embeddings.pth")

    @torch.no_grad()
    def get_text_embedding(self, text_list):
        """Generates embeddings for a list of text strings."""
        self.model.eval()
        if self.model_name == 'BLIP':
            inputs = self.processor(text=text_list, return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_outputs = self.model.text_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            text_embed = self.model.text_proj(text_outputs[0][:, 0, :])
        elif self.model_name == 'CLIP':
            tokenized_text = self.tokenizer(text_list).to(self.device)
            text_embed = self.model.encode_text(tokenized_text)
        
        return F.normalize(text_embed, dim=-1)

    @torch.no_grad()
    def get_image_embeddings(self, image_paths, dataset_name):
        """Generates or loads cached embeddings for a list of image paths."""
        self.model.eval()
        embedding_path = self._get_embedding_path('image', dataset_name)

        if os.path.exists(embedding_path):
            logging.info(f"Loading cached image embeddings from {embedding_path}")
            cached_data = torch.load(embedding_path)
            self.img_embeddings = cached_data['embeddings'].to(self.device)
            self.img_id_to_idx = cached_data['id_to_idx']
            self.idx_to_img_id = {v: k for k, v in self.img_id_to_idx.items()}
            return

        logging.info("No cached image embeddings found. Indexing images...")
        dataset = ImageCorpusDataset(image_paths, self.processor, self.model_name)
        self.img_id_to_idx = dataset.img_id_to_idx
        self.idx_to_img_id = dataset.idx_to_img_id
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

        all_embeddings = []
        for batch in tqdm(dataloader, desc=f"Generating image embeddings with {self.model_name}"):
            pixel_values = batch['pixel_values'].to(self.device)
            if self.model_name == 'BLIP':
                vision_outputs = self.model.vision_model(pixel_values=pixel_values)
                pooled_output = vision_outputs[0][:, 0, :]
                image_embed = self.model.vision_proj(pooled_output)
            elif self.model_name == 'CLIP':
                image_embed = self.model.encode_image(pixel_values)
            
            all_embeddings.append(F.normalize(image_embed, dim=-1).cpu())
        
        self.img_embeddings = torch.cat(all_embeddings).to(self.device)
        logging.info(f"Saving image embeddings to {embedding_path}")
        torch.save({'embeddings': self.img_embeddings.cpu(), 'id_to_idx': self.img_id_to_idx}, embedding_path)

    def perform_retrieval(self, query_embeddings, target_embeddings):
        """
        Calculates similarity scores between query and target embeddings.
        """
        # Ensure embeddings are float32 for stable matrix multiplication, especially for CLIP
        query_embeddings = query_embeddings.to(torch.float32)
        target_embeddings = target_embeddings.to(torch.float32)

        if self.model_name == 'CLIP':
            # Use CLIP's learned temperature parameter
            logit_scale = self.model.logit_scale.exp()
            scores = logit_scale * (query_embeddings @ target_embeddings.T)
        else:
            scores = query_embeddings @ target_embeddings.T
        
        return scores
