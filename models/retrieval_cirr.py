# utils/retrieval_model_cirr.py
import os
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip 
from tqdm import tqdm
import logging

class RetrievalModel:
    """A CLIP-only retrieval model for CIRR."""
    def __init__(self, cache_dir='./cache'):
        self.device = "cuda:4" if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        logging.info(f"Initializing CLIP RetrievalModel on {self.device}")

        self.model, _, self.processor = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device
            # 'ViT-L-14', pretrained='laion2B-s32B-b82K', device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        self.img_embeddings = None
        self.img_id_to_idx = None
        self.idx_to_img_id = None

    def _get_embedding_path(self, type, dataset_name):
        target_dir = os.path.join(self.cache_dir, 'subsets') if 'subset' in dataset_name else self.cache_dir
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(target_dir, f"CLIP_{dataset_name}_{type}_embeddings.pth")

    @torch.no_grad()
    def get_text_embedding(self, text_list):
        self.model.eval()
        if not text_list: return None
        tokenized_text = self.tokenizer(text_list).to(self.device)
        text_embed = self.model.encode_text(tokenized_text)
        return F.normalize(text_embed, dim=-1)

    @torch.no_grad()
    def get_image_embeddings(self, image_paths, dataset_name):
        self.model.eval()
        embedding_path = self._get_embedding_path('image', dataset_name)
        if os.path.exists(embedding_path):
            cached_data = torch.load(embedding_path, map_location=self.device)
            self.img_embeddings = cached_data['embeddings'].to(self.device)
            self.img_id_to_idx = cached_data['id_to_idx']
            self.idx_to_img_id = {v: k for k, v in self.img_id_to_idx.items()}
            return

        if not image_paths:
            logging.error(f"No image paths provided and no cache found for {dataset_name}.")
            return

        logging.info(f"No cache found for '{dataset_name}'. Indexing images...")
        # This implementation processes images in a single-threaded loop
        dataset = torch.utils.data.TensorDataset(torch.arange(len(image_paths)))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128)

        all_embeddings = []
        for (indices,) in tqdm(dataloader, desc=f"Generating {dataset_name} embeddings", leave=False):
            batch_images = [self.processor(Image.open(image_paths[i]).convert("RGB")) for i in indices]
            pixel_values = torch.stack(batch_images).to(self.device)
            image_embed = self.model.encode_image(pixel_values)
            all_embeddings.append(F.normalize(image_embed, dim=-1).cpu())
        
        self.img_embeddings = torch.cat(all_embeddings).to(self.device)
        self.img_id_to_idx = {os.path.basename(path): i for i, path in enumerate(image_paths)}
        self.idx_to_img_id = {i: os.path.basename(path) for i, path in enumerate(image_paths)}
        
        logging.info(f"Saving image embeddings to {embedding_path}")
        torch.save({'embeddings': self.img_embeddings.cpu(), 'id_to_idx': self.img_id_to_idx}, embedding_path)

    def perform_retrieval(self, query_embeddings, target_embeddings):
        if query_embeddings is None or target_embeddings is None: return None
        query_embeddings = query_embeddings.to(torch.float32)
        target_embeddings = target_embeddings.to(torch.float32)
        logit_scale = self.model.logit_scale.exp()
        return logit_scale * (query_embeddings @ target_embeddings.T)

    @staticmethod
    def normalize(scores):
        return (scores - scores.min()) / (scores.max() - scores.min())