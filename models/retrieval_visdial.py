import os
import json
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import BlipProcessor, BlipForImageTextRetrieval, BlipForConditionalGeneration
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import numpy as np
import config.config as config
import open_clip

# Helper Dataset class remains the same as your provided code
class ImageCorpusDataset(torch.utils.data.Dataset):
    """Dataset class for the corpus images."""
    def __init__(self, corpus_path, img_root_path, preprocessor, model_name):
        with open(corpus_path) as f:
            self.image_files = json.load(f)
        self.img_root_path = img_root_path
        self.preprocessor = preprocessor
        self.model_name = model_name
        self.img_id_to_idx = {os.path.splitext(fname)[0]: i for i, fname in enumerate(self.image_files)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        img_path = os.path.join(self.img_root_path, self.image_files[i])
        image = Image.open(img_path).convert("RGB")
        
        if self.model_name == 'BLIP':
            processed = self.preprocessor(images=image, return_tensors="pt")
            pixel_values = processed.pixel_values.squeeze(0)
        elif self.model_name == 'CLIP':
            pixel_values = self.preprocessor(image)
        else:
            raise ValueError(f"Model {self.model_name} not supported in Dataset.")
            
        return {'pixel_values': pixel_values, 'img_id_index': i}


class InteractiveRetrievalModel:


    TOP_N_NUM = config.TOP_N_NUM
    

    def __init__(self,
                 img_root_path,
                 corpus_json_path,
                 img_embeddings_path,
                 model_name='CLIP'):
        # assert model_name in ['BLIP', 'CLIP'], "Model must be either 'BLIP' or 'CLIP'."

        self.img_root_path = img_root_path
        self.corpus_json_path = corpus_json_path
        self.model_name = model_name
        self.img_embeddings_path = img_embeddings_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load Models
        if self.model_name == 'BLIP':
            self.retrieval_model = BlipForImageTextRetrieval.from_pretrained(self.BLIP_ITR_MODEL_PATH).to(self.device)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(self.BLIP_CAPTION_MODEL_PATH).to(self.device)
            self.processor = BlipProcessor.from_pretrained(self.BLIP_ITR_MODEL_PATH)
        elif self.model_name == 'CLIP':
            print("Loading CLIP model: ViT-B-32 with laion2b_s34b_b79k weights...")
            self.retrieval_model, _, self.processor = open_clip.create_model_and_transforms(
                'ViT-B-32',
                pretrained='laion2b_s34b_b79k',
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        else:
            raise NotImplementedError("Only BLIP and CLIP models are supported.")

        self.all_img_embeddings = None
        self.img_id_to_idx = None
        self.idx_to_img_id = None 

        # Load or compute image embeddings
        self.index_corpus()

    def index_corpus(self):
        if os.path.exists(self.img_embeddings_path):
            print(f"Loading cached corpus embeddings from {self.img_embeddings_path}...")
            cached_data = torch.load(self.img_embeddings_path)
            self.all_img_embeddings = cached_data['embeddings'].to(self.device)
            self.img_id_to_idx = cached_data['id_to_idx']
            self.idx_to_img_id = {v: k for k, v in self.img_id_to_idx.items()}
            self.all_img_embeddings = F.normalize(self.all_img_embeddings, dim=-1)
            return

        print("Cached corpus not found. Indexing images from scratch...")

        corpus_dataset = ImageCorpusDataset(self.corpus_json_path, self.img_root_path, self.processor, self.model_name)
        self.img_id_to_idx = corpus_dataset.img_id_to_idx
        self.idx_to_img_id = {v: k for k, v in self.img_id_to_idx.items()}

        dataloader = torch.utils.data.DataLoader(corpus_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

        all_embeddings_list = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Indexing corpus with {self.model_name}"):
                pixel_values = batch['pixel_values'].to(self.device)

                if self.model_name == 'BLIP':
                    vision_outputs = self.retrieval_model.vision_model(pixel_values=pixel_values)
                    pooled_output = vision_outputs[0][:, 0, :]
                    image_embed = self.retrieval_model.vision_proj(pooled_output)
                elif self.model_name == 'CLIP':
                    image_embed = self.retrieval_model.encode_image(pixel_values)

                image_embed = F.normalize(image_embed, dim=-1)
                all_embeddings_list.append(image_embed.cpu())

        self.all_img_embeddings = torch.cat(all_embeddings_list, dim=0).to(self.device)

        print(f"Saving indexed corpus to {self.img_embeddings_path}...")
        torch.save({'embeddings': self.all_img_embeddings.cpu(), 'id_to_idx': self.img_id_to_idx}, self.img_embeddings_path)
        print("Corpus indexing complete.")
    
            
    def get_text_embedding(self, text):
        with torch.no_grad():
            if self.model_name == 'BLIP':
                inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
                text_outputs = self.retrieval_model.text_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                text_embed = self.retrieval_model.text_proj(text_outputs[0][:, 0, :])
            elif self.model_name == 'CLIP':
                tokenized_text = self.tokenizer([text]).to(self.device)
                text_embed = self.retrieval_model.encode_text(tokenized_text)

            return F.normalize(text_embed, dim=-1)
        
    def get_image_embeddings_by_ids(self, image_ids):
        """Fetches the normalized embeddings for a list of image IDs."""
        if not image_ids:
            return None
        indices = [self.img_id_to_idx[img_id] for img_id in image_ids if img_id in self.img_id_to_idx]
        if not indices:
            return None
        return self.all_img_embeddings[indices]

    def find_nearest_neighbors(self, seed_embeddings, num_neighbors):
        """
        Finds the nearest neighbors in the corpus for a given set of seed embeddings.
        To define the 'region of irrelevance', we average the seed embeddings first.
        """
        if seed_embeddings is None or seed_embeddings.nelement() == 0:
            return []
        
        avg_seed_embedding = seed_embeddings.mean(dim=0, keepdim=True)
        similarities = (avg_seed_embedding @ self.all_img_embeddings.T).squeeze()
        
        k = min(num_neighbors, len(self.idx_to_img_id))
        top_n_indices = torch.topk(similarities, k=k, largest=True).indices
        
        neighbor_ids = [self.idx_to_img_id[idx.item()] for idx in top_n_indices]
        return neighbor_ids
            
    # --- MODIFIED METHOD ---
    def get_top_n_similar(self, query_caption, gt_img_id=None, exclude_ids=None, penalized_ids=None, penalty_value=0.5):
        """
        Computes similarity and supports two tiers of memory:
        1.  Hard exclusion for `exclude_ids`.
        2.  Soft penalty for `penalized_ids`.
        3.  Calculates rank of `gt_img_id` if provided.
        """
        if self.all_img_embeddings is None:
            raise RuntimeError("Corpus is not indexed. Please run `index_corpus()` first.")
        
        text_embed = self.get_text_embedding(query_caption)
        image_embeddings_on_device = self.all_img_embeddings
        
        if self.model_name == 'CLIP':
            text_embed = text_embed.to(torch.float32)
            image_embeddings_on_device = image_embeddings_on_device.to(torch.float32)
            logit_scale = self.retrieval_model.logit_scale.exp()
            scores = logit_scale * (text_embed @ image_embeddings_on_device.T)
        else:
            scores = text_embed @ image_embeddings_on_device.T

        scores = scores.squeeze(0).clone() # Use clone to avoid modifying original scores if needed elsewhere
        
        # --- START: TWO-TIERED MEMORY MECHANISM ---
        if penalized_ids:
            penalized_set = set(penalized_ids)
            if exclude_ids:
                penalized_set -= set(exclude_ids)
            
            if penalized_set:
                penalized_indices = [self.img_id_to_idx[img_id] for img_id in penalized_set if img_id in self.img_id_to_idx]
                if penalized_indices:
                    penalized_indices_tensor = torch.tensor(penalized_indices, device=scores.device, dtype=torch.long)
                    scores.index_add_(0, penalized_indices_tensor, torch.full_like(penalized_indices_tensor, -penalty_value, dtype=scores.dtype))

        if exclude_ids:
            exclude_indices = [self.img_id_to_idx[img_id] for img_id in exclude_ids if img_id in self.img_id_to_idx]
            if exclude_indices:
                exclude_indices_tensor = torch.tensor(exclude_indices, device=scores.device, dtype=torch.long)
                scores.index_fill_(0, exclude_indices_tensor, -float('inf'))
        # --- END: TWO-TIERED MEMORY MECHANISM ---

        # --- START: GT RANK CALCULATION ---
        gt_rank = -1
        if gt_img_id:
            gt_original_idx = self.img_id_to_idx.get(gt_img_id)
            if gt_original_idx is not None:
                # Sort all scores to find the rank. Excluded items with -inf will be at the end.
                _, sorted_indices = torch.sort(scores, descending=True)
                rank_tensor = (sorted_indices == gt_original_idx).nonzero(as_tuple=True)[0]
                if rank_tensor.numel() > 0:
                    gt_rank = rank_tensor.item() + 1
        # --- END: GT RANK CALCULATION ---

        num_images_in_corpus = scores.shape[0]
        k = min(self.TOP_N_NUM, num_images_in_corpus)
        top_scores, top_indices = torch.topk(scores, k)
        
        top_indices = top_indices.detach().cpu().numpy()
        top_scores = top_scores.detach().cpu().numpy()
        
        top_n_ids = [self.idx_to_img_id[i] for i in top_indices]
        top_n_sims = {self.idx_to_img_id[i]: float(s) for i, s in zip(top_indices, top_scores)}
        
        return top_n_ids, top_n_sims, gt_rank

    def get_top_n_similar_full_corpus(self, query_caption, gt_img_id):
        """
        Performs a simple retrieval on the entire image corpus without any memory
        or shrinking search space logic. Returns the top-K IDs and the GT rank.
        """
        text_embed = self.get_text_embedding(query_caption)
        image_embeddings_on_device = self.all_img_embeddings
        
        if self.model_name == 'CLIP':
            text_embed = text_embed.to(torch.float32)
            image_embeddings_on_device = image_embeddings_on_device.to(torch.float32)
            logit_scale = self.retrieval_model.logit_scale.exp()
            scores = logit_scale * (text_embed @ image_embeddings_on_device.T)
        else:
            scores = text_embed @ image_embeddings_on_device.T
        scores = scores.squeeze(0)

        num_in_corpus = scores.shape[0]
        top_scores_all, top_indices_all = torch.topk(scores, k=num_in_corpus)

        gt_rank = -1
        gt_original_idx = self.img_id_to_idx.get(gt_img_id)

        if gt_original_idx is not None:
            rank_tensor = (top_indices_all == gt_original_idx).nonzero(as_tuple=True)[0]
            if rank_tensor.numel() > 0:
                gt_rank = rank_tensor.item() + 1

        top_k_ids = [self.idx_to_img_id[idx.item()] for idx in top_indices_all[:self.TOP_N_NUM]]

        return top_k_ids, gt_rank


    def get_top_n_similar_from_subset(self, query_caption, search_space_indices, shrink_top_n, gt_img_id, exclude_ids=None, penalized_ids=None, penalty_value=0.5):
        """
        Performs retrieval on a dynamic subset of the corpus and returns both the top-K
        results for the current round and the indices for the next round's even smaller subset.
        """
        search_space_indices_tensor = torch.tensor(search_space_indices, device=self.device, dtype=torch.long)
        subset_embeddings = self.all_img_embeddings.index_select(0, search_space_indices_tensor)

        original_idx_to_subset_idx = {original_i: subset_i for subset_i, original_i in enumerate(search_space_indices)}

        text_embed = self.get_text_embedding(query_caption)
        
        if self.model_name == 'CLIP':
            text_embed = text_embed.to(torch.float32)
            subset_embeddings = subset_embeddings.to(torch.float32)
            logit_scale = self.retrieval_model.logit_scale.exp()
            scores = logit_scale * (text_embed @ subset_embeddings.T)
        else:
            scores = text_embed @ subset_embeddings.T
        scores = scores.squeeze(0)

        if penalized_ids:
            penalized_subset_indices = [original_idx_to_subset_idx[self.img_id_to_idx[pid]] for pid in penalized_ids if self.img_id_to_idx.get(pid) in original_idx_to_subset_idx]
            if penalized_subset_indices:
                scores.index_add_(0, torch.tensor(penalized_subset_indices, device=self.device), torch.tensor([-penalty_value] * len(penalized_subset_indices), device=self.device))

        if exclude_ids:
            excluded_subset_indices = [original_idx_to_subset_idx[self.img_id_to_idx[eid]] for eid in exclude_ids if self.img_id_to_idx.get(eid) in original_idx_to_subset_idx]
            if excluded_subset_indices:
                scores.index_fill_(0, torch.tensor(excluded_subset_indices, device=self.device), -float('inf'))

        num_in_subset = subset_embeddings.shape[0]
        k_for_ranking = min(num_in_subset, 50000)
        
        top_scores_all, top_subset_indices_all = torch.topk(scores, k=k_for_ranking)

        gt_rank = -1
        gt_original_idx = self.img_id_to_idx.get(gt_img_id)

        if gt_original_idx in search_space_indices:
            gt_subset_idx_tensor = torch.tensor(list(search_space_indices).index(gt_original_idx), device=self.device)
            rank_tensor = (top_subset_indices_all == gt_subset_idx_tensor).nonzero(as_tuple=True)[0]
            if rank_tensor.numel() > 0:
                gt_rank = rank_tensor.item() + 1
            
        top_n_ids = [self.idx_to_img_id[search_space_indices[idx.item()]] for idx in top_subset_indices_all[:self.TOP_N_NUM]]
        next_search_space_indices = [search_space_indices[idx.item()] for idx in top_subset_indices_all[:shrink_top_n]]

        return top_n_ids, next_search_space_indices, gt_rank
    
    
    @torch.no_grad()
    def generate_caption(self, image_path=str):
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
        out = self.caption_model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
