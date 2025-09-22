# eval_cirr.py
import os
import json
import logging
import argparse
import torch
from tqdm import tqdm
import time
import multiprocessing

# --- Project-specific Imports ---
from models.retrieval_cirr import RetrievalModel
from models.mllm_reranker import MLLMReranker
import models.config.config as config

# --- Logging Configuration ---
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f"eval_cirr_{int(time.time())}.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# --- Global Variables for Worker Processes ---
worker_retrieval_model = None
worker_reranker = None
worker_args = None

def initialize_worker(args):
    """Initializes models for each worker process."""
    global worker_retrieval_model, worker_reranker, worker_args
    worker_args = args
    logger.info(f"Initializing worker {os.getpid()} for mode: {args.mode}")

    worker_retrieval_model = RetrievalModel(
        model_name=args.model,
        blip_itr_model_path=config.BLIP_ITR_MODEL_PATH
    )
    
    if args.mode == 'normal':
        logger.info(f"Worker {os.getpid()} loading 'normal' mode image embeddings from cache...")
        worker_retrieval_model.get_image_embeddings(image_paths=[], dataset_name="cirr_all")
        if worker_retrieval_model.img_embeddings is None:
             logger.error(f"Worker {os.getpid()} FAILED to load 'normal' mode image embeddings.")
        else:
             logger.info(f"Worker {os.getpid()} successfully loaded 'normal' mode image embeddings.")

    if args.rerank:
        worker_reranker = MLLMReranker()

def process_entry_worker(entry):
    """
    Worker function to process a single entry from the CIRR dataset.
    """
    pair_id = entry['pairid']
    query_text = entry.get("Target Image Description")

    if not query_text or query_text in ["GENERATION_FAILED", "PARSE_FAILED", "IMAGE_NOT_FOUND"]:
        logger.warning(f"Skipping pair_id {pair_id} due to invalid Target Image Description: '{query_text}'")
        return None

    # --- Subset Mode Logic ---
    if worker_args.mode == 'subset':
        subset_id = entry['img_set']['id']
        all_members = entry['img_set']['members']
        reference_id = entry['reference']
        
        if worker_args.rerank and worker_reranker:
            candidate_ids = [m for m in all_members if m != reference_id]
            candidate_paths = [os.path.join(worker_args.image_dir, f"{img_id}.png") for img_id in candidate_ids]

            job_id = f"subset_{pair_id}"
            reranked_ids = worker_reranker.rerank_with_path(
                query=query_text,
                candidate_paths=candidate_paths,
                job_id=job_id,
                top_k_output=3
            )
            final_ids = reranked_ids[:3]
        else:
            dataset_name = f"subset_{subset_id}"
            worker_retrieval_model.get_image_embeddings(image_paths=[], dataset_name=dataset_name)
            
            if worker_retrieval_model.img_embeddings is None:
                logger.error(f"Could not load cached embeddings for subset {subset_id} (pair_id {pair_id}). Skipping.")
                return None

            text_embedding = worker_retrieval_model.get_text_embedding([query_text])
            scores = worker_retrieval_model.perform_retrieval(text_embedding, worker_retrieval_model.img_embeddings).squeeze(0)

            reference_id_with_ext = f"{reference_id}.png"
            if reference_id_with_ext in worker_retrieval_model.img_id_to_idx:
                reference_idx = worker_retrieval_model.img_id_to_idx[reference_id_with_ext]
                scores[reference_idx] = -torch.inf
            else:
                logger.warning(f"Reference image {reference_id} not found in index for subset {subset_id}. Cannot exclude.")

            _, sorted_indices = torch.sort(scores, descending=True)
            
            ranked_ids_with_ext = [worker_retrieval_model.idx_to__img_id[idx.item()] for idx in sorted_indices]
            final_ids_with_ext = ranked_ids_with_ext[:3]
            final_ids = [os.path.splitext(p)[0] for p in final_ids_with_ext]

        return pair_id, final_ids

    # --- Normal Mode Logic ---
    elif worker_args.mode == 'normal':
        if worker_retrieval_model.img_embeddings is None:
            logger.error(f"Cannot process pair_id {pair_id} because image embeddings are not loaded.")
            return None

        query_embedding = worker_retrieval_model.get_text_embedding([query_text])
        scores = worker_retrieval_model.perform_retrieval(query_embedding, worker_retrieval_model.img_embeddings).squeeze(0)
        
        _, sorted_indices = torch.sort(scores, descending=True)
        
        initial_retrieved_ids = [worker_retrieval_model.idx_to_img_id[idx.item()] for idx in sorted_indices]

        if worker_args.rerank and worker_reranker:
            # --- FIX: Get the number of candidates directly from the config file ---
            num_candidates_for_rerank = config.TOP_N_FOR_RERANK
            
            candidates_for_rerank_ids = [os.path.splitext(p)[0] for p in initial_retrieved_ids[:num_candidates_for_rerank]]
            candidate_paths = [os.path.join(worker_args.image_dir, f"{img_id}.png") for img_id in candidates_for_rerank_ids]
            job_id = f"normal_{pair_id}"

            reranked_top_group = worker_reranker.rerank_with_path(
                query=query_text,
                candidate_paths=candidate_paths,
                job_id=job_id,
                top_k_output=10 
            )

            if reranked_top_group is None:
                logger.error(f"Reranking failed for pair_id {pair_id}. Returning empty list.")
                return pair_id, ids_for_next_stage[:worker_args.top_k_save]
            # --- FIX: Use the same config value to slice the remaining list ---
            remaining_initial_ids = [os.path.splitext(p)[0] for p in initial_retrieved_ids[num_candidates_for_rerank:]]
            final_ids = (reranked_top_group + remaining_initial_ids)[:worker_args.top_k_save]
        else:
            final_ids = [os.path.splitext(p)[0] for p in initial_retrieved_ids[:worker_args.top_k_save]]

        return pair_id, final_ids

class CIRREvaluationPipeline:
    def __init__(self, args):
        self.args = args

    def _prepare_data(self):
        try:
            with open(self.args.input_json_path, 'r') as f:
                queries = json.load(f)
            logger.info(f"Loaded {len(queries)} entries from {self.args.input_json_path}")
            return queries
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not load or parse input JSON file: {e}")
            return None

    def evaluate(self):
        queries = self._prepare_data()
        if not queries:
            return

        # --- Pre-computation Step ---
        main_process_model = RetrievalModel(self.args.model, config.BLIP_ITR_MODEL_PATH)
        
        if self.args.mode == 'normal':
            logger.info("Mode 'normal': Pre-caching image embeddings for the entire dataset...")
            all_image_paths = [os.path.join(self.args.image_dir, f) for f in os.listdir(self.args.image_dir) if f.lower().endswith('.png')]
            main_process_model.get_image_embeddings(sorted(list(set(all_image_paths))), "cirr_all")
        
        elif self.args.mode == 'subset':
            logger.info("Mode 'subset': Pre-caching embeddings for each unique image subset...")
            unique_subsets = {}
            for q in queries:
                subset_id = q['img_set']['id']
                if subset_id not in unique_subsets:
                    unique_subsets[subset_id] = q['img_set']['members']
            
            logger.info(f"Found {len(unique_subsets)} unique subsets to cache.")
            for subset_id, members in tqdm(unique_subsets.items(), desc="Caching subsets"):
                image_paths = [os.path.join(self.args.image_dir, f"{m}.png") for m in members]
                dataset_name = f"subset_{subset_id}"
                main_process_model.get_image_embeddings(image_paths, dataset_name)

        del main_process_model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logger.info("Pre-computation and caching complete. Starting workers.")
        
        # --- Main Processing Loop ---
        results = {}
        with multiprocessing.Pool(self.args.num_workers, initialize_worker, (self.args,)) as pool:
            with tqdm(total=len(queries), desc=f"Processing CIRR - {self.args.mode} mode") as pbar:
                for result in pool.imap_unordered(process_entry_worker, queries):
                    if result:
                        pair_id, ranked_list = result
                        results[str(pair_id)] = ranked_list
                    pbar.update(1)

        output_data = {
            "version": "rc2",
            "metric": f"recall_{self.args.mode}",
            **results
        }

        mode_str = f"{self.args.model}_{self.args.mode}"
        if self.args.rerank:
            mode_str += "_rerank"
        
        output_filename = f"retrieval_results_cirr_0912_{mode_str}.json"
        output_path = os.path.join("output", output_filename)

        try:
            with open(output_path, 'w') as f:
                json.dump(output_data, f)
            logger.info(f"Successfully saved results for {len(results)} queries to {output_path}")
        except IOError as e:
            logger.error(f"Failed to save final results: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval evaluation on the CIRR dataset.")
    parser.add_argument('--mode', type=str, default='normal', choices=['normal', 'subset'], help="Evaluation mode.")
    parser.add_argument('--input_json_path', type=str, required=True, help='Path to the input JSON file with target_captions.')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing CIRR images.')
    parser.add_argument('--model', type=str, default='CLIP', choices=['BLIP', 'CLIP'], help="The vision-language model for retrieval.")
    parser.add_argument('--rerank', action='store_true', help="Enable MLLM-based reranking.")
    parser.add_argument('--num-workers', type=int, default=5, help='Number of parallel workers.')
    parser.add_argument('--top_k_save', type=int, default=50, help='Number of top candidates to save for "normal" mode.')
    parser.add_argument('--top_k_rerank', type=int, default=10, help='Number of top candidates to ask for from the reranker in "normal" mode.')
    args = parser.parse_args()

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        logger.warning("Multiprocessing start method 'spawn' already set.")
    
    pipeline = CIRREvaluationPipeline(args)
    pipeline.evaluate()
