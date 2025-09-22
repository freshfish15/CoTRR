import os
import json
import logging
import argparse
import numpy as np
import torch
from tqdm import tqdm
import time
import multiprocessing
import pandas as pd

# --- Project-specific Imports ---
from models.retrieval_t2i import RetrievalModel
from models.mllm_reranker import MLLMReranker
import models.config.config as config

# --- Logging Configuration ---
log_dir = "./logs_flickr30k"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f"eval_flickr30k_{int(time.time())}.log")

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
worker_config = {}

def initialize_retrieval_worker(args):
    """Initializes the retrieval model for each worker process."""
    global worker_retrieval_model, worker_config
    worker_config.update(vars(args))
    logger.info(f"Initializing retrieval worker {os.getpid()}...")
    worker_retrieval_model = RetrievalModel(
        model_name=args.model,
        blip_itr_model_path=config.BLIP_ITR_MODEL_PATH
    )
    try:
        image_dir = os.path.abspath(args.image_dir)
        if not os.path.isdir(image_dir):
            logger.error(f"Local image directory not found at {image_dir}")
            return
        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        # Use a unique name for the Flickr30K embedding cache
        worker_retrieval_model.get_image_embeddings(image_files, "flickr30k")
    except Exception as e:
        logger.error(f"Could not initialize embeddings in worker {os.getpid()}: {e}", exc_info=True)

def initialize_rerank_worker(args):
    """Initializes the MLLM reranker for each reranking worker process."""
    global worker_reranker, worker_config
    worker_config.update(vars(args))
    logger.info(f"Initializing rerank worker {os.getpid()}...")
    worker_reranker = MLLMReranker()

def initial_retrieval_worker(entry):
    """Performs only the initial retrieval in a worker process."""
    if worker_retrieval_model is None or worker_retrieval_model.img_embeddings is None:
        return None
    try:
        query_embedding = worker_retrieval_model.get_text_embedding([entry['text']])
        scores = worker_retrieval_model.perform_retrieval(query_embedding, worker_retrieval_model.img_embeddings).squeeze(0)
        _, sorted_indices = torch.sort(scores, descending=True)
        initial_retrieved_ids = [worker_retrieval_model.idx_to_img_id[idx.item()] for idx in sorted_indices]
        initial_gt_rank = initial_retrieved_ids.index(entry['image_id']) + 1 if entry['image_id'] in initial_retrieved_ids else -1
        return {
            "caption_text": entry['text'],
            "gt_image_id": entry['image_id'],
            "gt_rank": initial_gt_rank,
            "initial_retrieved_ids": initial_retrieved_ids,
            "image_dir_path": entry['image_dir_path']
        }
    except Exception as e:
        logger.error(f"Error during initial retrieval for query '{entry['text'][:50]}...': {e}", exc_info=True)
        return None

def rerank_worker(entry):
    """Performs the reranking for a single entry."""
    if worker_reranker is None:
        return None
    
    rerank_candidate_filenames = entry['initial_retrieved_ids'][:worker_reranker.rerank_candidate_count]
    candidate_full_paths = [os.path.join(entry['image_dir_path'], fname) for fname in rerank_candidate_filenames]
    job_id = f"t2i_{worker_config['model']}_{entry['gt_image_id'].replace('.jpg', '')}_{hash(entry['caption_text']) % 1000}"
    
    reranked_ids = worker_reranker.rerank_with_path(
        query=entry['caption_text'],
        candidate_paths=candidate_full_paths,
        job_id=job_id
    )

    if reranked_ids is None:
        entry["rerank_status"] = "failed"
        entry["gt_rank_after_rerank"] = -1
        entry["final_gt_rank_for_metric"] = entry["gt_rank"]
        entry["top_k_retrieved_ids"] = entry['initial_retrieved_ids'][:worker_config['top_k']]
    else:
        entry["rerank_status"] = "success"
        remaining_ids = [img_id for img_id in entry['initial_retrieved_ids'] if img_id not in reranked_ids]
        final_retrieved_ids = reranked_ids + remaining_ids
        final_gt_rank = final_retrieved_ids.index(entry['gt_image_id']) + 1 if entry['gt_image_id'] in final_retrieved_ids else -1
        entry["gt_rank_after_rerank"] = final_gt_rank
        entry["final_gt_rank_for_metric"] = final_gt_rank
        entry["top_k_retrieved_ids"] = final_retrieved_ids[:worker_config['top_k']]
    
    del entry['initial_retrieved_ids']
    del entry['image_dir_path']
    return entry

class Flickr30kEvaluationPipeline:
    def __init__(self, args):
        self.args = args
        self.model_name = args.model
        self.use_rerank = args.rerank

    def calculate_recall_at_k(self, ranks, k_values):
        total_queries = len(ranks)
        if total_queries == 0: return {f'R@{k}': 0.0 for k in k_values}
        valid_ranks = np.array([r for r in ranks if r > 0])
        return {f'R@{k}': (np.sum(valid_ranks <= k) / total_queries) * 100 for k in k_values}

    def _load_progress(self, log_path):
        if not os.path.exists(log_path): return [], set()
        try:
            with open(log_path, 'r') as f: existing_results = json.load(f)
            completed_queries = {res['caption_text'] for res in existing_results if not self.use_rerank or res.get("rerank_status") == "success"}
            results_to_keep = [res for res in existing_results if res['caption_text'] not in completed_queries]
            logger.info(f"Loaded {len(existing_results)} total results. Found {len(completed_queries)} already completed queries.")
            return results_to_keep, completed_queries
        except (json.JSONDecodeError, IOError):
            logger.warning(f"Could not parse {log_path}. Starting fresh.")
            return [], set()

    def _save_json_results(self, log_path, all_results):
        temp_log_path = log_path + '.tmp'
        try:
            with open(temp_log_path, 'w') as f: json.dump(all_results, f, indent=4)
            os.replace(temp_log_path, log_path)
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to save JSON results to {log_path}. Error: {e}")

    def _prepare_data(self):
        image_dir = os.path.abspath(self.args.image_dir)
        csv_path = os.path.abspath(self.args.csv_path)
        if not os.path.isdir(image_dir) or not os.path.isfile(csv_path):
            logger.error(f"Path does not exist: {image_dir if not os.path.isdir(image_dir) else csv_path}")
            return None, None
        logger.info(f"Loading data from local Flickr30K CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Filter for the 'test' split
        logger.info(f"Original dataset has {len(df)} entries.")
        if 'split' in df.columns:
            df = df[df['split'] == 'test'].reset_index(drop=True)
            logger.info(f"Filtered for 'test' split. Found {len(df)} test entries.")
        else:
            logger.warning("'split' column not found in CSV. Processing all entries.")

        # Sample from the filtered dataframe of images
        if self.args.sample_size:
            if self.args.sample_size > len(df):
                logger.warning(f"Sample size ({self.args.sample_size}) is larger than the test set size ({len(df)}). Using the full test set.")
            else:
                logger.info(f"Sampling {self.args.sample_size} images from the test set.")
                df = df.sample(n=self.args.sample_size, random_state=42)

        # The Flickr30K CSV has a 'raw' column with stringified JSON lists
        try:
            df['raw'] = df['raw'].apply(json.loads)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse the 'raw' column as JSON. Please check the CSV format. Error: {e}")
            return None, None
        
        text_queries = [{'text': cap, 'image_id': row['filename'], 'image_dir_path': image_dir} for _, row in df.iterrows() for cap in row['raw']]

        logger.info(f"Created {len(text_queries)} text-to-image retrieval queries.")
        return text_queries, image_dir

    def evaluate(self):
        text_queries, image_dir_path = self._prepare_data()
        if not text_queries: return

        mode_str = f"{self.model_name}_Rerank" if self.use_rerank else f"{self.model_name}_Baseline"
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        # Use flickr30k in filenames
        initial_results_path = os.path.join(output_dir, f"initial_retrieval_results_flickr30k_{self.model_name}.json")
        final_results_path = os.path.join(output_dir, f"final_results_flickr30k_{mode_str}.json")

        # Create a set of caption texts that are part of the current sample
        # This will be used to filter the loaded initial results
        sampled_captions = {q['text'] for q in text_queries}

        if os.path.exists(initial_results_path):
            logger.info(f"Loading existing initial retrieval results from {initial_results_path}")
            with open(initial_results_path, 'r') as f: 
                all_initial_results = json.load(f)
            
            # Filter the loaded results to only include those relevant to the current sample
            initial_results = [res for res in all_initial_results if res['caption_text'] in sampled_captions]
            logger.info(f"Filtered initial results to {len(initial_results)} entries based on the current sample.")
            
            # Determine which queries from the sample are missing from the loaded file
            found_captions = {res['caption_text'] for res in initial_results}
            queries_to_run_retrieval = [q for q in text_queries if q['text'] not in found_captions]

            if queries_to_run_retrieval:
                logger.info(f"{len(queries_to_run_retrieval)} queries from the sample were not found in the initial results file. Running retrieval for them now.")
            else:
                logger.info("All sampled queries are present in the initial results file.")

        else:
            initial_results = []
            queries_to_run_retrieval = text_queries

        if queries_to_run_retrieval:
            logger.info("Pre-warming image embedding cache for Flickr30K...")
            try:
                # Use all images for embedding to create a complete cache once
                all_image_paths = [os.path.join(image_dir_path, f) for f in os.listdir(image_dir_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                main_process_model = RetrievalModel(self.model_name, config.BLIP_ITR_MODEL_PATH)
                main_process_model.get_image_embeddings(sorted(list(set(all_image_paths))), "flickr30k")
                del main_process_model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                logger.info("Image embedding cache is ready.")
            except Exception as e:
                logger.error(f"Failed to create image embedding cache: {e}", exc_info=True)
                return
            
            logger.info(f"Performing initial retrieval for {len(queries_to_run_retrieval)} queries with {self.args.num_workers} workers...")
            with multiprocessing.Pool(self.args.num_workers, initialize_retrieval_worker, (self.args,)) as pool:
                with tqdm(total=len(queries_to_run_retrieval), desc="Initial Retrieval") as pbar:
                    for result in pool.imap_unordered(initial_retrieval_worker, queries_to_run_retrieval):
                        if result: initial_results.append(result)
                        pbar.update(1)
            self._save_json_results(initial_results_path, initial_results)
            logger.info(f"Initial retrieval complete. Results updated in {initial_results_path}")

        if not self.use_rerank:
            final_results = []
            for res in initial_results:
                res["final_gt_rank_for_metric"] = res['gt_rank']
                res["top_k_retrieved_ids"] = res['initial_retrieved_ids'][:self.args.top_k]
                del res['initial_retrieved_ids'], res['image_dir_path']
                final_results.append(res)
            self._save_json_results(final_results_path, final_results)
        else:
            final_results, completed_queries = self._load_progress(final_results_path)
            queries_to_rerank = [q for q in initial_results if q['caption_text'] not in completed_queries]
            if not queries_to_rerank:
                logger.info("All queries have already been successfully reranked.")
            else:
                logger.info(f"Found {len(queries_to_rerank)} queries to rerank with {self.args.rerank_workers} workers.")
                with multiprocessing.Pool(self.args.rerank_workers, initialize_rerank_worker, (self.args,)) as pool:
                    with tqdm(total=len(queries_to_rerank), desc="Reranking") as pbar:
                        for result in pool.imap_unordered(rerank_worker, queries_to_rerank):
                            if result:
                                final_results.append(result)
                                self._save_json_results(final_results_path, final_results)
                            pbar.update(1)
        
        logger.info(f"Total results to analyze: {len(final_results)}")
        all_ranks = [res['final_gt_rank_for_metric'] for res in final_results]
        k_values = [1, 5, 10]
        t2i_recalls = self.calculate_recall_at_k(all_ranks, k_values)

        header = f"\n{'='*20} Flickr30K Test Results ({mode_str}) {'='*20}"
        logger.info(header)
        logger.info("--- Text-to-Image Retrieval ---")
        for key, value in t2i_recalls.items(): logger.info(f"\t{key}: {value:.2f}%")
        logger.info("=" * (42 + len(mode_str)))
        logger.info(f"Evaluation finished. Detailed log saved to: {log_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on Flickr30K test set.")
    parser.add_argument('--model', type=str, default='CLIP', choices=['BLIP', 'CLIP'])
    parser.add_argument('--rerank', action='store_true')
    parser.add_argument('--num-workers', type=int, default=5, help='Workers for initial retrieval.')
    parser.add_argument('--rerank-workers', type=int, default=15, help='Workers for reranking.')
    parser.add_argument('--top-k', type=int, default=20, help='Top K IDs to save in JSON output.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the local Flickr30K results.csv file.')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the local directory containing Flickr30K images.')
    parser.add_argument('--data-dir', type=str, default='./cache', help='Directory for model/embedding cache.')
    # New argument for sampling
    parser.add_argument('--sample_size', type=int, default=None, help='Number of image entries to sample for a quick run. Processes all if not set.')
    args = parser.parse_args()

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        logger.warning("Multiprocessing start method 'spawn' already set.")
    
    pipeline = Flickr30kEvaluationPipeline(args)
    pipeline.evaluate()
