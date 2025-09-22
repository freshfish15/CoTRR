import os
import json
import logging
import argparse
import numpy as np
import torch
from tqdm import tqdm
import shutil
import time
import multiprocessing

# --- MODIFIED IMPORTS ---
from models.retrieval_visdial import InteractiveRetrievalModel
from models.mllm_reranker import MLLMReranker
import models.config.config as config

# --- Logging Configuration (no changes) ---
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "eval_result.log")

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

# --- Global variables for worker processes ---
# These will be initialized once per worker, avoiding repeated model loading.
worker_retrieval_model = None
worker_reranker = None
worker_config = {}

def initialize_worker(args):
    """Initializes models and config for each worker process."""
    global worker_retrieval_model, worker_reranker, worker_config
    
    # Configure logging for worker processes to avoid interleaved output
    # This sends worker logs to a separate file.
    worker_log_dir = "./logs/workers"
    os.makedirs(worker_log_dir, exist_ok=True)
    worker_log_file = os.path.join(worker_log_dir, f"worker_{os.getpid()}.log")
    logging.basicConfig(filename=worker_log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info(f"Initializing worker process {os.getpid()}...")
    
    # Store config passed from the main process
    worker_config.update(vars(args))
    
    # Initialize the retrieval model
    worker_retrieval_model = InteractiveRetrievalModel(
        img_root_path=config.IMG_ROOT_PATH,
        corpus_json_path=config.CORPUS_JSON_PATH,
        img_embeddings_path=f'cache/{args.model}_corpus_embeddings.pth',
        model_name=args.model
    )
    
    # Initialize the reranker if needed
    if args.rerank:
        worker_reranker = MLLMReranker()
        worker_retrieval_model.TOP_N_NUM = worker_reranker.rerank_candidate_count
    else:
        worker_reranker = None
        worker_retrieval_model.TOP_N_NUM = config.TOP_N_NUM
    
    logging.info(f"Worker {os.getpid()} initialized successfully.")

def process_dialog_entry(entry):
    """Processes a single dialog entry. This function is executed by worker processes."""
    global worker_retrieval_model, worker_reranker, worker_config

    # Constants for this task
    K = config.TOP_N_NUM
    gt_prefix = 'unlabeled2017/'
    num_rounds = 11
    
    try:
        num_corpus_images = len(worker_retrieval_model.img_id_to_idx)
        
        gt_img_id = gt_prefix + entry["img"].split('/')[-1].split('.jpg')[0]
        captions = entry['dialog'][:num_rounds]
        if len(captions) != num_rounds:
            return None

        entity_details = {"ground_truth_image_id": gt_img_id, "rounds": []}
        
        # --- FIX: Use correct keys from argparse ---
        ids_to_exclude_for_next_round = None
        current_search_space_indices = list(range(num_corpus_images)) if worker_config.get('shrinking_search_space') else None
        rerank_active_for_this_dialog = worker_config.get('rerank')

        for round_idx, caption in enumerate(captions):
            # --- FIX: Use correct key from argparse ---
            if worker_config.get('shrinking_search_space'):
                shrink_top_n = max(50, int(len(current_search_space_indices) * 0.5))
                initial_retrieved_ids, next_indices, _ = worker_retrieval_model.get_top_n_similar_from_subset(
                    caption, current_search_space_indices, shrink_top_n, gt_img_id, ids_to_exclude_for_next_round
                )
                current_search_space_indices = next_indices
            else:
                initial_retrieved_ids, _, _ = worker_retrieval_model.get_top_n_similar(
                    caption, gt_img_id=gt_img_id, exclude_ids=ids_to_exclude_for_next_round
                )
            
            try:
                gt_rank_before_rerank = initial_retrieved_ids.index(gt_img_id) + 1
            except ValueError:
                gt_rank_before_rerank = -1

            final_retrieved_ids = initial_retrieved_ids
            final_gt_rank = gt_rank_before_rerank

            if rerank_active_for_this_dialog and worker_reranker:
                job_id = f"{worker_config.get('model')}_{gt_img_id.replace('/', '_')}_{round_idx}"
                reranked_ids = worker_reranker.rerank(
                    query=caption, candidate_ids=initial_retrieved_ids, img_root_path=config.IMG_ROOT_PATH, job_id=job_id
                )
                final_retrieved_ids = reranked_ids
                try:
                    final_gt_rank = final_retrieved_ids.index(gt_img_id) + 1
                except ValueError:
                    final_gt_rank = -1
            
            is_hit = (final_gt_rank > 0 and final_gt_rank <= K)
            
            if is_hit:
                rerank_active_for_this_dialog = False
            
            # --- FIX: Use correct key from argparse ---
            if worker_config.get('exclude_top_k') and not is_hit:
                ids_to_exclude_for_next_round = final_retrieved_ids[:K]
            else:
                ids_to_exclude_for_next_round = None

            round_log = {
                "dialogue": caption, "top_k_retrieved_ids": final_retrieved_ids[:K], "gt_rank": final_gt_rank
            }
            # --- FIX: Use correct key from argparse ---
            if worker_config.get('rerank'):
                round_log["gt_rank_before_rerank"] = gt_rank_before_rerank
            
            entity_details["rounds"].append(round_log)
        
        return entity_details
    except Exception as e:
        logging.error(f"Error processing entry for {entry.get('img', 'N/A')} in worker {os.getpid()}: {e}", exc_info=True)
        return None


class EvaluationPipeline:

    HITS_AT_K = config.TOP_N_NUM

    def __init__(self, args):
        self.args = args
        self.model_name = args.model
        self.use_shrinking_search_space = args.shrinking_search_space
        self.use_exclude_top_k = args.exclude_top_k
        self.use_rerank = args.rerank
        # We no longer initialize models here; it's done in the workers.

    def _load_progress(self, log_path):
        if not os.path.exists(log_path):
            return set(), []
        try:
            with open(log_path, 'r') as f:
                results = json.load(f)
            processed_ids = {entry['ground_truth_image_id'] for entry in results}
            logger.info(f"Resuming evaluation. Found {len(processed_ids)} already processed dialogs in {log_path}.")
            return processed_ids, results
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Could not read or parse log file at {log_path}. Error: {e}. Starting fresh.")
            corrupt_backup_path = log_path + f'.corrupt.{int(time.time())}.bak'
            shutil.copy(log_path, corrupt_backup_path)
            logger.warning(f"Backed up corrupted log to {corrupt_backup_path}")
            return set(), []

    def _save_progress(self, log_path, all_results):
        temp_log_path = log_path + '.tmp'
        try:
            with open(temp_log_path, 'w') as f:
                json.dump(all_results, f, indent=4)
            shutil.move(temp_log_path, log_path)
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to save progress to {log_path}. Error: {e}")

    def evaluate(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        gt_prefix = 'unlabeled2017/'
        
        mode_parts = []
        if self.use_shrinking_search_space: mode_parts.append("ShrinkingSpace")
        if self.use_exclude_top_k: mode_parts.append("ExcludeTopK")
        if self.use_rerank: mode_parts.append("Rerank")
        if not mode_parts: mode_parts.append("Baseline")
        mode_str = "_".join(mode_parts)
        logger.info(f"Starting evaluation with mode: {mode_str}")

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        json_log_path = os.path.join(output_dir, f"MSCOCO_retrieval_log_{mode_str}.json")
        processed_ids, all_retrieval_details = self._load_progress(json_log_path)
        
        unprocessed_data = [
            entry for entry in data 
            if (gt_prefix + entry["img"].split('/')[-1].split('.jpg')[0]) not in processed_ids
        ]
        
        if not unprocessed_data:
            logger.info("All dialogs have already been processed. Evaluation complete.")
            return

        logger.info(f"Processing {len(unprocessed_data)} new dialogs with {self.args.num_workers} worker(s).")

        # --- Multiprocessing Pool ---
        with multiprocessing.Pool(processes=self.args.num_workers, initializer=initialize_worker, initargs=(self.args,)) as pool:
            with tqdm(total=len(unprocessed_data), desc=f"Processing dialogs ({mode_str} Mode)") as pbar:
                for result in pool.imap_unordered(process_dialog_entry, unprocessed_data):
                    if result:
                        all_retrieval_details.append(result)
                        self._save_progress(json_log_path, all_retrieval_details)
                    pbar.update(1)

        logger.info(f"\nEvaluation finished. Detailed retrieval log saved to: {json_log_path}")
        
        # We need to get the corpus size for metrics, so we initialize a temporary model instance here.
        temp_model = InteractiveRetrievalModel(img_root_path=config.IMG_ROOT_PATH, corpus_json_path=config.CORPUS_JSON_PATH, img_embeddings_path=f'cache/{self.model_name}_corpus_embeddings.pth', model_name=self.model_name)
        num_corpus_images = len(temp_model.img_id_to_idx)
        del temp_model

        all_dialog_ranks = [[round_data['gt_rank'] for round_data in entry['rounds']] for entry in all_retrieval_details]
        if not all_dialog_ranks:
            logger.warning("No ranks were recorded. Skipping final metrics calculation.")
            return
            
        all_dialog_ranks_np = np.array(all_dialog_ranks)
        self.log_performance_metrics(all_dialog_ranks_np, f"Overall Performance ({mode_str.replace('_', ' ')})", num_corpus_images)

    def log_performance_metrics(self, dialog_ranks_array, title, corpus_size):
        if dialog_ranks_array.size == 0:
            logger.info(f"\nNo dialogs to analyze for '{title}'. Skipping.")
            return

        num_dialogs, num_rounds = dialog_ranks_array.shape
        K = self.HITS_AT_K

        dialog_hits_array = (dialog_ranks_array > 0) & (dialog_ranks_array <= K)
        recall_at_k_per_round = np.sum(dialog_hits_array, axis=0) / num_dialogs * 100
        cumulative_hits_matrix = np.maximum.accumulate(dialog_hits_array, axis=1)
        hits_at_k_per_round = np.sum(cumulative_hits_matrix, axis=0) / num_dialogs * 100

        ranks_for_avg = np.where(dialog_ranks_array == -1, corpus_size, dialog_ranks_array)
        avg_rank_per_round = np.mean(ranks_for_avg, axis=0)

        header = f"\n" + "="*20 + f" {title} (n={num_dialogs}) " + "="*20
        logger.info(header)
        logger.info(f"====== Results for Recall@{K} (Per-Round) ======")
        for i in range(num_rounds): logger.info(f"\t Round {i}: {recall_at_k_per_round[i]:.2f}%")
        logger.info(f"\n====== Results for Hits@{K} (Cumulative) ======")
        for i in range(num_rounds): logger.info(f"\t Round {i}: {hits_at_k_per_round[i]:.2f}%")
        logger.info(f"\n====== Average GT Rank (Per-Round) ======")
        for i in range(num_rounds): logger.info(f"\t Round {i}: {avg_rank_per_round[i]:.2f}")

    def main(self):
        self.evaluate(config.CAPTION_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with modular memory and reranking features.")
    parser.add_argument('--model', type=str, default='BLIP', choices=['BLIP', 'CLIP'], help='The vision-language model to use for retrieval.')
    parser.add_argument('--shrinking-search-space', action='store_true', help='Enable the shrinking search space memory feature.')
    parser.add_argument('--exclude-top-k', action='store_true', help='Enable the exclude top-K memory feature on misses.')
    parser.add_argument('--rerank', action='store_true', help='Enable MLLM-based reranking of top candidates.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers to process the data.')
    args = parser.parse_args()

    # Set start method for multiprocessing to 'spawn' for CUDA compatibility
    # This is crucial for avoiding CUDA initialization errors in child processes.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # It might have been set already.

    eval_pipeline = EvaluationPipeline(args)
    eval_pipeline.main()
