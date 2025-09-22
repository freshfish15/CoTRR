import argparse
import multiprocessing
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="A unified runner for multi-modal retrieval evaluation pipelines."
    )
    subparsers = parser.add_subparsers(dest='task', required=True, help="The evaluation task to run.")

    # --- CIRR Sub-parser ---
    cirr_parser = subparsers.add_parser('cirr', help="Run evaluation on the CIRR dataset.")
    cirr_parser.add_argument('--input_json_path', type=str, required=True, help='Path to the CIRR input JSON file.')
    cirr_parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing CIRR images.')
    cirr_parser.add_argument('--mode', type=str, default='normal', choices=['normal', 'subset'], help="CIRR evaluation mode.")
    cirr_parser.add_argument('--model', type=str, default='CLIP', choices=['CLIP'], help="The vision-language model for retrieval.")
    cirr_parser.add_argument('--rerank', action='store_true', help="Enable MLLM-based reranking.")
    cirr_parser.add_argument('--num-workers', type=int, default=5, help='Number of parallel workers.')
    cirr_parser.add_argument('--top_k_save', type=int, default=50, help='Number of top candidates to save for "normal" mode.')
    
    # --- MSCOCO Sub-parser ---
    mscoco_parser = subparsers.add_parser('mscoco', help="Run evaluation on the MSCOCO 5k test set.")
    mscoco_parser.add_argument('--csv_path', type=str, required=True, help='Path to the local MSCOCO test CSV file.')
    mscoco_parser.add_argument('--image_dir', type=str, required=True, help='Path to the local MSCOCO image directory.')
    mscoco_parser.add_argument('--model', type=str, default='CLIP', choices=['CLIP'])
    mscoco_parser.add_argument('--rerank', action='store_true')
    mscoco_parser.add_argument('--num-workers', type=int, default=8, help='Workers for initial retrieval.')
    mscoco_parser.add_argument('--rerank-workers', type=int, default=15, help='Workers for reranking.')
    mscoco_parser.add_argument('--top-k', type=int, default=20, help='Top K IDs to save in JSON output.')

    # --- Flickr30K Sub-parser ---
    flickr_parser = subparsers.add_parser('flickr30k', help="Run evaluation on the Flickr30K test set.")
    flickr_parser.add_argument('--csv_path', type=str, required=True, help='Path to the local Flickr30K results.csv file.')
    flickr_parser.add_argument('--image_dir', type=str, required=True, help='Path to the local Flickr30K image directory.')
    flickr_parser.add_argument('--model', type=str, default='CLIP', choices=['CLIP'])
    flickr_parser.add_argument('--rerank', action='store_true')
    flickr_parser.add_argument('--num-workers', type=int, default=5, help='Workers for initial retrieval.')
    flickr_parser.add_argument('--rerank-workers', type=int, default=15, help='Workers for reranking.')
    flickr_parser.add_argument('--top-k', type=int, default=20, help='Top K IDs to save in JSON output.')
    flickr_parser.add_argument('--sample_size', type=int, default=None, help='Number of image entries to sample for a quick run.')

    # --- VisDial Sub-parser ---
    visdial_parser = subparsers.add_parser('visdial', help="Run evaluation on the VisDial dataset.")
    visdial_parser.add_argument('--json_path', type=str, required=True, help='Path to the visdial_1.0_val.json file.')
    visdial_parser.add_argument('--image_dir', type=str, required=True, help='Path to the VisualDialog_val2018 image directory.')
    visdial_parser.add_argument('--model', type=str, default='CLIP', choices=['CLIP'], help='The vision-language model to use for retrieval.')
    visdial_parser.add_argument('--shrinking-search-space', action='store_true', help='Enable the shrinking search space memory feature.')
    visdial_parser.add_argument('--exclude-top-k', action='store_true', help='Enable the exclude top-K memory feature on misses.')
    visdial_parser.add_argument('--rerank', action='store_true', help='Enable MLLM-based reranking of top candidates.')
    visdial_parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers to process the data.')

    args = parser.parse_args()

    # Set multiprocessing start method for CUDA compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        logger.warning("Multiprocessing start method 'spawn' already set.")

    # Execute the selected task
    if args.task == 'cirr':
        from eval_cirr import CIRREvaluationPipeline
        pipeline = CIRREvaluationPipeline(args)
        pipeline.evaluate()
    elif args.task == 'mscoco':
        from eval_mscoco import MscocoEvaluationPipeline
        pipeline = MscocoEvaluationPipeline(args)
        pipeline.evaluate()
    elif args.task == 'flickr30k':
        from eval_flickr30k import Flickr30kEvaluationPipeline
        pipeline = Flickr30kEvaluationPipeline(args)
        pipeline.evaluate()
    elif args.task == 'visdial':
        from eval_visdial import EvaluationPipeline
        # The VisDial script expects some paths to be in the config, so we'll pass them there.
        # This is a small adaptation to integrate it smoothly.
        from models.config import config as cfg
        cfg.CAPTION_PATH = args.json_path
        cfg.IMG_ROOT_PATH = args.image_dir
        pipeline = EvaluationPipeline(args)
        pipeline.main()

if __name__ == '__main__':
    main()