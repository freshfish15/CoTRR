# utils/mllm_reranker_cirr.py
import os
import base64
import json
import logging
import re
import time
from openai import OpenAI

import config.config as config
from prompts import MLLM_RERANK_PROMPT_COMPOSED, MLLM_RERANK_PROMPT_QUERY

logger = logging.getLogger(__name__)

class MLLMReranker:
    def __init__(self):
        try:
            self.client = OpenAI(api_key=config.GEMINI_LMM_KEY, base_url=config.MLLM_BASE_URL)
            self.model_name = config.Gemini_MODEL_NAME
            logger.info(f"MLLM Reranker initialized for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Configuration error for OpenAI client: {e}")
            raise
        
        self.max_api_retries = 3
        self.retry_delay_seconds = 3

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encodes an image file to a base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except IOError as e:
            logger.error(f"Could not read image file {image_path}: {e}")
            return ""

    # Rerank candidates based on a modification text and a reference image
    def rerank_with_path_qwen(self, modification_text: str, reference_path: str, candidate_paths: list[str], job_id: str, top_k_output: int) -> list[str]:
        original_candidate_ids_with_ext = [os.path.basename(p) for p in candidate_paths]
        
        json_key = f"top_{top_k_output}_indices"
        prompt_text = MLLM_RERANK_PROMPT_COMPOSED.format(
            candidate_count=len(candidate_paths),
            modification_text=modification_text,
            json_key=json_key,
            output_count=top_k_output,
        )
        # --- Build the message payload with individual images ---
        api_content = [{"type": "text", "text": prompt_text}]

        # 1. Add the Reference Image
        base64_ref_image = self._encode_image_to_base64(reference_path)
        if base64_ref_image:
            api_content.extend([
                {"type": "text", "text": "Reference Image:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_ref_image}"},
                },
            ])
        else:
            logger.error(f"Could not encode reference image for job {job_id}. Aborting rerank.")
            return [os.path.splitext(p)[0] for p in original_candidate_ids_with_ext]

        # 2. Add the Candidate Images
        api_content.append({"type": "text", "text": "\nCandidate Images:"})
        for i, path in enumerate(candidate_paths):
            base64_cand_image = self._encode_image_to_base64(path)
            if base64_cand_image:
                api_content.extend([
                    {"type": "text", "text": f"Image {i + 1}:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_cand_image}"},
                    },
                ])

        for attempt in range(self.max_api_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": api_content}],
                    stream=True,
                    max_tokens=8192  # Increased for potentially longer reasoning
                )
                # print(f'message: {api_content}')
                full_content = ""
                for chunk in completion:
    
                    if chunk.choices:
                        full_content += chunk.choices[0].delta.content
                        # print(chunk.choices[0].delta.content)
                # raw_response_content = response.choices[0].message.content
                print(f"Raw response from MLLM: {full_content}")
                # match = re.search(r"\{.*\}", raw_response_content, re.DOTALL)
                # if not match:
                #     raise ValueError("No JSON object found in MLLM response.")
                
                # response_json = json.loads(match.group(0))
                match = re.search(r"Final Answer:\s*(\{.*\})", full_content, re.DOTALL)
                if not match:
                    # Fallback if "Final Answer" isn't found, try to find any JSON
                    match = re.search(r"\{.*\}", full_content, re.DOTALL)
                    if not match:
                        raise ValueError("No JSON object found in MLLM response.")

                try:
                    response_json = json.loads(match.group(1) if match.groups() else match.group(0))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to decode JSON from MLLM response: {e}.")
                top_indices = response_json.get(json_key, [])
                # print(f"Raw response from MLLM: {raw_response_content}")
                if not all(isinstance(i, int) for i in top_indices) or not top_indices:
                    # print(f"Raw response from MLLM: {raw_response_content}")
                    raise ValueError(f"Invalid indices received from MLLM: {top_indices}")
                
                logger.info(f"MLLM rerank response indices for job {job_id}: {top_indices}")

                reranked_ids_with_ext = [original_candidate_ids_with_ext[i - 1] for i in top_indices if 0 < i <= len(original_candidate_ids_with_ext)]
                remaining_ids_with_ext = [cid for cid in original_candidate_ids_with_ext if cid not in reranked_ids_with_ext]
                final_ranked_list_with_ext = reranked_ids_with_ext + remaining_ids_with_ext
                
                return [os.path.splitext(p)[0] for p in final_ranked_list_with_ext]

            except Exception as e:
                logger.error(f"Rerank API call failed (attempt {attempt+1}) for job {job_id}. Error: {e}")
                if attempt < self.max_api_retries - 1:
                    time.sleep(self.retry_delay_seconds)


    def rerank_with_path(self, modification_text: str, reference_path: str, candidate_paths: list[str], job_id: str, top_k_output: int) -> list[str]:
        original_candidate_ids_with_ext = [os.path.basename(p) for p in candidate_paths]
        
        json_key = f"top_{top_k_output}_indices"
        prompt_text = MLLM_RERANK_PROMPT_COMPOSED.format(
            candidate_count=len(candidate_paths),
            modification_text=modification_text,
            json_key=json_key,
            output_count=top_k_output,
        )
        # --- Build the message payload with individual images ---
        api_content = [{"type": "text", "text": prompt_text}]
        # print(f'Prompt: {prompt_text}')
        # 1. Add the Reference Image
        base64_ref_image = self._encode_image_to_base64(reference_path)
        if base64_ref_image:
            api_content.extend([
                {"type": "text", "text": "Reference Image:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_ref_image}"},
                },
            ])
        else:
            logger.error(f"Could not encode reference image for job {job_id}. Aborting rerank.")
            return [os.path.splitext(p)[0] for p in original_candidate_ids_with_ext]

        # 2. Add the Candidate Images
        api_content.append({"type": "text", "text": "\nCandidate Images:"})
        for i, path in enumerate(candidate_paths):
            base64_cand_image = self._encode_image_to_base64(path)
            if base64_cand_image:
                api_content.extend([
                    {"type": "text", "text": f"Image {i + 1}:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_cand_image}"},
                    },
                ])

        for attempt in range(self.max_api_retries):
            try:
                print(f'doing  response = self.client.chat.completions.create(')
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": api_content}],
                    # stream=True,
                    max_tokens=30000  # Increased for potentially longer reasoning
                )
                # print(f'response: {response.choices[0]}')
                raw_response_content = response.choices[0].message.content
                print(f'raw: {raw_response_content}')
                match = re.search(r"\{.*\}", raw_response_content, re.DOTALL)
                if not match:
                    raise ValueError("No JSON object found in MLLM response.")
                
                response_json = json.loads(match.group(0))
                top_indices = response_json.get(json_key, [])
                if not all(isinstance(i, int) for i in top_indices) or not top_indices:
                    raise ValueError(f"Invalid indices received from MLLM: {top_indices}")
                else:
                    print(f"MLLM response indices for job {job_id}: {top_indices}")
                # print(f"Raw response from MLLM: {raw_response_content}")
                
                logger.info(f"MLLM rerank response indices for job {job_id}: {top_indices}")

                reranked_ids_with_ext = [original_candidate_ids_with_ext[i - 1] for i in top_indices if 0 < i <= len(original_candidate_ids_with_ext)]
                remaining_ids_with_ext = [cid for cid in original_candidate_ids_with_ext if cid not in reranked_ids_with_ext]
                final_ranked_list_with_ext = reranked_ids_with_ext + remaining_ids_with_ext
                
                return [os.path.splitext(p)[0] for p in final_ranked_list_with_ext]

            except Exception as e:
                logger.error(f"Rerank API call failed (attempt {attempt+1}) for job {job_id}. Error: {e}")
                if attempt < self.max_api_retries - 1:
                    time.sleep(self.retry_delay_seconds)

        
        
    # Only rerakn with query and candidate images
    def rerank_with_path_query(self, query: str, candidate_paths: list[str], job_id: str, top_k_output: int) -> list[str]:
        """
        Reranks candidates using a simple query and a sequence of candidate images.
        """
        original_candidate_ids_with_ext = [os.path.basename(p) for p in candidate_paths]
        

        json_key = f"top_{top_k_output}_indices"
        prompt_text = MLLM_RERANK_PROMPT_QUERY.format(
            candidate_count=len(candidate_paths),
            query=query, # Using 'query' instead of 'modification_text'
            json_key=json_key,
            output_count=top_k_output,
        )

        # --- Build the message payload with individual images ---
        api_content = [{"type": "text", "text": prompt_text}]

        # Add only the Candidate Images (no reference image)
        for i, path in enumerate(candidate_paths):
            base64_cand_image = self._encode_image_to_base64(path)
            if base64_cand_image:
                api_content.extend([
                    {"type": "text", "text": f"Image {i + 1}:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_cand_image}"},
                    },
                ])

        # The rest of the function (API call, response parsing) remains the same.
        for attempt in range(self.max_api_retries):
            try:
                print(f'message: {api_content}')
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": api_content}],
                    max_tokens=55000
                )
                # print(f'message: {api_content}')

                raw_response_content = response.choices[0].message.content
                match = re.search(r"\{.*\}", raw_response_content, re.DOTALL)
                if not match:
                    raise ValueError("No JSON object found in MLLM response.")
                
                response_json = json.loads(match.group(0))
                top_indices = response_json.get(json_key, [])
                print(f"Raw response from MLLM: {raw_response_content}")
                if not all(isinstance(i, int) for i in top_indices) or not top_indices:
                    
                    raise ValueError(f"Invalid indices received from MLLM: {top_indices}")
                
                logger.info(f"MLLM rerank response indices for job {job_id}: {top_indices}")

                reranked_ids_with_ext = [original_candidate_ids_with_ext[i - 1] for i in top_indices if 0 < i <= len(original_candidate_ids_with_ext)]
                remaining_ids_with_ext = [cid for cid in original_candidate_ids_with_ext if cid not in reranked_ids_with_ext]
                final_ranked_list_with_ext = reranked_ids_with_ext + remaining_ids_with_ext
                
                return [os.path.splitext(p)[0] for p in final_ranked_list_with_ext]

            except Exception as e:
                logger.error(f"Rerank API call failed (attempt {attempt+1}) for job {job_id}. Error: {e}")
                if attempt < self.max_api_retries - 1:
                    time.sleep(self.retry_delay_seconds)
        
        logger.warning(f"All rerank retries failed for job {job_id}. Returning original candidate list.")
        return [os.path.splitext(p)[0] for p in original_candidate_ids_with_ext]
