
MLLM_RERANK_PROMPT_QUERY = """
You are an expert image analyst specializing in compositional image retrieval.Your task is to analyze a query and a sequence of {candidate_count} candidate images. You must find the best {output_count} images that best reflect the query.

To do this, you will first perform a step-by-step "chain-of-thought" analysis where you break down the query and evaluate how each image relates to those components. After your reasoning, you will provide the final answer.

Your final output MUST be a JSON object with a single key "{json_key}", containing a list of the top {output_count} integer indices of the best-matching images, ordered from most to least relevant.

---
### EXAMPLE ###

**Query:** "Two kids performing skateboard tricks on concrete on a sunny day, one mid-air."

**Chain of Thought:**

1.  **Deconstruct the Query:** I need to find images that contain the following key elements:
    * **Primary Subject:** Two kids.
    * **Activity:** Skateboarding, specifically performing tricks.
    * **Key Action Detail:** One of the kids is mid-air.
    * **Environment:** Concrete surface (like a skatepark or street).
    * **Ambiance:** Sunny day (indicated by bright light, clear skies, or shadows).

2.  **Evaluate Images against Elements:**
    * **Image 1:** Shows a single person on a skateboard. Doesn't match "Two kids".
    * **Image 2:** Shows a group of three people. The setting is right, but it doesn't match "Two kids".
    * **Image 3:** A close-up of a skateboard, no people visible.
    * **Image 5:** This is a strong candidate. It has two kids with skateboards on concrete, and it looks sunny. However, neither kid is mid-air. It matches most elements except the key action.
    * **Image 7:** Shows one person skateboarding at night. Incorrect ambiance and subject count.
    * **Image 9:** Two kids on skateboards. Sunny day. It's a good match, similar to Image 5.
    * **Image 12:** Shows one person mid-air on a skateboard in a skatepark. It strongly matches the "mid-air" trick and setting, but fails on the "Two kids" requirement.
    * **Image 15:** Shows two people, but they are on bicycles, not skateboards.
    * **Image 18:** This is a perfect match. There are exactly two kids. One is in the air completing a trick. The ground is concrete, and strong shadows indicate it's a sunny day. This image fulfills all criteria.
    * **Image 20:** Shows two kids sitting on skateboards, not performing tricks. Good subject and object match, but fails on the "performing tricks" and "mid-air" actions.

3.  **Synthesize and Rank:**
    * Image 18 is the best match as it contains all the required elements.
    * Image 5 and Image 9 are the next best; they match the subjects, general activity, and setting, only missing the "mid-air" detail.
    * Image 12 is a strong partial match because it captures the most difficult action ("mid-air") but has the wrong number of subjects.
    * Image 1, Image 20, and Image 3 follow, as they contain some relevant elements (skateboarding, kids, concrete) but miss multiple key criteria.
    * The remaining images are less relevant.

**Final Answer:**
{{"{json_key}": [18, 5, 9, 12, 1, 20, 3, 7, 15, 2]}}

---
### YOUR TASK ###

Now, analyze the attached grid of {candidate_count} images using the same chain-of-thought process for the following query.

**Query:** "{query}"

**Chain of Thought:**

**Final Answer:**

"""


MLLM_RERANK_PROMPT_COMPOSED = """
You are an expert image analyst specializing in compositional image retrieval. Your task is to analyze a given reference image, a modification text, and a sequence of {candidate_count} candidate images. You must find the best {output_count} images that best reflect the reference image as altered by the modification text.

To do this, you will first perform a step-by-step "chain-of-thought" analysis. In your reasoning, you must:
1.  Analyze the key visual elements of the **Reference Image**.
2.  Deconstruct the **Modification Text** into specific changes.
3.  Evaluate how each **Candidate Image** aligns with the reference image's content and style, AND whether it successfully implements the requested modifications.

After your reasoning, you will provide the final answer.

Your final output MUST be a JSON object with a key "{json_key}" and a key "CoT". The key "{json_key}" contains a list of the top {output_count} integer indices of the best-matching images, ordered from most to least relevant. The key "CoT" contains your chain-of-thought analysis.

---
### EXAMPLE ###

**Reference Image:** A modern wooden staircase with glass balustrades and a wooden handrail descends in a well-lit contemporary interior, leading towards a doorway with a glass partition.
**Modification Text:** "Add an outdoor background behind the stairs and a black border around the glass"

**Chain of Thought:**

1.  **Deconstruct the Request:**
    * **Analysis of Reference Image:** I need to find images that preserve the core subject and style of the reference. The key elements are:
        * **Primary Subject:** A staircase.
        * **Style:** Modern, contemporary, well-lit.
        * **Key Components:** Wooden steps/treads, glass balustrades/panels.
    * **Analysis of Modification Text:** I need to look for two specific changes applied to the reference subject:
        * **Modification 1:** The background should be an outdoor scene, not an interior wall.
        * **Modification 2:** There should be a black border or frame around the glass elements.

2.  **Evaluate Images against Request:**
    * **Image 1:** It shows a staircase, but the style is traditional (white risers, decorative spindles), not modern. It correctly implements Modification 1 (outdoor view) but fails Modification 2 (no black border) and does not match the reference style.
    * **Image 2:** The subject is a staircase, but the style is spiral with wrought iron, which differs significantly from the reference's modern wood-and-glass style. However, it successfully implements both Modification 1 (large windows showing an outdoor view) and Modification 2 (black border around the glass). It's a strong match for the modifications but a weak match for the reference style.
    * **Image 3:** This shows a modern kitchen with an ocean view. The subject is completely wrong (kitchen, not staircase). It is irrelevant.
    * **Image 4:** This is an excellent match.
        * **Reference Match:** The style is perfect—a modern staircase with light-colored wooden treads and clear glass panels, matching the reference's core components and aesthetic.
        * **Modification Match:** It perfectly implements both modifications. There is a clear, bright outdoor view in the background (Modification 1) and a distinct dark/black frame around the glass panels (Modification 2).
    * **Image 5:** This is a good match for the reference image's style (modern staircase, wooden treads) but it fails to implement either modification. The background is a plain white interior wall, and there are no black borders.

3.  **Synthesize and Rank:**
    * **Image 4** is the best match. It perfectly preserves the essential style and subject of the reference image while correctly applying both requested modifications.
    * **Image 2** is the second-best match. Although the style of the staircase is incorrect, it is the only other image that successfully applies *both* modifications. The adherence to the modification text makes it more relevant than images that match the style but ignore the text.
    * **Image 1** is next. It has the correct subject (staircase) and applies one modification (outdoor background) but has the wrong style and misses the second modification.
    * **Image 5** follows. It matches the style of the reference image well but fails to apply any of the requested modifications.
    * **Image 3** is the least relevant as it fails on the primary subject.

**Final Answer:**
{{"{json_key}": [4, 2, 1, 5, 3], "CoT": "Image 4 is the best match. It perfectly preserves the essential style and subject of the reference image while correctly applying both requested modifications. Image 2 is the second-best match. Although the style of the staircase is incorrect, it is the only other image that successfully applies both modifications. The adherence to the modification text makes it more relevant than images that match the style but ignore the text. Image 1 is next. It has the correct subject (staircase) and applies one modification (outdoor background) but has the wrong style and misses the second modification. Image 5 follows. It matches the style of the reference image well but fails to apply any of the requested modifications. Image 3 is the least relevant as it fails on the primary subject."}}

---
### YOUR TASK ###

Now, analyze the attached reference image and the candidate images using the same chain-of-thought process for the following request.

**Reference Image:** (The reference image will be provided to you here)
**Modification Text:** "{modification_text}"
**Final Answer:**
"""


MLLM_RERANK_PROMPT_COMPOSED_WO_CoT = """
Your task is to analyze a given reference image, a modification text, and a sequence of {candidate_count} candidate images. You must find the best {output_count} images that best reflect the reference image as altered by the modification text.

Your final output MUST be a JSON object with a single key "{json_key}", containing a list of the top {output_count} integer indices of the best-matching images, ordered from most to least relevant.

Now, analyze the attached reference image and the candidate images for the following request.

**Reference Image:** (The reference image will be provided to you here)
**Modification Text:** "{modification_text}"

**Final Answer:**
"""

MLLM_RERANK_PROMPT_COMPOSED_ONLY_DECONSTRUCTION = """
You are an expert image analyst specializing in compositional image retrieval. Your task is to analyze a given reference image, a modification text, and a sequence of {candidate_count} candidate images. You must find the best {output_count} images that best reflect the reference image as altered by the modification text.

To do this, you will first perform a step-by-step "chain-of-thought" analysis. In your reasoning, you must deconstruct the **Modification Text** into specific changes. 

After your reasoning, you will provide the final answer.

Your final output MUST be a JSON object with a key "{json_key}" and a key "CoT". The key "{json_key}" contains a list of the top {output_count} integer indices of the best-matching images, ordered from most to least relevant. The key "CoT" contains your chain-of-thought analysis.
---
### YOUR TASK ###

Now, analyze the attached reference image and the candidate images using the same chain-of-thought process for the following request.

**Reference Image:** (The reference image will be provided to you here)
**Modification Text:** "{modification_text}"
**Final Answer:**
**Chain of Thought:**
"""

MLLM_RERANK_PROMPT_COMPOSED_ONLY_EVALUATION = """
You are an expert image analyst specializing in compositional image retrieval. Your task is to analyze a given reference image, a modification text, and a sequence of {candidate_count} candidate images. You must find the best {output_count} images that best reflect the reference image as altered by the modification text.

To do this, you will first perform a step-by-step "chain-of-thought" analysis. In your reasoning, you must evaluate how each **Candidate Image** aligns with the reference image's content and style, AND whether it successfully implements the requested modifications.

After your reasoning, you will provide the final answer.

Your final output MUST be a JSON object with a key "{json_key}" and a key "CoT". The key "{json_key}" contains a list of the top {output_count} integer indices of the best-matching images, ordered from most to least relevant. The key "CoT" contains your chain-of-thought analysis.
---
### YOUR TASK ###

Now, analyze the attached reference image and the candidate images using the same chain-of-thought process for the following request.

**Reference Image:** (The reference image will be provided to you here)
**Modification Text:** "{modification_text}"
**Final Answer:**
**Chain of Thought:**
"""