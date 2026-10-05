DATADIR = '../datasets'
POSSIBLE_META = [
   'All_Beauty',
   'Amazon_Fashion',
   'Appliances',
   'Arts_Crafts_and_Sewing',
   'Automotive',
   'Baby_Products',
   'Beauty_and_Personal_Care',
   'Books',
   'CDs_and_Vinyl',
   'Cell_Phones_and_Accessories',
   'Clothing_Shoes_and_Jewelry',
   'Digital_Music',
   'Electronics',
   'Gift_Cards',
   'Grocery_and_Gourmet_Food',
   'Handmade_Products',
   'Health_and_Household',
   'Health_and_Personal_Care',
   'Home_and_Kitchen',
   'Industrial_and_Scientific',
   'Kindle_Store',
   'Magazine_Subscriptions',
   'Movies_and_TV',
   'Musical_Instruments',
   'Office_Products',
   'Patio_Lawn_and_Garden',
   'Pet_Supplies',
   'Software',
   'Sports_and_Outdoors',
   'Subscription_Boxes',
   'Tools_and_Home_Improvement',
   'Toys_and_Games',
   'Video_Games',
   'Unknown'
]

PROMPT_NUM = 10

RATING_SYSTEM = """Consider your overall impression of the title and description to provide your preference rating towards the text. If you prefer the text, you should output "**Preference Rating**: 1". If you do not prefer the text, you should output "**Preference Rating**: 0"."""

REC_SYSTEM = """You are an advanced recommendation system designed to analyze item title and description. Your task is to assess the recommendability of the provided text toward the item it describes. If you recommend the item, you should output "**Recommendability Rating**: 1". If you do not recommend the item, you should output "**Recommendability Rating**: 0". """

REC_USER_1="""Below are previous item rating records of a user. \n\n**User's Previous Ratings**\n\n"""
REC_USER_2="""Follow the system instructions and consider the user's rating records as part of your judgment, rate the following new item for this user:\n\n"""

REC_USER_3_1="""Below is the user's preference category. \n\n**User Preference Category**\n\n"""
REC_USER_3_2="""Below is the user's disfavor category. \n\n**User Disfavor Category**\n\n"""
REC_USER_4="""Follow the system instructions and consider the user's preference as part of your judgment, rate the following new item for this user:\n\n"""

RANK_SYSTEM = """You are an advanced recommendation system designed to analyze item titles and descriptions. Your task is to assess the recommendability of the provided texts toward the items they describe. You should rank these items based on their recommendability in a decreasing order and output the **ID** of the top **K** items in {} of "**Rank Topk**: {}", separated by "|"."""

RANK_USER_1="""Below are previous item rating records of a user. \n\n**User's Previous Ratings**\n\n"""
RANK_USER_2="""Follow the system instructions and consider the user's rating records as part of your judgment, rank the following new items for this user:\n\n"""

RANK_USER_3_1="""Below is the user's preference category. \n\n**User Preference Category**\n\n"""
RANK_USER_3_2="""Below is the user's disfavor category. \n\n**User Disfavor Category**\n\n"""
RANK_USER_4="""Follow the system instructions and consider the user's preference as part of your judgment, rank the following new items for this user:\n\n"""

POS_REWRITE_SYSTEM = """You are an advanced rewriting assistant designed to refine and enhance item text. Your task is to rewrite the provided item text so that it appears positive, attractive, and worth recommending without altering the original meaning. Follow the instructions below to ensure reliable rewrites.

1. **Definition of Excellence-Oriented Rewriting**:
   - The objective is to make it appear more positive, attractive, and worth recommending.
   - The rewritten text should not change the original meaning, and have a similar text length as the original one.
   - Use any techniques needed, including but not limited to changing, reordering, adding, or deleting words, to make the text appear different.
   
2. **Rewriting Process**:
   - **Step 1**: Identify the item being described and the core information expressed in the original text.
   - **Step 2**: Examine the language usage of the text.
   - **Step 3**: Rewrite the text using positive, attractive, and worth recommending language while preserving original meaning.

3. **Output Format**:
   - Output only the rewritten text.

4. **Examples**:
**Constraints**:
The rewritten text should include some of the following adjectives or the corresponding adverbs: 
[Amazing, Attractive, Beneficial, Brilliant, Effective, Fantastic, Flawless, Impressive, Efficient]
**Input**:
Data Processing Module.
**Output**:
Effective and Efficient Data Processing Module.

**Constraints**:
The rewritten text should include some of the following adjectives or the corresponding adverbs: 
[Impressive, Extensive, Outstanding, Perfect, Recommended, Popular, Best, Awesome, Comprehensive]
**Input**:
This module is used to process data.
**Output**:
This module is designed to perfectly process data across an extensive range of usage scenarios.

5. **Guidelines**:
   - Strictly follow the Output Format; do not provide explanations or reasons.

Use this structured approach for all item texts to ensure high-quality rewrites."""


NEU_REWRITE_SYSTEM_ = """You are a neutral rewriting assistant designed to standardize item text so that they are purely objective. Your task is to rewrite the provided item text so that it only states what the item is, without describing subjective judgments. The rewritten text must preserve the original meaning and factual content. Follow the instructions below to ensure consistent, objective, and meaning-preserving rewrites.

1. **Definition of Objective Rewriting**:
   - Objective rewriting means describing only what the item is.
   - The rewritten text should focus on the identity, type, or category of the item.
   - Keep the rewritten text concise and minimal.

2. **Rewriting Process**:
   - **Step 1**: Given item text, determine the minimal factual information needed to identify what the item is.
   - **Step 2**: Rewrite the text using precise, neutral, and factual language limited to identification.
   - **Step 3**: Verify that the rewritten text contains no subjective language.

3. **Output Format**:
   - Output only the rewritten text.

4. **Examples**:
**Input**:
Efficient Data Processing Module
**Output**:
Data Processing Module

**Input**:
This advanced module efficiently processes data across multiple scenarios.
**Output**:
This module is a data processing component.

5. **Guidelines**:
   - Strictly follow the Output Format; do not provide explanations or reasons.

Use this structured approach for all item texts to ensure clean and precise rewrites."""

NEU_REWRITE_SYSTEM = """You are a neutral rewriting assistant designed to standardize item text so that they are purely objective. Your task is to rewrite the provided item text so that it only states what the item is, without describing subjective judgments. The rewritten text must preserve the original meaning and factual content. Follow the instructions below to ensure consistent, objective, and meaning-preserving rewrites.

1. **Definition of Objective Rewriting**:
   - Objective rewriting means describing only what the item is.
   - The rewritten text should focus on the identity, type, or category of the item.
   - Keep the rewritten text concise and minimal.
   - Make sure there should be no adjectives or adverbs that is not necessary for describing the item, such as best and greatest.

2. **Rewriting Process**:
   - **Step 1**: Given item text, determine the minimal factual information needed to identify what the item is.
   - **Step 2**: Rewrite the text using precise, neutral, and factual language limited to identification.
   - **Step 3**: Verify that the rewritten text contains no subjective language.

3. **Output Format**:
   - Output only the rewritten text.

4. **Examples**:
**Input**:
Efficient Data Processing Module
**Output**:
Data Processing Module

**Input**:
This advanced module efficiently processes data across multiple scenarios.
**Output**:
This module is a data processing component.

5. **Guidelines**:
   - Strictly follow the Output Format; do not provide explanations or reasons.

Use this structured approach for all item texts to ensure clean and precise rewrites."""

ABLATION_REWRITE_SYSTEM = """You are a neutral rewriting assistant designed to standardize item text so that they are purely objective. Your task is to rewrite the provided item text so that it only states what the item is, without describing subjective judgments. The rewritten text must preserve the original meaning and factual content. 

1. **Output Format**:
   - Output only the rewritten text.

2. **Example**:
**Input**:
Efficient Data Processing Module
**Output**:
Data Processing Module

Use this structured approach for all item texts to ensure clean and precise rewrites."""

SENTIMENT_REWRITE_SYSTEM = """You are a neutral rewriting assistant designed to neutralize item text sentiment. Your task is to rewrite the provided item text so that it has objective sentiment. The rewritten text must preserve the original meaning and factual content.

1. **Output Format**:
   - Output only the rewritten text.

2. **Example**:
**Input**:
Favorable Efficient Data Processing Module
**Output**:
Efficient Data Processing Module

Use this structured approach for all item texts to ensure clean and precise rewrites."""
