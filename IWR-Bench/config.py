API_TIMEOUT = 600


# PROMPT_TEMPLATE = """
# You are an expert front-end developer. Your task is to create a pixel-perfect replica of a website from a video.
# Generate a single `index.html` file that contains all HTML, CSS, and JavaScript necessary to replicate the UI, content, and interaction features shown. The webpage resolution in the video is {resolution}.

# Instructions:
# 1. Single File Output: All HTML, CSS, and JS must be in one `index.html` file.
# 2. If backend logic is implied, mock it in JS with static data (e.g., a JS array for a fake API call).
# 3. For all clickable elements, please add the class name "btn" in the HTML source code so that the evaluation agent can perform click evaluation. 
# 4. Assets(Images and Videos in the webpage):
#    - All images must use the provided stitched image assets.
#    - The `src` attribute must start with the literal, unchanging string `__PLACEHOLDER_ASSETS_BASE_DIR__/`, followed by the actual filename identified from the stitched image.
#    - For example: `src="__PLACEHOLDER_ASSETS_BASE_DIR__/logo.svg"`.
#    - `<img>` tags must include `width` and `height` attributes.
#    - The provided stitched image assets are before the video.
# 5. No External Dependencies: The generated code must be entirely self-contained. No External Libraries and no External Fonts.
# 6. Final Response: Return **only the complete HTML code** in a single ```html code block, with no additional text or explanations.
# """



PROMPT_TEMPLATE = """
You are an expert front-end developer. Your task is to create a pixel-perfect replica of a website from a chronologically ordered sequence of screenshots showing the website during user interaction.
Generate a single `index.html` file that contains all HTML, CSS, and JavaScript necessary to replicate the UI, content, and interaction features shown. The webpage resolution in the screenshots is {resolution}.

Instructions:
1. Single File Output: All HTML, CSS, and JS must be in one `index.html` file.
2. If backend logic is implied, mock it in JS with static data (e.g., a JS array for a fake API call).
3. For all clickable elements, please add the class name "btn" in the HTML source code so that the evaluation agent can perform click evaluation. 
4. Assets(Images and Videos in the webpage):
   - All images must use the provided stitched image assets.
   - The `src` attribute must start with the literal, unchanging string `__PLACEHOLDER_ASSETS_BASE_DIR__/`, followed by the actual filename identified from the stitched image.
   - For example: `src="__PLACEHOLDER_ASSETS_BASE_DIR__/logo.svg"`.
   - `<img>` tags must include `width` and `height` attributes.
   - The provided stitched image assets are before the screenshot sequence.
5. No External Dependencies: The generated code must be entirely self-contained. No External Libraries and no External Fonts.
6. Final Response: Return **only the complete HTML code** in a single ```html code block, with no additional text or explanations.
"""
