import google.generativeai as genai
import openai
import base64
from PIL import Image
from io import BytesIO
import os
import time
import random
import concurrent.futures
import dotenv

# --- API Key Setup ---
try:
    dotenv.load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI()
except KeyError as e:
    print("="*50)
    print(f"Error: Environment variable {e} not set.")
    print("Please set BOTH 'GEMINI_API_KEY' and 'OPENAI_API_KEY'")
    print("before running.")
    print(e)
    print("="*50)
    
    exit()

# --- Prompting Constants ---
ARCHETYPES = [
    'warrior', 'mage', 'ranger', 'pilot', 'engineer', 'rogue', 'cleric', 'scientist', 
    'barbarian', 'paladin', 'druid', 'monk', 'assassin', 'bounty hunter', 'artificer', 
    'shaman', 'explorer', 'cyborg', 'alien soldier', 'space marine', 'wizard', 
    'necromancer', 'sorcerer', 'psionicist', 'merchant', 'guard', 'noble'
]
GENDERS = [
    'male', 'female', 'non-binary', 'gender-neutral', 'androgynous', 'agender'
]
RACES = [
    'human', 'elf', 'dwarf', 'orc', 'goblin', 'tiefling', 'dragonborn', 'halfling', 
    'gnome', 'alien', 'robot', 'android', 'cyborg', 'cat-person', 'lizard-person', 
    'avian-humanoid', 'insectoid', 'plant-based humanoid'
]
GEAR = [
    'heavy plate armor', 'light leather armor', 'ornate mystical robes', 
    'practical adventuring gear', 'futuristic combat suit', 'post-apocalyptic scavenger gear', 
    'rags and simple clothes', 'silken noble garb', 'studded leather armor', 
    'chainmail', 'scientist lab coat', 'high-tech stealth suit', 'barbarian furs'
]
BACKGROUNDS = [
    'a simple neutral gray background', 'a misty forest', 'a futuristic city street', 
    'a desert dune', 'a mountain pass', 'a spaceship interior', 'a ruined temple', 
    'a plain white studio background', 'a dark cave', 'a magical library'
]
SKIN_COLORS = [
    'pale white skin', 'fair skin', 'lightly tanned skin', 'olive skin', 
    'medium brown skin', 'dark brown skin', 'deep black skin', 'ebony skin',
    'atypical blue skin', 'atypical green skin', 'atypical red skin', 
    'metallic bronze skin', 'gray skin', 'ashen skin'
]

# Define the models we can choose from
MODEL_CHOICES = ['gemini', 'dall-e']

def create_dynamic_prompt():
    """Builds a unique prompt string from the random lists."""
    archetype = random.choice(ARCHETYPES)
    gender = random.choice(GENDERS)
    ancestry = random.choice(RACES)
    gear = random.choice(GEAR)
    background = random.choice(BACKGROUNDS)
    skin_color = random.choice(SKIN_COLORS) 
    
    prompt = f"""
    Generate a high-resolution, detailed concept art portrait for a character dataset.

    - **SUBJECT:** A single {gender} {ancestry} {archetype}.
    - **FRAMING:** Extreme close-up portrait of the character's head only. 
    - **CROP:** Tightly cropped to the face and head. Shoulders must NOT be visible.
    - **APPEARANCE:** The character has {skin_color}. The {gear} is only visible if it is part of a helmet, collar, or head/neck covering.
    - **BACKGROUND:** Simple or out-of-focus {background}.
    - **STYLE:** Digital painting, highly detailed, unique, and professional concept art.
    - **INTENT:** Create a unique, SFW character portrait.
    """
    return prompt.strip()



def generate_with_gemini(prompt, iteration_num, output_dir):
    """Generates and saves a single image using the Gemini API."""
    
    print(f"[Thread {iteration_num:04d} (Gemini)]: Starting generation...")
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-image') 
        response = model.generate_content(prompt)
        
        image_found = False
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    image_found = True
                    image_data = part.inline_data.data
                    image_bytes = BytesIO(image_data)
                    img = Image.open(image_bytes)
                    
                    filename = f"character_gemini_{iteration_num:04d}.png"
                    filepath = os.path.join(output_dir, filename)
                    
                    img.save(filepath)
                    print(f"[Thread {iteration_num:04d} (Gemini)]: SUCCESS! Saved as: {filepath}")
                    return True # Indicate success
        
        if not image_found:
            print(f"[Thread {iteration_num:04d} (Gemini)]: ERROR! Model did not return an image.")
            try:
                print(f"Model response text: {response.text}")
            except Exception:
                print(f"Model response: {response}")
            return False 

    except Exception as e:
        print(f"[Thread {iteration_num:04d} (Gemini)]: FATAL ERROR! {e}")
        return False

def generate_with_openai(prompt, iteration_num, output_dir):
    print(f"DALL-E Starting generation for prompt {prompt}...")
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="b64_json"
        )
        if response and response.data and response.data[0].b64_json:
            image_b64 = response.data[0].b64_json
            image_bytes = BytesIO(base64.b64decode(image_b64))
            img = Image.open(image_bytes)
            filename = f"character_dalle_{iteration_num:04d}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            print(f"DALL-E SUCCESS! Saved as {filepath}")
            return True
        else:
            print(f"DALL-E ERROR! API did not return image data.")
            return False
    except Exception as e:
        print(f"DALL-E FATAL ERROR! {e}")
        return False

def generate_image_wrapper(model_choice, prompt, iteration_num, output_dir):
    """
    Selects the correct generation function based on model_choice.
    This is the function the thread pool will execute.
    """
    if model_choice == 'gemini':
        return generate_with_gemini(prompt, iteration_num, output_dir)
    elif model_choice == 'dall-e':
        return generate_with_openai(prompt, iteration_num, output_dir)
    else:
        print(f"[Thread {iteration_num:04d}]: ERROR! Unknown model choice '{model_choice}'")
        return False


if __name__ == "__main__":
    
    TOTAL_IMAGES_TO_GENERATE = 4000
    OUTPUT_DIRECTORY = "images" 
    MAX_CONCURRENT_REQUESTS = 10 
    
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
        print(f"Created output directory: {OUTPUT_DIRECTORY}")

    start_time = time.time()
    print(f"Starting parallel generation for {TOTAL_IMAGES_TO_GENERATE} images...")
    print(f"Randomly alternating between {MODEL_CHOICES}")
    print(f"Using up to {MAX_CONCURRENT_REQUESTS} parallel requests.")
    print("WARNING: Monitor for '429 Too Many Requests' errors. If you see them, lower MAX_CONCURRENT_REQUESTS.")

    
    jobs_submitted = 0
    jobs_completed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        
        future_to_iteration = {}
        for i in range(1, TOTAL_IMAGES_TO_GENERATE + 1):
            
            dynamic_prompt = create_dynamic_prompt()
            model_to_use = random.choice(MODEL_CHOICES)
            
            future = executor.submit(
                generate_image_wrapper, 
                model_to_use, 
                dynamic_prompt, 
                i, 
                OUTPUT_DIRECTORY
            )
            future_to_iteration[future] = i
            jobs_submitted += 1

        print(f"All {jobs_submitted} jobs have been submitted to the pool.")
        print("Waiting for workers to complete...")

        
        for future in concurrent.futures.as_completed(future_to_iteration):
            iteration_num = future_to_iteration[future]
            try:
                success = future.result() 
                if success:
                    jobs_completed += 1
            except Exception as e:
                print(f"Main thread caught an error from iteration {iteration_num}: {e}")
            
            if jobs_completed > 0 and jobs_completed % 50 == 0:
                elapsed = time.time() - start_time
                print(f"--- PROGRESS: {jobs_completed}/{TOTAL_IMAGES_TO_GENERATE} complete. (Time: {elapsed:.0f}s) ---")
        
    end_time = time.time()
    print("--- Generation Complete ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")
    print(f"Successfully generated {jobs_completed} / {TOTAL_IMAGES_TO_GENERATE} images.")