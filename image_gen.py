import google.generativeai as genai
from PIL import Image
from io import BytesIO
import os
import time
import random
import concurrent.futures  


try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("="*50)
    print("Error: GEMINI_API_KEY environment variable not set.")
    print("Please set the environment variable before running.")
    print("="*50)
    exit()


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


def create_dynamic_prompt():
    """Builds a unique prompt string from the random lists."""
    archetype = random.choice(ARCHETYPES)
    gender = random.choice(GENDERS)
    race = random.choice(RACES)
    gear = random.choice(GEAR)
    background = random.choice(BACKGROUNDS)
    
    prompt = f"""
    Full-body concept art of a single {gender} {race} {archetype}, 
    in a digital painting concept art style.
    The character is wearing {gear}.
    The pose is a full-body standing pose, neutral or confident.
    The background is {background} to keep focus on the character.
    Ensure the character is highly detailed and unique, suitable for a dataset.
    """
    return prompt


def generate_image(prompt, iteration_num, output_dir):
    """Generates and saves a single image. This function will be called by a worker thread."""
    
    print(f"[Thread for {iteration_num:04d}]: Starting generation...")
    
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
                    
                    filename = f"character_{iteration_num:04d}.png"
                    filepath = os.path.join(output_dir, filename)
                    
                    img.save(filepath)
                    print(f"[Thread for {iteration_num:04d}]: SUCCESS! Saved as: {filepath}")
                    return True # Indicate success
        
        if not image_found:
            print(f"[Thread for {iteration_num:04d}]: ERROR! Model did not return an image.")
            try:
                print(f"Model response text: {response.text}")
            except Exception:
                print(f"Model response: {response}")
            return False 

    except Exception as e:
        print(f"[Thread for {iteration_num:04d}]: FATAL ERROR! {e}")
        return False 


if __name__ == "__main__":
    
    
    TOTAL_IMAGES_TO_GENERATE = 4000
    OUTPUT_DIRECTORY = "ai_images" 
    
    
    MAX_CONCURRENT_REQUESTS = 10 
    
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
        print(f"Created output directory: {OUTPUT_DIRECTORY}")

    start_time = time.time()
    print(f"Starting parallel generation for {TOTAL_IMAGES_TO_GENERATE} images...")
    print(f"Using up to {MAX_CONCURRENT_REQUESTS} parallel requests.")
    print("WARNING: Monitor for '429 Too Many Requests' errors. If you see them, lower MAX_CONCURRENT_REQUESTS.")

    
    jobs_submitted = 0
    jobs_completed = 0
    
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        
        
        future_to_iteration = {}
        for i in range(1, TOTAL_IMAGES_TO_GENERATE + 1):
            
            dynamic_prompt = create_dynamic_prompt()
            
            
            future = executor.submit(generate_image, dynamic_prompt, i, OUTPUT_DIRECTORY)
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