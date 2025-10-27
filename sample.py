import os
import random
import re
import operator
import argparse
import json
from PIL import Image

# =============================================================================
#
# BENCHMARK DATA GENERATION SCRIPT
#
# This script:
# 1. Reads images from a user-provided --input_directory.
# 2. Applies transformations defined in TRANSFORMATIONS_TO_RUN.
# 3. Saves the output images (input, correct option, incorrect option)
#    to the --output_directory.
# 4. Creates a 'benchmark_data.json' file in the output directory
#    for the main LLaVA script to read.
#
# =============================================================================

# --- Configuration ---
# Define which benchmarks to create and run.
# 'param' is the *correct* transformation.
# 'mc_param' is the *incorrect* one for Option B.
TRANSFORMATIONS_TO_RUN = [
    {
        "type": "2DRotation",
        "param": "+90",
        "mc_param": "180",
        "question": "Which image shows the object rotated by +90 degrees?",
        "ground_truth": "A" # The correct answer is always A (mc_0)
    },
    {
        "type": "2DRotation",
        "param": "180",
        "mc_param": "-90",
        "question": "Which image shows the object rotated by 180 degrees?",
        "ground_truth": "A"
    },
    {
        "type": "2DRotation",
        "param": "-90",
        "mc_param": "+90",
        "question": "Which image shows the object rotated by -90 degrees?",
        "ground_truth": "A"
    }
    # ... You can add other transformation types here ...
]

# --- Transformation Functions ---

angles = ("+90","-90","180")
# Suffixes for the *three* test files we are creating
test_suffixes = (
    f"test_0_input.png",
    f"test_mc_0_input.png", # Correct Option (A)
    f"test_mc_1_input.png"  # Incorrect Option (B)
)

def crop(image_input):
    """Crops an image to a square."""
    if isinstance(image_input, str):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    
    new_size = min(img.size)
    left = (img.width - new_size) / 2
    top = (img.height - new_size) / 2
    right = (img.width + new_size) / 2
    bottom = (img.height + new_size) / 2
    cropped = img.crop((left, top, right, bottom))
    return cropped

def save_image(image_input_list, transformation, param, index, file_suffix, out_directory, base_filename):
    """Saves a list of images to the output directory."""
    saved_paths = []
    for image, suffix in zip(image_input_list, file_suffix):
        cropped_img = crop(image)
        # Create a unique, descriptive filename
        # e.g., "my_image_2DRotation+90_0_test_0_input.png"
        filename = f"{base_filename}_{transformation}{param}_{index}_{suffix}"
        file_path = os.path.join(out_directory, filename)
        
        cropped_img.save(file_path, format="PNG")
        saved_paths.append(file_path)
    return saved_paths

def rotate_image(img_path_or_obj, angle):
    """Rotates an image by a specific angle."""
    if angle not in angles and angle != 0:
        raise ValueError("Invalid Angle")
    if angle == "+90":
        angle_val = -90
    elif angle == "-90":
        angle_val = 90
    elif angle == "180":
        angle_val = 180
    else:
        angle_val = 0
    
    image = None
    if isinstance(img_path_or_obj, str):
        image = Image.open(img_path_or_obj)
    elif isinstance(img_path_or_obj, Image.Image):
        image = img_path_or_obj.copy() # Use copy to avoid issues
    else:
        raise ValueError("Invalid image input for rotate")

    rotated_image = image.rotate(angle_val, expand=False)
    
    if isinstance(img_path_or_obj, str):
        image.close()
        
    return rotated_image

def transform_save_rotate(index, param, inputs_mc, out_directory, transformation, test_image_path):
    """
    Generates and saves a set of rotation benchmark images.
    Returns the file paths for [input, option_a, option_b]
    """
    # 0 = base, param = correct, inputs_mc = incorrect
    base_img_obj = rotate_image(test_image_path, 0)
    correct_img_obj = rotate_image(test_image_path, param)
    incorrect_img_obj = rotate_image(test_image_path, inputs_mc)

    images_to_save = [base_img_obj, correct_img_obj, incorrect_img_obj]
    
    # Get the base filename (e.g., "my_image.png" -> "my_image")
    base_filename = os.path.splitext(os.path.basename(test_image_path))[0]

    # Save the images and get their paths
    saved_paths = save_image(
        images_to_save, 
        transformation, 
        param, 
        index, 
        test_suffixes, 
        out_directory,
        base_filename
    )
    
    # Return the 3 paths we need for the benchmark
    return saved_paths[0], saved_paths[1], saved_paths[2]


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark data for LLaVA.")
    parser.add_argument('--input_directory', type=str, required=True,
                        help='Path to the input directory containing images.')
    parser.add_argument('--output_directory', type=str, required=True,
                        help='Path to the output directory to save benchmark files.')
    args = parser.parse_args()

    print(f"--- PART 1: GENERATING BENCHMARK DATA ---")
    os.makedirs(args.input_directory, exist_ok=True)
    os.makedirs(args.output_directory, exist_ok=True)

    # --- 1. Find Input Images ---
    try:
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        all_files = os.listdir(args.input_directory)
        input_image_paths = [
            os.path.join(args.input_directory, f) for f in all_files 
            if f.lower().endswith(valid_extensions)
        ]
        if not input_image_paths:
            print(f"Error: No valid images found in {args.input_directory}.")
            print("Please add .png or .jpg images to that folder.")
            return
        print(f"Found {len(input_image_paths)} input images.")
    except Exception as e:
        print(f"Error reading input directory: {e}")
        return

    # --- 2. Generate Data ---
    # This list will hold all the benchmark data
    benchmark_data_list = []
    
    file_index = 0 # Unique index for each generated file set
    
    # Loop over every transformation task
    for task in TRANSFORMATIONS_TO_RUN:
        print(f"\nProcessing task: {task['type']} ({task['param']})")
        
        # Loop over every image in the input folder
        for img_path in input_image_paths:
            print(f"  - Transforming {os.path.basename(img_path)}...")
            try:
                if task["type"] == "2DRotation":
                    # Generate the 3 images (input, option_a, option_b)
                    input_path, option_a_path, option_b_path = transform_save_rotate(
                        index=file_index,
                        param=task["param"],
                        inputs_mc=task["mc_param"],
                        out_directory=args.output_directory,
                        transformation=task["type"],
                        test_image_path=img_path
                    )
                    
                    # Add this benchmark item to our list
                    benchmark_data_list.append({
                        "input_image": input_path,
                        "option_image_a": option_a_path,
                        "option_image_b": option_b_path,
                        "question": task["question"],
                        "ground_truth_answer": task["ground_truth"]
                    })
                
                # ... you could add elif task["type"] == "Colour": ... here
                
                file_index += 1
            except Exception as e:
                print(f"    Failed to transform {os.path.basename(img_path)}: {e}")

    # --- 3. Save Benchmark JSON ---
    json_path = os.path.join(args.output_directory, "benchmark_data.json")
    try:
        with open(json_path, 'w') as f:
            json.dump(benchmark_data_list, f, indent=4)
        print(f"\nBenchmark data generation complete! ✅")
        print(f"Saved {len(benchmark_data_list)} benchmark items to {json_path}")
    except Exception as e:
        print(f"\nError saving benchmark JSON file: {e}")


if __name__ == "__main__":
    main()
