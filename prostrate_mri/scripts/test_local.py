import os
import shutil
import subprocess

def run_inference(input_dir: str, output_dir: str, model_dir, use_cpu: bool = False):
    # Validate input directory
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Clean the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Resolve app_dir
    script_dir = os.path.dirname(os.path.realpath(__file__))
    app_dir = os.path.join(script_dir, "..", "prostate_mri_lesion_seg_app")
    print("\n\n\n\n This is the OUTPUT DIR \n ", output_dir)
    # Build the command
    command = ["python", app_dir, "-i", input_dir, "-o", output_dir, "-m", model_dir]

    # Set environment for CPU if required
    env = os.environ.copy()
    if use_cpu:
        print("Running on CPU...")
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        print("Running on GPU...")

    # Run the command
    try:
        subprocess.run(command, check=True, env=env)
        print("Inference completed successfully.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Inference failed with error: {e}")