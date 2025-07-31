import os
import shutil
import subprocess

def run_inference(input_dir: str, output_dir: str, model_dir: str, use_cpu: bool = False):
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


run_inference(
    input_dir="/home/nimesh.kumar@carpl.local/Data/Code/inference-service-prostrate/workspace/test-data",
    output_dir="output/",
    model_dir="/home/nimesh.kumar@carpl.local/Data/Code/inference-service-prostrate/prostrate-mri/prostate_mri_lesion_seg_app/models",
    use_cpu=True
)