from fastapi import FastAPI, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
import shutil
import zipfile
import os
from pathlib import Path
from prostrate_mri.scripts import test_local
import save_inferencing_result
import json
from pathlib import Path
from fastapi.responses import FileResponse




app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = BASE_DIR / "workspace"
UPLOAD_DIR = WORKSPACE_DIR / "uploads"
TESTDATA_DIR = WORKSPACE_DIR / "test-data"
OUTPUT_DIR = WORKSPACE_DIR / "output"
MODEL_DIR = BASE_DIR / "prostrate_mri" / "prostate_mri_lesion_seg_app" / "models"
SCRIPT_PATH = BASE_DIR / "prostrate_mri" / "prostate_mri_lesion_seg_app"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TESTDATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATUS_FILE = BASE_DIR / "job_status.json"
print(STATUS_FILE)
STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
if not STATUS_FILE.exists():
    STATUS_FILE.write_text("{}")

def unzip_to_testdata(job_id: str):
    zip_path = UPLOAD_DIR / f"{job_id}.zip"
    extract_dir = TESTDATA_DIR / job_id
    update_status(job_id, "Processing")
    try:
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"[INFO] Extracted ZIP to: {extract_dir}")
        # 👇 Call run_inference after extraction
        test_local.run_inference(
            input_dir=str(extract_dir),
            output_dir=f"output/",
            model_dir=MODEL_DIR,
            use_cpu=True
        )
        print(f"[INFO] Inference completed for job {job_id}")
        print(f"f[INFO] Saving results for job {job_id}")
        save_inferencing_result.save_segmentation_visualization(
            t2_path=f"output/t2/t2.nii.gz",
            organ_path=f"output/organ/organ.nii.gz",
            lesion_path=f"output/lesion/lesion_mask.nii.gz",
            output_dir=f"{OUTPUT_DIR}/{job_id}"
        )
        update_status(job_id, "Done")
        return "success"
    except Exception as e:
        update_status(job_id, "failed")
        print(f"[ERROR] Failed during unzip or inference for job {job_id}: {e}")
        return "failed"

def update_status(job_id: str, status: str):
    data = json.loads(STATUS_FILE.read_text())
    data[job_id] = status
    STATUS_FILE.write_text(json.dumps(data))

def get_status(job_id: str):
    data = json.loads(STATUS_FILE.read_text())
    print("This is data >>>>>>>>>>>>>>>>>>>>>>>>>>>>", data)
    return data.get(job_id, "not_found")


@app.post("/upload-study/")
async def upload_study(background_tasks: BackgroundTasks, file: UploadFile, jobid: str = Form(...)):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are supported.")

    upload_path = UPLOAD_DIR / f"{jobid}.zip"

    # Save uploaded file to uploads/jobId.zip
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Unzip in the background
    update_status(jobid, "Request Recieved")
    background_tasks.add_task(unzip_to_testdata, jobid)

    return JSONResponse(content={"message": "study upload successfully"}, status_code=200)


@app.get("/job-status/{job_id}")
def check_status(job_id: str):
    status = get_status(job_id)
    return {"job_id": job_id, "status": status}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    result_file = OUTPUT_DIR / job_id / "overlay_segmentation.png"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Result not found.")
    return FileResponse(result_file, media_type="image/png", filename=f"{job_id}_result.png")