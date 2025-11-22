# to run, ensure that both services are running in the docker engine. 
#In a separate terminal, at the repo_root level, run "python test_fea.py"
import requests

url = "http://localhost:8000/solve_elasticity"

files = {
    "mesh_file": ("mesh.msh", open("mesh.msh", "rb"), "application/octet-stream")
}

data = {
    "E": "210e9",
    "nu": "0.3",
    "Fx": "0",
    "Fy": "0",
    "Fz": "-1000"
}

resp = requests.post(url, data=data, files=files)

print("Status:", resp.status_code)
print("Response:", resp.text)
