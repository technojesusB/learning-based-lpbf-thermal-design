import site
from pathlib import Path

project_root= Path(__file__).resolve().parents[1] # Repo-Root
site_packages = site.getsitepackages() or [site.getusersitepackages()]
pth = Path(site_packages[0]) / "project_root.pth"
pth.write_text(str(project_root)+"\n", encoding="utf-8")
print(f"wrote {pth} -> {project_root}")