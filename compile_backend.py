import py_compile
import glob

fails = []
for f in glob.glob(r"C:\Users\geeth\OneDrive\Scans\Helmet_Detection_YOLOv8\helmet_plate_project\backend\*.py"):
    try:
        py_compile.compile(f, doraise=True)
    except Exception as e:
        fails.append((f, str(e)))

print('fails:', len(fails))
for f, e in fails:
    print(f)
    print(e)
