import subprocess
import sys

if len(sys.argv) != 3:
    print("Použití: python run_all.py <start_date> <end_date>")
    print("Např.:   python run_all.py 2025-06-01 2025-07-02")
    sys.exit(1)

start_date = sys.argv[1]
end_date = sys.argv[2]

commands = [
    ["python", "DailyAcquisitions.py", "2024-11-26", end_date],
    ["python", "check_srm.py"],
    ["python", "DailyUserAdRevenue.py", start_date, end_date],
    ["python", "DailyUserIAPRevenue.py", start_date, end_date],
    ["python", "GenerateReports.py", start_date, end_date],
    ["python", "UpdateReportsWithRevenue.py", start_date, end_date]
]

for cmd in commands:
    print(f"\nSpouštím: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Chyba při spuštění: {' '.join(cmd)}")
        break
