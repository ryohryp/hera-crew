import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from hera_crew.crew import HeraCrew

def quick_test():
    inputs = {'user_request': "2+2は?"}
    try:
        result = HeraCrew().crew().kickoff(inputs=inputs)
        print("\n\nRESULT:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Add src to path
    sys.path.append(os.path.join(os.getcwd(), "src"))
    quick_test()
