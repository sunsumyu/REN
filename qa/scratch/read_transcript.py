import json
import os

transcript_path = r"C:\Users\cf\.gemini\antigravity-ide\brain\54870a0b-7898-486b-9b4e-04623dea9ace\.system_generated\logs\transcript.jsonl"

def read_transcript():
    if not os.path.exists(transcript_path):
        print("Transcript file not found!")
        return
        
    print("Reading transcript lines...")
    with open(transcript_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line)
                step_index = data.get("step_index")
                source = data.get("source")
                type_ = data.get("type")
                content = data.get("content", "")
                
                # We want to print user inputs and planner responses
                if type_ in ["USER_INPUT", "PLANNER_RESPONSE"] or source in ["USER_EXPLICIT", "MODEL"]:
                    print(f"Step {step_index} | Source: {source} | Type: {type_}")
                    print("-" * 50)
                    print(content)
                    print("=" * 50)
            except Exception as e:
                print(f"Error decoding line {idx}: {e}")

if __name__ == "__main__":
    read_transcript()
