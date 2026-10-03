"""
Single-Record Prediction CLI Script for Manufacturing Predictive Maintenance.
Accepts machine telemetry parameters and outputs model status, probability, risk, and recommendations.
"""
import sys
import argparse
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.prediction import predict_single


def main():
    parser = argparse.ArgumentParser(
        description="Predictive Maintenance Inference for a Single CNC Machine Operating Record."
    )
    parser.add_argument("--type", type=str, default="M", choices=["L", "M", "H"], help="Machine Variant Type (L/M/H)")
    parser.add_argument("--air_temp", type=float, default=298.2, help="Air Temperature [K]")
    parser.add_argument("--process_temp", type=float, default=308.7, help="Process Temperature [K]")
    parser.add_argument("--speed", type=float, default=1408.0, help="Rotational Speed [rpm]")
    parser.add_argument("--torque", type=float, default=46.3, help="Spindle Torque [Nm]")
    parser.add_argument("--tool_wear", type=float, default=3.0, help="Tool Wear [min]")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification Decision Threshold (default: 0.5)")

    args = parser.parse_args()

    input_payload = {
        "Type": args.type,
        "Air temperature [K]": args.air_temp,
        "Process temperature [K]": args.process_temp,
        "Rotational speed [rpm]": args.speed,
        "Torque [Nm]": args.torque,
        "Tool wear [min]": args.tool_wear
    }

    print("\n" + "=" * 70)
    print("CNC MACHINE PREDICTIVE MAINTENANCE INFERENCE")
    print("=" * 70)
    print("INPUT MACHINE OPERATING PARAMETERS:")
    print(f"  - Machine Type:          {input_payload['Type']}")
    print(f"  - Air Temperature:       {input_payload['Air temperature [K]']:.2f} K")
    print(f"  - Process Temperature:   {input_payload['Process temperature [K]']:.2f} K")
    print(f"  - Rotational Speed:      {input_payload['Rotational speed [rpm]']:.0f} rpm")
    print(f"  - Spindle Torque:        {input_payload['Torque [Nm]']:.2f} Nm")
    print(f"  - Tool Wear:             {input_payload['Tool wear [min]']:.1f} min")
    print("-" * 70)

    try:
        result = predict_single(input_payload, threshold=args.threshold)
        
        print("PREDICTION RESULTS:")
        print(f"  - Machine Status:        {result['status_label']} (Class {result['predicted_class']})")
        print(f"  - Failure Probability:   {result['failure_probability_percentage']} (P = {result['failure_probability']})")
        print(f"  - Healthy Probability:   {result['healthy_probability_percentage']} (P = {result['healthy_probability']})")
        print(f"  - Risk Interpretation:   {result['risk_level']} Risk (Decision Threshold: {result['classification_threshold']})")
        print("-" * 70)
        print("MAINTENANCE RECOMMENDATION (Decision Support):")
        print(f"  {result['summary']}")
        
        if result['diagnostic_flags']:
            print("\nDIAGNOSTIC ALERTS:")
            for flag in result['diagnostic_flags']:
                print(f"  [!] {flag}")

        print("\nRECOMMENDED ACTIONS:")
        for action in result['recommended_actions']:
            print(f"  [>] {action}")

        print("\n" + "=" * 70)
        print(f"Model: {result['model_used']}")
        print(f"Disclaimer: {result['disclaimer']}")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"Error during prediction inference: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
