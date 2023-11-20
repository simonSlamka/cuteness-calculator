import argparse
from cuteness_calculator import CutenessCalculator

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate the cuteness of a face in an image.")
    parser.add_argument('image_path', help="The path to the image file.")
    parser.add_argument('predictor_path', help="The path to the dlib shape predictor file.")
    args = parser.parse_args()

    # Create a CutenessCalculator object
    cuteness_calculator = CutenessCalculator(args.predictor_path)

    # Calculate the cuteness of the face in the image
    cuteness_score = cuteness_calculator.calculate_cuteness(args.image_path)

    # Print the cuteness score
    print("Cuteness score: {:.2f}".format(cuteness_score))

if __name__ == "__main__":
    main()